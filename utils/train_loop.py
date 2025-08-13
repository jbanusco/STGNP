import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import shutil
from pathlib import Path
import datetime
import logging
# import wandb
from torch.utils.tensorboard import SummaryWriter

from utils.train_utils import load_checkpoint, save_checkpoint
from utils.plot_utils import save_mean_trajectories, plot_latent_trajectories, fig_to_tensorboard


def update_encoding_state(model_input, state=False):
    for param in model_input.embedding_nodes.parameters():
        param.requires_grad = state

    for param in model_input.context_encoder.parameters():
        param.requires_grad = state
    
    if model_input.hidden_dim_ext > 0:
        for param in model_input.embedding_ext.parameters():
            param.requires_grad = state

    for param in model_input.embedding_edges_space.parameters():
        param.requires_grad = state

    for param in model_input.embedding_edges_time.parameters():
        param.requires_grad = state

    for param in model_input.L_r_to_mu_sigma.parameters():
        param.requires_grad = state
    
    for param in model_input.D_r_to_mu_sigma.parameters():
        param.requires_grad = state

    for param in model_input.Espace_to_mu_sigma.parameters():
        param.requires_grad = state

    for param in model_input.Etime_to_mu_sigma.parameters():
        param.requires_grad = state


def update_decoder_state(model_input, state=False):    
    for param in model_input.decoder.parameters():
        param.requires_grad = state

    for param in model_input.hidden_to_mean.parameters():
        param.requires_grad = state

    for param in model_input.hidden_to_sigma.parameters():
        param.requires_grad = state


def update_multiplex_state(model_input, state=False):    
    for param in model_input.stgcn.parameters():
        param.requires_grad = state


def update_classifier_state(model_input, state=False):
    for param in model_input.dyn_classifier.parameters():
        param.requires_grad = state

    # for param in model_input.context_encoder.parameters():
    #     param.requires_grad = state


def add_trajectories_to_tensorboard(writer, output, epoch, mode='Train', is_prediction=False):
    if is_prediction:
        p_y_pred = output[0]
        latent_rec = output[1]
        tgt_data = output[-1]
        latent_numpy_train = latent_rec.float().cpu().detach().numpy()
        append_name = 'Predict '
    else:
        p_y_pred = output[1]
        latent_rec = output[2]
        tgt_data = output[-1]
        latent_numpy_train = latent_rec.float().cpu().detach().numpy()
        append_name = ''

    rec_ft = p_y_pred.mean.float()
    rec_std = p_y_pred.scale.float()

    latent_fig = plot_latent_trajectories(latent_numpy_train)
    pred_fig = save_mean_trajectories(rec_ft, tgt_data)

    # Convert to TensorBoard format
    writer.add_image(f"{append_name}Latent Trajectories/{mode}", fig_to_tensorboard(latent_fig), global_step=epoch, dataformats="HWC")
    writer.add_image(f"{append_name}Mean Predicted vs Target/{mode}", fig_to_tensorboard(pred_fig), global_step=epoch, dataformats="HWC")

    # Close figures
    plt.close(latent_fig)
    plt.close(pred_fig)


def logging_setup_train(run_log_folder):
    # Set up logging
    log_file = os.path.join(run_log_folder, "training.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers (to avoid duplicate logs)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Define logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info('Logging to file and console...')


def train_model(model: torch.nn.Module, 
                epochs: int,  
                optimizer: torch.optim,  
                criterion: torch.nn.modules.loss, 
                train_dataloader: DataLoader,          
                test_dataloader: DataLoader,    
                val_dataloader: DataLoader = None,      
                device: torch.cuda.device = 'cpu',
                save_folder: str = None,
                reload_model: bool = False,
                scheduler: torch.optim.lr_scheduler = None,
                early_stop: bool = False,
                early_stop_patience: int = 10,
                tolerance: float = 1e-4,
                print_epoch = 1,
                lr_schedule_patience = 15,
                batch_loop: callable = None,
                save_model: bool = True,
                model_to_load: str = None,
                fine_tune_model: bool = False,
                **kwargs,
                ) -> dict:
    """
    Train the model.
    """    
        
    track_experiment = kwargs.get('track_experiment', False)
    project_name = kwargs.get('project_name', 'stmgcn')    
    hyperparameters = kwargs.get('hyperparams', None)
    use_optuna = kwargs.get('use_optuna', False)
    error_score = kwargs.get('error_score', torch.FloatTensor([-np.inf]).to(device))
    metrics_used = kwargs.get('metrics_used', 'accuracy')
    # Store if we want to output probabilities at the end
    output_probs = kwargs.get('output_probs', False)
    # Set output_probs to False to prevent error during training
    kwargs['output_probs'] = False
    kwargs['predict_beyond'] = True

    model.to(device)
    criterion.to(device)

    # logging.getLogger().setLevel(logging.INFO)
    parent_log_folder = os.path.join(Path(save_folder).parent, 'logs')
    os.makedirs(parent_log_folder, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"experiment_{run_id}"
    run_log_folder = os.path.join(parent_log_folder, run_name)
    os.makedirs(run_log_folder, exist_ok=True)
    logging_setup_train(run_log_folder)

    if track_experiment and not use_optuna:
        logging.info('Logging to wandb...\n')                
        writer = SummaryWriter(log_dir=run_log_folder)
        if hyperparameters is not None:
            writer.add_hparams(hyperparameters, {'hparam/score': 0.0}, run_name=run_name)
            # writer.add_hparams(hyperparameters, {'hparam/score': 0.0})

    # Verify early stop patience minimum 1/3 of the epochs
    if early_stop:
        if early_stop_patience < int(epochs/3):
            early_stop_patience = int(epochs/3)
            logging.info(f'Early stop patience too high. Setting to 1/3 of the epochs: {early_stop_patience}\n')


    # Just to check if the model is on the right device
    if device == 'cuda':
        for name, param in model.named_parameters():
            if param.device.type != "cuda":
                print(f"Param {name} is on {param.device}")

    # ===========================================
    # ============== Setup options ==============
    if scheduler is None:
        lr_reduce_factor = 0.6
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_reduce_factor, patience=lr_schedule_patience)

    checkpoint_filename = os.path.join(save_folder, 'checkpoint.pt')
    model_filename = os.path.join(save_folder, 'model.pt')
    tmp_best_filename = os.path.join(save_folder, 'tmp_best.pt')
    if save_model:
        logging.info(f'Saving data to {save_folder}\n')
    else:
        logging.info('Not saving data.\n')        
        save_folder = os.path.join(os.getcwd(), 'tmp') if save_folder is None else save_folder
        os.makedirs(save_folder, exist_ok=True)

    # ===========================================
    # ============== Load previous model ========
    if model_to_load is not None:
        # Load this model 
        print(f'Loading model {model_to_load}...\n')
        logging.info(f'Loading model {model_to_load}...\n')
        checkpoint = torch.load(model_to_load)
        model.load_state_dict(checkpoint['model_state_dict'])
        reload_model = False
        epochs = epochs if fine_tune_model else 0

    if reload_model and os.path.exists(model_filename):
        print('Loading trained model...\n')
        logging.info('Loading trained model...\n')
        init_epoch, best_epoch, best_score, losses, metrics = load_checkpoint(model_filename, model, optimizer, scheduler, device)
        losses_train, losses_valid, losses_test = losses        
        track_metrics_train, track_metrics_valid, track_metrics_test = metrics
        init_epoch = epochs
        logging.info('Trained model loaded successfully.\n')
        print('Trained model loaded successfully.')

    elif reload_model and os.path.exists(checkpoint_filename):
        print('Loading checkpoint...\n')
        logging.info('Loading checkpoint...\n')
        init_epoch, best_epoch, best_score, losses, metrics = load_checkpoint(checkpoint_filename, model, optimizer, scheduler, device)
        losses_train, losses_valid, losses_test = losses
        track_metrics_train, track_metrics_valid, track_metrics_test = metrics
        last_checkpoint = init_epoch
        logging.info('Checkpoint loaded successfully.\n')
        print('Checkpoint loaded successfully.\n')

    else:
        logging.info('No previous model found. Starting from scratch.')
        best_score = -np.inf
        best_epoch = 0
        init_epoch = 0
        last_checkpoint = 0

        # losses_train, losses_valid, losses_test = [], [], []
        losses_train = torch.zeros(epochs, device='cpu')
        losses_valid = torch.zeros(epochs, device='cpu')
        losses_test = torch.zeros(epochs, device='cpu')

        track_metrics_train = {
            'accuracy': torch.zeros(epochs, device='cpu'),
            'mse': torch.zeros(epochs, device='cpu')
        }
        track_metrics_valid = {
            'accuracy': torch.zeros(epochs, device='cpu'),
            'mse': torch.zeros(epochs, device='cpu')
        }
        track_metrics_test = {
            'accuracy': torch.zeros(epochs, device='cpu'),
            'mse': torch.zeros(epochs, device='cpu')
        }
        
    # ===========================================
    # ================== Train ==================
    ramp_epochs = 20
    failed = False
    kwargs['apply_penalties'] = False
    kwargs['scaler'] = None
    warmup_epochs = kwargs.get('warmup_epochs', 20)
    
    for epoch in range(init_epoch, epochs):        
        if epoch > warmup_epochs:
            kwargs['apply_penalties'] = True
            kwargs['warmup'] = False
            rampup_weight = min(1.0, np.abs(epoch - warmup_epochs) / ramp_epochs)
        elif epoch == warmup_epochs:
            kwargs['apply_penalties'] = True
            kwargs['warmup'] = False
            rampup_weight = min(1.0, np.abs(epoch - warmup_epochs) / ramp_epochs)
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
                new_lr = current_lr * 0.5  # or any decay factor
                param_group['lr'] = new_lr
        else:
            kwargs['warmup'] = True
            rampup_weight = 1.0  # No ramp-up during warmup
        kwargs['rampup_weight'] = rampup_weight        
        kwargs['epoch'] = epoch

        # ====================
        # ===== Training ===== 
        loss_train, metrics_train, output_train, outout_pred_train = batch_loop(model, train_dataloader, criterion, optimizer, 
                                                                                device, train_model=True, **kwargs)
        
        if track_experiment and epoch % print_epoch == 0:
            # Histograms of weights and gradients
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(name, param, epoch)
                    try:
                        if param.grad is not None:
                            writer.add_histogram(f'{name}_grad', param.grad, epoch)
                        else:
                            print(f"No gradient for parameter: {name}")
                    except Exception as e:
                        print(f"Error logging gradient for parameter: {name}")
                        # print(e)

        # =====================================
        # ===== Validation (if available) =====
        if val_dataloader is not None:
            with torch.no_grad():
                loss_val, metrics_val, output_val, outout_pred_val = batch_loop(model, val_dataloader, criterion, optimizer, 
                                                                                device, train_model=False, **kwargs)

            # Compute score (for hyperparameter tuning) and model saving
            score = -np.log(metrics_val['mae_pred'])
            
        else:
            output_val = None
            outout_pred_val = None
            loss_val = np.nan            
            metrics_val = {'accuracy': np.nan, 'mse': np.nan}

            # Compute score (for hyperparameter tuning) and model saving
            score = -np.log(metrics_train['mae_pred'])
            
        with torch.no_grad():
            if not kwargs.get('warmup', False):
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # If using ReduceLROnPlateau, step with the validation loss
                    scheduler.step(loss_val)
                else:
                    scheduler.step()

        # ===================
        # ===== Testing =====
        if test_dataloader is not None:
            with torch.no_grad():                            
                loss_test, metrics_test, output_test, outout_pred_test = batch_loop(model, test_dataloader, criterion, optimizer, 
                                                                                    device, train_model=False, **kwargs)
        else:
            output_test = None
            outout_pred_test = None
            loss_test = np.nan
            metrics_test = {'accuracy': np.nan, 'mse': np.nan}

        # ================================================================
        # =================== Log and save tmp files =====================
        with torch.no_grad():
            losses_train[epoch] = loss_train
            losses_valid[epoch] = loss_val
            losses_test[epoch] = loss_test
            track_metrics_train['accuracy'][epoch] = metrics_train['accuracy']
            track_metrics_valid['accuracy'][epoch] = metrics_val['accuracy']
            track_metrics_test['accuracy'][epoch] = metrics_test['accuracy']
            if 'mse' in metrics_train:
                track_metrics_train['mse'][epoch] = metrics_train['mse']
                track_metrics_valid['mse'][epoch] = metrics_val['mse']
                track_metrics_test['mse'][epoch] = metrics_test['mse']

        acc_train = track_metrics_train['accuracy'][epoch]
        acc_valid = track_metrics_valid['accuracy'][epoch]
        acc_test = track_metrics_test['accuracy'][epoch]
        if track_experiment:
            # Trajectories to tensorboard
            if epoch % print_epoch == 0:
                with torch.no_grad():
                    add_trajectories_to_tensorboard(writer, output_train, epoch, mode='Train', is_prediction=False)
                    if output_val is not None:
                        add_trajectories_to_tensorboard(writer, output_val, epoch, mode='Validation', is_prediction=False)
                    if output_test is not None:
                        add_trajectories_to_tensorboard(writer, output_test, epoch, mode='Test', is_prediction=False)

                    if kwargs.get('predict_beyond', False):
                        # Predict beyond
                        add_trajectories_to_tensorboard(writer, outout_pred_train, epoch, mode='Train_Predict', is_prediction=True)
                        if outout_pred_val is not None:
                            add_trajectories_to_tensorboard(writer, outout_pred_val, epoch, mode='Validation_Predict', is_prediction=True)  
                        if outout_pred_test is not None:
                            add_trajectories_to_tensorboard(writer, outout_pred_test, epoch, mode='Test_Predict', is_prediction=True)

            # Learning rate            
            for i, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(f'LR/param_group_{i}', param_group['lr'], epoch)

            # Losses
            for key, avg_val in metrics_train.items():
                writer.add_scalar(f'Losses_Train/{key}', avg_val, epoch)

            if val_dataloader is not None:
                for key, avg_val in metrics_val.items():
                    writer.add_scalar(f'Losses_Val/{key}', avg_val, epoch)

            if test_dataloader is not None:
                for key, avg_val in metrics_test.items():
                    writer.add_scalar(f'Losses_Test/{key}', avg_val, epoch)

        if epoch % print_epoch == 0:
            str_msg = f'==== Epoch [{epoch}/{epochs}] ====\n' \
            f'Training:\nLoss: {losses_train[epoch]:.4f}\tAccuracy: {acc_train:.3f}\n' \
            f'Validation:\nLoss: {losses_valid[epoch]:.4f}\tAccuracy: {acc_valid:.4f}\n' \
            f'Test:\nLoss: {losses_test[epoch]:.4f}\tAccuracy: {acc_test:.4f}\n' \
            f'MAE: {metrics_train["mae"]:.4f}\tMSE: {metrics_train["mse"]:.4f}\n' \
            f'MAE pred: {metrics_train["mae_pred"]:.4f}\tMSE pred: {metrics_train["mse_pred"]:.4f}\n' \
            f'Best Score: {best_score:.4f}\tScore: {score:.4f}\n' \
            f'LR: {scheduler.get_last_lr()}\n'
            # print(str_msg)
            logging.info(str_msg)

        if epoch % 25 == 0:
            last_checkpoint = epoch
            # Save checkopoint
            add_dict = {'losses_train': losses_train, 
                        'losses_valid': losses_valid,
                        'losses_test': losses_test,
                        'metrics_train': track_metrics_train, 
                        'metrics_valid': track_metrics_valid,
                        'metrics_test': track_metrics_test,
                        'best_score': best_score, 'best_epoch': best_epoch,
                        'epoch': epoch}
            save_checkpoint(checkpoint_filename, model, optimizer, scheduler, add_dict)

        if torch.isnan(losses_train[-1]) or torch.isinf(losses_train[-1]) or torch.isneginf(losses_train[-1]):
            print("ERROR DURING TRAINING. NaN or INF loss, stopping training")
            logging.info("NaN loss, stopping training")
            failed = True
            break
        
        if best_score <= score:
            # If this condition is met, we save the model otherwise, pass
            save_best_model = True
            if best_score == score and losses_valid[-1] > losses_valid[best_epoch]:
                save_best_model = False
            if epoch < int(warmup_epochs):
                # Wait until the warmup is over
                save_best_model = False  

            if save_best_model:                            
                # Save temporary best model       
                logging.info(f'NEW BEST SCORE! - Saving temporary best model at epoch {epoch}')
                best_score = score
                best_epoch = epoch
                add_dict = {'losses_train': losses_train, 
                            'losses_valid': losses_valid,
                            'losses_test': losses_test,
                            'metrics_train': track_metrics_train, 
                            'metrics_valid': track_metrics_valid,
                            'metrics_test': track_metrics_test,
                            'best_score': best_score, 'best_epoch': best_epoch,
                            'epoch': epoch}
                save_checkpoint(tmp_best_filename, model, optimizer, scheduler, add_dict)

                if epoch > last_checkpoint:
                    shutil.copyfile(tmp_best_filename, checkpoint_filename)

            if failed:
                break

        # Check early stop
        if early_stop and epoch > early_stop_patience:
            if np.sum(np.abs(np.diff(losses_train)[-early_stop_patience:]) < tolerance) == early_stop_patience:
                print("Convergence reached")
                logging.info("Convergence reached")
                break            

    # ===========================================
    # ============== Save final model ===========
    if failed:
        res_data = {'losses_train': losses_train,
                    'losses_valid': losses_valid,
                    'losses_test': losses_test,
                    'metrics_train': track_metrics_train, 
                    'metrics_valid': track_metrics_valid,
                    'metrics_test': track_metrics_test,
                    'best_epoch': 0,
                    'best_score': error_score,
                    }
        return res_data

    if os.path.isfile(tmp_best_filename):
        # Take the tmp best model and save it as the final model
        logging.info('Saving final model...')
        checkpoint = torch.load(tmp_best_filename)
        epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']  # Should be the same in this case
        best_score = checkpoint['best_score']
        model.load_state_dict(checkpoint['model_state_dict'])

        add_dict = {'losses_train': losses_train, 
                    'losses_valid': losses_valid,
                    'losses_test': losses_test,
                    'metrics_train': track_metrics_train, 
                    'metrics_valid': track_metrics_valid,
                    'metrics_test': track_metrics_test,
                    'best_score': best_score, 'best_epoch': best_epoch,
                    'epoch': epoch}
        save_checkpoint(model_filename, model, optimizer, scheduler, add_dict)
        os.remove(tmp_best_filename)  # Remove tmp best model
        logging.info('Final model saved successfully.')

    if os.path.isfile(checkpoint_filename):
        logging.info('Removing checkpoints...')
        os.remove(checkpoint_filename)  # Remove checkpoint

    # ======================================
    # ===== Load and check best model ======

    print('Model trained!')
    best_point = torch.load(model_filename, map_location=torch.device(device))
    model.load_state_dict(best_point['model_state_dict'])
    best_epoch = best_point['best_epoch']
    best_score = best_point['best_score']
    if failed:        
        best_score = error_score  # This is to avoid using this model (a very bad score, assuming we want to maximize the score)

    best_acc_train = track_metrics_train['accuracy'][best_epoch]
    best_acc_valid = track_metrics_valid['accuracy'][best_epoch]
    best_acc_test = track_metrics_test['accuracy'][best_epoch]
    str_msg = f'==== Best epoch: {best_epoch} [{best_epoch}/{epochs}] ====\n' \
    f'Training:\nLoss: {losses_train[best_epoch]:.4f}\tAccuracy: {best_acc_train:.4f}\n' \
    f'Validation:\nLoss: {losses_valid[best_epoch]:.4f}\tAccuracy: {best_acc_valid:.4f}\n' \
    f'Test:\nLoss: {losses_test[best_epoch]:.4f}\tAccuracy: {best_acc_test:.4f}\n' \
    f'Best Score: {best_score:.4f}\n'

    print(str_msg)
    logging.info(str_msg)
    
    res_data = {
        'losses_train': losses_train.cpu().numpy(),
        'losses_valid': losses_valid.cpu().numpy(),
        'losses_test': losses_test.cpu().numpy(),
        'metrics_train': {k: v.cpu().numpy() for k, v in track_metrics_train.items()},
        'metrics_valid': {k: v.cpu().numpy() for k, v in track_metrics_valid.items()},
        'metrics_test': {k: v.cpu().numpy() for k, v in track_metrics_test.items()},
        'best_epoch': best_epoch,
        'best_score': best_score,
    }

    # Finish wandb run
    if track_experiment:
        writer.flush()
        writer.close()

    if output_probs:
        logging.info('Getting test probabilities...')
        kwargs['output_probs'] = True
        # Using the best model
        with torch.no_grad():            
            _, _, decisions, probs = batch_loop(model, test_dataloader, criterion, optimizer, device, 
                                                train_model=False, **kwargs)
        res_data['probs'] = probs
        res_data['decisions'] = decisions

    if not save_model:
        logging.info('Not saving model. Returning results only and deleting the model.')
        # Remove the save folder
        shutil.rmtree(save_folder)
        # os.remove(model_filename)

    return res_data