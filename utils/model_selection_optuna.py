
import os
import logging
import torch
import numpy as np
import pandas as pd
import pickle
import joblib
import sys
import sqlite3
import socket
import multiprocessing
multiprocessing.set_start_method("fork", force=True)
from sklearn.model_selection import train_test_split
from utils.model_selection_sklearn import cv_stratified_split
import optuna


def get_postgres_db_url():
    """
    Automatically detect if we are running locally or remotely,
    and return the appropriate PostgreSQL database URL.
    """
    LOCAL_HOSTNAME = "hos70292"  # Change to your local machine's hostname
    LOGIN_NODE_HOSTNAME = "login"  # Change to your cluster's login node hostname
    # POSTGRES_DB_URL = "postgresql://postgres:optuna@localhost:5432/optuna_db"
    # storage_name = POSTGRES_DB_URL

    # Default credentials (must match your PostgreSQL setup)
    DB_USER = "postgres"
    DB_PASS = "optuna"
    DB_NAME = "optuna_db"
    DB_PORT = 5432  # Default PostgreSQL port

    # Determine the correct database host
    current_hostname = socket.gethostname()
    if current_hostname == LOCAL_HOSTNAME:
        DB_HOST = "localhost"
        logging.info("Running locally, connecting to PostgreSQL on localhost.")
    elif current_hostname == LOGIN_NODE_HOSTNAME:
        # DB_HOST = LOGIN_NODE_HOSTNAME  # Use the login node if available
        logging.info(f"Running on the cluster login node: {LOGIN_NODE_HOSTNAME}")
        DB_HOST = "155.105.223.17"  # Use the IP address of the cluster login node
    else:
        # Assume we are on a compute node and PostgreSQL is on the login node
        # DB_HOST = LOGIN_NODE_HOSTNAME
        DB_HOST = "155.105.223.17"  # Use the IP address of the cluster login node
        logging.info(f"Running remotely on {current_hostname}, connecting to PostgreSQL on {LOGIN_NODE_HOSTNAME}")

    return f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def hypertune_optuna(objective,
                     save_folder,
                     study_name,
                     **kwargs):
    """Implement hyperparameter tuning with Optuna"""
    max_cpus = kwargs.get('max_cpus', 1)
    available_cpus = multiprocessing.cpu_count()
    num_cpus = min(available_cpus, max_cpus)
    objective.num_jobs = num_cpus
    print(f"Avaliable CPUs: {available_cpus}, Using {num_cpus} CPUs, Max CPUs: {max_cpus}, Jobs: {objective.num_jobs}")

    # Sqlite or Postgres
    sq_database = kwargs.get('sq_database', False)

    # Get the optuna logger
    log_filename = os.path.join(save_folder, f'{study_name}.log')

    # Create the folder and the filename
    os.makedirs(save_folder, exist_ok=True)

    optuna.logging.get_logger("optuna").setLevel(logging.INFO)
    # If there is already a handler to the sys.stdout skip this one
    if not any([isinstance(handler, logging.StreamHandler) for handler in optuna.logging.get_logger("optuna").handlers]):
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    # Remove prevoius handlers
    for handler in optuna.logging.get_logger("optuna").handlers:
        if isinstance(handler, logging.FileHandler):
            optuna.logging.get_logger("optuna").removeHandler(handler)
    optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(log_filename))    
    
    # Get the names
    if sq_database:
        db_path = os.path.join(save_folder, f"{study_name}.db")

        # Enable WAL mode
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.close()

        # Use SQLite with WAL mode
        storage_name = f"sqlite:///{db_path}?cache=shared"
    else:
        storage_name = get_postgres_db_url()        


    study_filename = os.path.join(f"{save_folder}", f"{study_name}.pkl")
    objective.study_name = study_name

    num_trials = kwargs.get('num_trials', 10)
    load_previous = kwargs.get('load_previous', False)

    # Optuna sampler and study creation
    sampler_filename = os.path.join(f"{save_folder}", "sampler_optuna.pkl")
    if sq_database and os.path.isfile(sampler_filename):
        sampler = pickle.load(open(sampler_filename, "rb"))    
        logging.info(f'Loaded sampler from {sampler_filename}')
    else:
        if sq_database:
            # Make the sampler behave in a deterministic way.
            sampler = optuna.samplers.TPESampler(seed=10)
        else:
            sampler = optuna.samplers.TPESampler()
        logging.info('Created new sampler')
    
    # Create the study
    if os.path.isfile(os.path.join(save_folder, f'{study_name}.db')) and not load_previous and sq_database:
        os.remove(os.path.join(save_folder, f'{study_name}.db'))
        logging.info(f'Removed previous study {study_name}.db')

    try:
        study = optuna.create_study(direction=objective.direction, 
                                    sampler=sampler, 
                                    study_name=f'{study_name}', 
                                    storage=storage_name, 
                                    load_if_exists=load_previous)
    except Exception as e:
        logging.error(f'Error creating study: {e}')
        raise

    try:
        last_trial = study.trials[-1].number + 1
    except IndexError:
        last_trial = 0
    
    if sq_database:
        n_trials = num_trials - last_trial
    else:
        n_trials = num_trials  # If we are using Postgres, we don't know the number of trials, so just run the number of trials requested
    logging.info(f'Last trial: {last_trial}\nNumber of trials: {n_trials}')
        
    if n_trials > 0:
        logging.info(f'Optimizing {n_trials} trials\n')
        study.optimize(objective, n_trials=n_trials, n_jobs=num_cpus, timeout=7200)  # 2 hours
        
    if not sq_database:
        # Here since there is no way to know the number of trials, we just save the study
        return None, None, study.best_trial.params

    # Save study
    joblib.dump(study, study_filename)

    # If there was a final model trained, remove it
    if os.path.isfile(os.path.join(save_folder, 'model.pt')):
        os.remove(os.path.join(save_folder, 'model.pt'))
        logging.info('Removed previous final model')

    # Save the sampler with pickle to be loaded later.
    with open(sampler_filename, "wb") as fout:
        pickle.dump(study.sampler, fout)

    # Get the trials history
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))    
    df.to_csv(os.path.join(save_folder, f'{study_name}_trials.csv'))
    logging.info(f'Saved trials history of {study_name} to {save_folder}\n')    

    # Define best model
    try:
        best_params = study.best_trial.params
        logging.info(f'Best params: {best_params}\n')

        print("Best trial:")
        print(" Value: ", study.best_trial.value)
        print(" Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
    except ValueError:
        print('No best trial found, go with default params')
        best_params = objective.default_params

    # Create the model and load best model
    used_params = objective.default_params.copy()
    used_params.update(best_params)  # Again, we don't want to update the default params
    model = objective.build_model(used_params)
    
    if kwargs.get('just_load_model', False):
        return model, None, used_params
    
    # Re-train final version of the model with the best parameters
    use_all_data = True
    if use_all_data:
        train_idx = np.concatenate((objective.train_idx, objective.valid_idx))
        valid_idx = None
    else:
        # Use a bit of validaton data during training to avoid overfitting -- this introduces some variability in the results.
        labels = objective.dataset.label.squeeze().data.numpy()
        all_train = np.concatenate((objective.train_idx, objective.valid_idx))
        labels = labels[all_train]
        train_idx, valid_idx, _, _ = train_test_split(all_train, labels, stratify=labels, test_size=0.15)

    objective.set_indices(train_idx, valid_idx, test_idx=objective.test_idx, normal_group_idx=None, save_norm=False) # This updates the normalisatiom
    # os.system(f"rm {save_folder}/model.pt")
    # os.system(f"rm {save_folder}/checkpoint.pt")
    res_training = objective._train(model, used_params, save_folder, final_model=True, output_probs=kwargs.get('output_probs', False))
    
    # np.asarray(res_training['metrics_test']['accuracy']).max()
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(res_training['losses_test'], label='Test')
    # ax[0].plot(res_training['losses_train'], label='Train')
    # ax[1].plot(res_training['metrics_test']['accuracy'], label='Test')
    # ax[1].plot(res_training['metrics_train']['accuracy'], label='Train')

    return model, res_training, best_params


def optuna_cv(objective_optuna, 
              cv_folder, 
              k_folds=5, 
              cv_indices=None,
              test_idx=None,
              num_trials=10, 
              append="", 
              load_previous=False,
              **kwargs,
              ):
    """Implement model cross-validation with Optuna for hyper-parameter tuning"""
    # Here we do K-fold stratified cross-validation; find best hyper-parameters on each fold; re-train in train+valid
    # and evaluate on the test set    

    y = objective_optuna.dataset.label.squeeze().data.numpy() 
    all_indices = np.arange(0, len(y))    
    
    acc_graph = np.zeros((k_folds,))
    list_params = []
    list_probs = []
    list_decisions = []
    
    if cv_indices is None:
        cv_indices, test_indices = cv_stratified_split(all_indices, y, k_folds=k_folds, test_size=0.2)
        test_idx = test_indices['X_test']

    # for ix_split, (train_index, valid_index) in enumerate(skfold.split(train_indices, train_labels)):
    for ix_split in range(0, k_folds):
        # Save folder
        split_name = f"Inner_{ix_split}{append}"
        save_folder_name = os.path.join(cv_folder, split_name)
        os.makedirs(save_folder_name, exist_ok=True)
        objective_optuna.save_dir = save_folder_name

        # Get indices -- the indices are with respect to the total dataset
        train_idx = cv_indices[ix_split]['X_train']       
        valid_idx = cv_indices[ix_split]['X_valid']                              
        objective_optuna.set_indices(train_idx, valid_idx, test_idx=test_idx)

        # Need to define train / test / valid indices                
        model, res_training, best_params = hypertune_optuna(objective_optuna, 
                                                            save_folder_name, 
                                                            objective_optuna.study_name,  
                                                            num_trials=num_trials, 
                                                            load_previous=load_previous,
                                                            output_probs=True)

        best_epoch = res_training['best_epoch']
        # Accuracy on the test split
        acc_test = res_training['metrics_test']['accuracy'][best_epoch]  
        acc_graph[ix_split] = acc_test

        # Save params
        list_params.append(best_params)

        # Get decisions and probs
        probs = res_training['probs']
        decisions = res_training['decisions'] 
        
        # list_decisions.append([y_pred, test_y])
        list_decisions.append([d.cpu() for d in decisions])
        list_probs.append(probs.cpu())
        
        # Print the test acuracy
        print(f"Test accuracy: {acc_test:.4f}")

        # Save the accuracy graph
        np.savetxt(os.path.join(save_folder_name, "accuracy_test.txt"), [acc_test], delimiter=",")

    return acc_graph, list_params, list_probs, list_decisions


def optuna_nested_cv(objective_optuna,
                     save_folder,
                     out_folds=10, 
                     in_folds=5,
                     test_size=0.2,
                     num_trials=10,
                     load_previous=False,
                     splits_file=None,
                     **kwargs,
                     ):
    """Implement model nested cross-validation"""    
    acc_filename = os.path.join(save_folder, "accuracy_cv.csv")
    prob_filename = os.path.join(save_folder, 'cv_data.pt')
    params_filename = os.path.join(save_folder, 'params.csv')
    redo = kwargs.get('redo', False)

    if os.path.isfile(splits_file):        
        with open(splits_file, 'rb') as file:
            splits = pickle.load(file)
        out_folds = len(splits['cv_indices'])
        in_folds = len(splits['cv_indices'][0])
        predefined_splits = True
    else:
        predefined_splits = False
    
    # Run the nested CV    
    if not os.path.isfile(acc_filename) or redo:
        indices = np.arange(objective_optuna.dataset.__len__())
        labels = objective_optuna.dataset.label.squeeze().data.numpy()

        acc = np.zeros((out_folds, in_folds))
        list_params_folds = []
        list_probs_folds = []
        list_decisions_folds = []
                
        # for ix, (train_index, test_index) in enumerate(skfold.split(indices, labels)):
        for ix_out in range(0, out_folds):
            if predefined_splits:
                cv_indices = splits['cv_indices'][ix_out]
                test_idx = splits['test_indices'][ix_out]['X_test']
            else:
                cv_indices, test_indices = cv_stratified_split(indices, labels, k_folds=in_folds, test_size=test_size)
                test_idx = test_indices['X_test']
        
            acc_fold, list_params, list_probs, list_decisions = optuna_cv(objective_optuna, 
                                                                          save_folder, 
                                                                          in_folds, 
                                                                          cv_indices,
                                                                          test_idx,
                                                                          num_trials=num_trials, 
                                                                          append=f"-Outer_{ix_out}",
                                                                          load_previous=load_previous,
                                                                          **kwargs,
                                                                          )


            acc[ix_out, :] = acc_fold
            list_params_folds.append(list_params)
            list_probs_folds.append(list_probs)
            list_decisions_folds.append(list_decisions)

        # ================= Save the results =================
        if out_folds == 1:
            test_mean = acc.mean(axis=1)
            test_std = acc.std(axis=1)
        else:
            test_mean = acc.mean()
            test_std = acc.std()

        print("Accuracy of nested CV:")
        print(acc)
        print(f"Mean: {test_mean:.2f}, Std: {test_std:.2f}")
        print("\n\n\n")
        
        df_results = pd.DataFrame(data=np.array([test_mean, test_std])[:, np.newaxis].T, 
                                columns=['Acc_Test', 'Acc_Std'], index=['Graph'])
        df_results['Edges'] = True
        df_results['Normalization'] = f"{objective_optuna.normalization}"
        # df_results['Global'] = global_data
        df_results.to_csv(acc_filename)

        # Save the decisions and the probabilities
        probs_folds = []
        pred_folds = []
        true_folds = []
        for f_ix in range(out_folds):
            probs_folds.append(torch.stack(list_probs_folds[f_ix]))
            tmp_pred = []
            tmp_true = []
            for s_ix in range(in_folds):
                tmp_pred.append(list_decisions_folds[f_ix][s_ix][0])
                tmp_true.append(list_decisions_folds[f_ix][s_ix][1])
            pred_folds.append(torch.stack(tmp_pred))
            true_folds.append(torch.stack(tmp_true))

        save_cv_data = {'probs': probs_folds, 
                        'pred_folds': pred_folds, 
                        'true_folds': true_folds,
                        'params_names': list(list_params_folds[0][0].keys())}
        torch.save(save_cv_data, prob_filename)

        # Parameters
        ix = 0
        df_params = pd.DataFrame(data=[])
        for ix_of, params_of in enumerate(list_params_folds):
            for ix_sp, params_sp in enumerate(params_of):
                tmp_params = pd.DataFrame.from_dict(data=params_sp, orient='index', columns=[ix]).T
                tmp_params['Accuracy'] = acc[ix_of, ix_sp]
                tmp_params['OuterFold'] = ix_of
                tmp_params['InnerFold'] = ix_sp
                df_params = pd.concat([df_params, tmp_params], axis=0)
                ix += 1
        df_params.to_csv(params_filename)
    else:
        df_params = pd.read_csv(params_filename, index_col=0)
        df_results = pd.read_csv(acc_filename, index_col=0)
        save_cv_data = torch.load(prob_filename)

    # Train on the final model

    # Select best model and train it on a train+valid split or all the data... [this would be the model that goes into production/in the challenge]
    
    return df_results, df_params, save_cv_data

