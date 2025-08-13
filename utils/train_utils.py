import torch 
import logging

def load_checkpoint(filename, model, optimizer, scheduler, device):
    try:
        # Load previous model
        checkpoint = torch.load(filename, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler']),

        init_epoch = checkpoint['epoch']
        best_score = checkpoint['best_score']
        best_epoch = checkpoint['best_epoch']

        # Losses
        losses_train = checkpoint.get('losses_train', None)
        losses_valid = checkpoint.get('losses_valid', None)
        losses_test = checkpoint.get('losses_test', None)
        losses = (losses_train, losses_valid, losses_test)

        # Metrics        
        track_metrics_train = checkpoint.get('metrics_train', None)
        track_metrics_valid = checkpoint.get('metrics_valid', None)
        track_metrics_test = checkpoint.get('metrics_test', None)
        metrics = (track_metrics_train, track_metrics_valid, track_metrics_test)

        return init_epoch, best_epoch, best_score, losses, metrics
    except Exception as e:
        print(f'Error loading checkpoint: {e}')
        logging.error(f'\nError loading checkpoint:\n {e}')
        raise RuntimeError(f'Error loading checkpoint: {e}')


def save_checkpoint(filename, model, optimizer, scheduler, add_dict):
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    save_dict.update(add_dict)
    torch.save(save_dict, filename)