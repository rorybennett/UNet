"""
Patience class to check if training should be cancelled due to no improvement in the validation set.

The choice of delta is quite important. If too high then the model does not train enough, if too low the model
over fits.

Both the best and the latest models are saved by default.

Beware the difference between epoch values. The main loop typically uses zero index, but when displaying values
it starts at 1.
"""

from os.path import join

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=100, delta=0.001, save_latest=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.best_epoch = 0
        self.save_latest = save_latest
        # If current score within this range, it is considered the same as previous score.
        self.delta = delta

    def __call__(self, val_loss, model, epoch, optimiser, save_path):
        score = -val_loss

        if self.best_score is None:
            print(f'Initial Save... ', end='')
            self.best_epoch = epoch
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimiser, save_path, 'model_best.pth')
        elif score < self.best_score + self.delta:
            print(f'No improvement (delta: {self.delta}, best: {abs(self.best_score):0.3f})... ', end='')
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            if self.save_latest:
                self.save_checkpoint(val_loss, model, epoch, optimiser, save_path, 'model_latest.pth')
        else:
            print('New best! Saving model... ', end='')
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model, epoch, optimiser, save_path, 'model_best.pth')
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimiser, save_path, model_name: str):
        """Saves model."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
        }, join(save_path, model_name))
        self.val_loss_min = val_loss
