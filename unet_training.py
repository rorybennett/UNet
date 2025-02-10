"""
Script for training a basic UNet model. Various models are available in the UNetModels package. This script
is mostly concerned with the training setup used by pytorch.
"""
import json
import os
from datetime import datetime
from os.path import join

import matplotlib.pyplot as plt
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from Datasets.ProstateBladderDataset import ProstateBladderDataset as PBD
from EarlyStopping.EarlyStopping import EarlyStopping
from UNetModels.UNet import UNet
from Utils import Utils, ArgsParser

########################################################################################################################
# Track run time of script.
########################################################################################################################
script_start = datetime.now()

########################################################################################################################
# Set seeds for semi-reproducibility.
########################################################################################################################
torch.manual_seed(2025)

########################################################################################################################
# Initialise parameters.
########################################################################################################################
args = ArgsParser.get_arg_parser()
dataset_dir = args.dataset_path  # Path to dataset directory.
images_dir = args.train_images_path  # Path to training images directory (dataset_dir is root).
labels_dir = args.train_labels_path  # Path to training labels directory (dataset_dir is root).
save_path = args.save_path  # Path to saving directory, where models, loss plots, and validation results are stored.

batch_size = args.batch_size  # Batch size.
oversampling_factor = args.oversampling_factor  # Factor to oversample.
num_epochs = args.epochs  # Training epochs.
image_size = args.image_size  # Image size for resizing, used in training and validation.
patience = args.patience  # Early stopping patience.
fold = args.fold  # Fold number to train ('all' for using all data).
# Get fold structure from file.
with open(join(dataset_dir, 'fold_structure.json'), 'r') as file:
    fold_structure = json.load(file)

# Use fold_structure.
key, vals = fold, fold_structure[f'fold_{fold}']


def main():
    print('====================================================================================================='
          f'{key} Training starting...')

    start_time = datetime.now()
    save_dir = f'{save_path}/{key}'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    ################################################################################################################
    # Transformers (a temporary dataset is used to calculate the training mean and std for normalising).
    ################################################################################################################
    mean, std = PBD(images_dir, labels_dir, vals['train'], image_size=(0, 0), train_mean=0,
                    train_std=0).get_mean_and_std()
    # These transforms are only applied to the oversampled images.
    transforms_train = v2.Compose([
        v2.RandomAffine(degrees=30, scale=(0.8, 1.1), translate=(0.1, 0.1), shear=15),
        v2.ElasticTransform(),
        v2.RandomCrop(int(5 * image_size / 6)),
        v2.Resize((image_size, image_size)),
        v2.GaussianNoise(mean=0, sigma=0.3, clip=False),
        v2.GaussianBlur(3)
    ])
    # Save training parameter values.
    with open(join(dataset_dir), 'training_parameters.txt') as file:
        file.write('Training Parameters\n'
                   '============================================================\n'
                   f'Dataset Path: {dataset_dir}\n'
                   f'Images Path: {images_dir}\n'
                   f'Labels Path: {labels_dir}\n'
                   f'Save Path: {save_dir}')

    ################################################################################################################
    # Datasets and dataloaders.
    ################################################################################################################
    train_dataset = PBD(images_dir, labels_dir, vals['train'], oversampling_factor=oversampling_factor,
                        transforms=transforms_train, image_size=(image_size, image_size), train_mean=mean,
                        train_std=std)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # for i in range(len(train_dataset)):
    #     train_dataset.__getitem__(i)
    #
    # exit()

    val_dataset = PBD(images_dir, labels_dir, vals['val'], image_size=(image_size, image_size), train_mean=mean,
                      train_std=std)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    ################################################################################################################
    # Model, optimiser, learning rate schedular.
    ################################################################################################################
    model = UNet().cuda()
    # Define optimiser and loss function.
    optimiser = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.005)
    lr_schedular = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=num_epochs, eta_min=0)

    early_stopping = EarlyStopping(patience=patience, delta=0.001)
    ################################################################################################################
    # Train through epochs.
    ################################################################################################################
    criterion = nn.BCEWithLogitsLoss()
    train_losses = []
    val_losses = []
    lr = []
    for epoch in range(num_epochs):
        ############################################################################################################
        # Train step.
        ############################################################################################################
        model.train()
        epoch_train_loss = 0
        for images, labels in train_dataloader:
            images = images.to('cuda')
            labels = labels.to('cuda').unsqueeze(1).float()  # Loss calculation requires this.

            optimiser.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimiser.step()

            epoch_train_loss += loss.item()
        epoch_train_loss = epoch_train_loss / len(train_dataloader)
        lr_schedular.step()
        ############################################################################################################
        # Validation step.
        ############################################################################################################
        epoch_val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to('cuda')
                labels = labels.to('cuda').unsqueeze(1).float()

                outputs = model(images)

                loss = criterion(outputs, labels)

                epoch_val_loss += loss.item()
            epoch_val_loss = epoch_val_loss / len(val_dataloader)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        lr.append(lr_schedular.get_last_lr()[0])

        time_now = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        print(f"{time_now} -- "
              f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_losses[-1]:.8f}, "
              f"Val Loss: {val_losses[-1]:.8f}, ", end='')

        early_stopping(epoch_val_loss, model, epoch, optimiser, save_path)

        Utils.plot_losses(early_stopping.best_epoch + 1, train_losses, val_losses, lr, save_path)

        if early_stopping.early_stop:
            print('Patience reached, stopping early.')
            break
        else:
            print()

    ################################################################################################################
    # Once training complete, plot validation images.
    ################################################################################################################
    val_model = UNet().cuda()
    val_model.load_state_dict(torch.load(join(save_path, 'model_latest.pth'))['model_state_dict'])
    val_model.eval()
    with torch.no_grad():
        counter = 0
        for images, labels in val_dataloader:
            images = images.to('cuda')
            labels = labels.to('cuda').unsqueeze(1).float()

            outputs = val_model(images)

            outputs = outputs.cpu().numpy()
            images = images.cpu().numpy()
            labels = labels.cpu().numpy()

            for i in range(len(images)):
                fig, ax = plt.subplots(1, 3, figsize=(16, 9))
                for a in ax:
                    a.axis('off')

                ax[0].set_title('Input image')
                ax[0].imshow(images[i].transpose(1, 2, 0), cmap='gray')
                ax[1].set_title('Output mask')
                o = (outputs[i].transpose(1, 2, 0) > 0.5).astype('uint8')
                ax[1].imshow(o, cmap='gray')
                ax[2].set_title('Ground truth')
                ax[2].imshow(labels[i].transpose(1, 2, 0), cmap='gray')
                plt.savefig(join(save_path, f'val_result_{counter}.png'))
                plt.close()
                counter += 1
    end_time = datetime.now()
    run_time = start_time - end_time
    print(f'{key} Training completed.\n'
          f'Best Epoch: {early_stopping.best_epoch + 1}.\n'
          f"End time: {end_time.strftime('%Y-%m-%d  %H:%M:%S')}.\n"
          f'Total run time: {run_time}.')
    print('=====================================================================================================')


if __name__ == '__main__':
    main()
