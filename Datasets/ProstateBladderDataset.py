"""
Dataset class for prostate and bladder segmentation.

Images are stored as tv_tensors.Image and labels are stored as tv_tensors.Mask.

The image and label data is assumed to be stored as .png with pixel values between (0, 255). The images
are read in as greyscale images, then divided by 255 to scale them between (0, 1). There were issues with the
v2.ToDType(torch.float32, scale=True) function (if the datatype is already float32 when this transform is called,
no transform is applied and the values are not scaled).
"""

import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from natsort import natsorted
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.tv_tensors import Mask, Image


class ProstateBladderDataset(Dataset):
    def __init__(self, images_dir, labels_dir, fold_numbers, train_mean, train_std, image_size, transforms=None,
                 oversampling_factor=1, verbose=False):
        """
        Initialise the ProstateBladderDataset class.
            - Images are read in as greyscale images (single channel) (0, 255) then immediately divided by 255. If
            the saved images are not in the range of (0, 255) then this will need to be manually changed.
            - fold_numbers: Since the directories are not split into folds, the fold_numbers contains the (indices + 1)
            of the images and labels to be used by the dataset (the rest are ignored). This allows for splitting
            of the dataset into training and validation without creating separate directories.
            - train_mean, train_std, and image_size: Normalising and resizing are always applied. Training normalising
            parameters are acquired using the get_mean_and_std() function.
            - transforms: Resize and normalise are always applied, so they do not need to be included here (unless
            a crop function is used, then it must be Resized to the required dimensions). These transforms are only
            applied if oversampling_factor > 1, otherwise transformations will not be applied. This came about because
            if every image is transformed, the original features are not being learned.

        :param images_dir: Directory containing images in .png format, with pixels in the range of (0, 255).
        :param labels_dir: Directory containing labels/masks in .png format, with pixels in the range of (0, 255)
        :param fold_numbers: Indices of the images and labels in the images_dir and labels_dir to be used.
        :param train_mean: Mean of the training set, used in normalising the training and validation data.
        :param train_std: Standard deviation of the training set.
        :param image_size: Final size of image before being given to the model.
        :param transforms: Transforms to be applied to oversampled data.
        :param oversampling_factor: How often to apply over sampling (2 means the dataset is doubled).
        :param verbose: Print extra information.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.oversampling_factor = oversampling_factor
        self.image_size = image_size
        self.train_mean = train_mean
        self.train_std = train_std
        self.verbose = verbose
        # Only use the images given in the fold_numbers list (which is offset by 1).
        self.image_files = [natsorted(os.listdir(images_dir))[i - 1] for i in fold_numbers]
        self.label_files = [natsorted(os.listdir(labels_dir))[i - 1] for i in fold_numbers]

        if self.verbose:
            print(f'Initialising the ProstateBladder dataset'
                  f'\tImages root: {self.images_dir}.\n'
                  f'\tLabels root: {self.labels_dir}.\n'
                  f'\tTotal Images: {len(self.image_files)}.\n'
                  f'\tTotal Labels: {len(self.label_files)}.\n'
                  f'\tImage Size: {self.image_size}.\n'
                  f'\tTraining Dataset Mean: {self.train_mean}.\n'
                  f'\tTraining Dataset Standard Deviation: {self.train_std}.\n'
                  f'\tOversampling count: {self.oversampling_factor}.')

        self.validate_dataset()

    def __len__(self):
        return len(self.image_files) * self.oversampling_factor

    def __getitem__(self, idx):
        """
        Overwrite the __getitem__() class (this is called whenever an image and mask from the dataset is needed by
        the dataloader class. Reads the image and mask from the disk, converts them to relative tv_tensors,
        applies necessary then optional transforms (optional only applied if over_sampling > 1), and can display
        images and labels before and after transforms if verbose == True.

        :param idx: Index of item in dataset (0, original size * oversampling_factor).

        :return: (image, label)
        """
        original_idx = idx // self.oversampling_factor
        image_path = os.path.join(self.images_dir, self.image_files[original_idx])
        label_path = os.path.join(self.labels_dir, self.label_files[original_idx])
        # Read image with range [0, 255], then convert to [0, 1].
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        fig, ax = plt.subplots(2, 2)
        if self.verbose:
            # Show pre-transform image and label.
            ax[0, 0].imshow(image, cmap='gray')
            ax[0, 1].imshow(label, cmap='gray')
        # Convert to tensors.
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.uint8)
        # Create correct tv_tensor classes (ensures correct transforms are applied)
        image = Image(image)
        label = Mask(label)

        # Apply optional transforms.
        if self.transforms:
            # Only apply optional transforms on oversampled images.
            if idx % self.oversampling_factor != 0:
                image, label = self.transforms(image, label)
        # Apply required transforms.
        required_transforms = v2.Compose([
            v2.Resize(self.image_size),
            v2.Normalize([self.train_mean], [self.train_std])
        ])
        image, label = required_transforms(image, label)

        if self.verbose:
            # Show post-transform image and label.
            ax[1, 0].imshow(image.cpu().numpy().transpose((1, 2, 0)), cmap='gray')
            ax[1, 1].imshow(image.cpu().numpy().transpose((1, 2, 0)), cmap='gray')
            ax[1, 1].imshow(label.cpu().numpy(), cmap='gray', alpha=0.8)
            plt.show()

        return image, label

    def get_mean_and_std(self):
        """
        Return the mean and std of the current dataset, scaled between (0, 1). The assumption is that the input images
        are scaled to (0, 255), so if the mean and std seem wildly off just make sure the input pixels are scaled
        correctly. These values are required for the normalising transform, and the training values should be used on
        the validation dataset.

        :return: dataset_mean, dataset_std.
        """
        means = []
        stds = []
        for image in self.image_files:
            # Read image using cv2.
            image = cv2.imread(os.path.join(self.images_dir, image), cv2.IMREAD_GRAYSCALE) / 255
            # Calculate mean and std using cv2.
            m, s = cv2.meanStdDev(image)
            means.append(m)
            stds.append(s)

        dataset_mean = np.mean(means)
        dataset_std = np.mean(stds)

        if self.verbose:
            print(f'Calculated Dataset Mean: {dataset_mean}.\n'
                  f'Calculated Dataset Standard Deviation: {dataset_std}.')

        return dataset_mean, dataset_std

    def validate_dataset(self):
        """
        Minor dataset validation:
            1. Ensure there are the same amount of images as labels.
            2. Ensure the labels and images have the same naming convention.
        """
        if self.verbose:
            print(f'Running dataset validation...')
        # Validation 1.
        assert len(self.image_files) == len(self.label_files), "Number of images != Number of labels/masks."
        # Validation 2.
        for i in range(len(self.image_files)):
            if self.image_files[i].split('.')[0] != self.label_files[i].split('.')[0]:
                assert False, f"There is a mismatch between images and labels at {self.image_files[i]} and {self.label_files[i]}."

        if self.verbose:
            print(f'Dataset validation passed.')
