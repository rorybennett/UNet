import json
from os.path import join

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from Datasets.ProstateBladderDataset import ProstateBladderDataset
from UNetModels.UNet import UNet


def main():
    save_path = f'TrainingResults/temp'
    model_path = f'TrainingResults/Dataset011_sAUSprostate/fold_all'
    model = UNet().cuda()
    model.load_state_dict(torch.load(join(model_path, 'model_best.pth'))['model_state_dict'])

    size = 600
    final_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((size, size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    dataset_dir = 'UNet Datasets/Initial 19'
    dataset = 'Dataset011_sAUSprostate'
    images_dir = f'{dataset_dir}/{dataset}/imagesTr'
    labels_dir = f'{dataset_dir}/{dataset}/labelsTr'

    with open(join(dataset_dir, dataset, 'fold_structure.json'), 'r') as file:
        folds = json.load(file)

    for key, vals in reversed(folds.items()):
        final_dataset = ProstateBladderDataset(images_dir, labels_dir, vals['val'], transform=final_transform)
        final_dataloader = DataLoader(final_dataset, batch_size=1, shuffle=False)

        model.eval()
        with torch.no_grad():
            counter = 0
            for images, labels in final_dataloader:
                images = images.to('cuda')
                labels = labels.to('cuda')

                outputs = model(images)

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
                    o = (outputs[i] > 0.5).transpose(1, 2, 0).astype('uint8') * 255
                    ax[1].imshow(o, cmap='gray')
                    ax[2].set_title('Ground truth')
                    ax[2].imshow(labels[i].transpose(1, 2, 0), cmap='gray')
                    plt.savefig(join(save_path, f'val_result_{counter}.png'))
                    plt.close()
                    counter += 1
        exit()


if __name__ == '__main__':
    main()
