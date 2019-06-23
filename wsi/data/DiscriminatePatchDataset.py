import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
import openslide
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class DiscriminatePatchDataset(Dataset):

    def __init__(self, csv_file, root_dir):
        self.image_index_label = pd.read_csv(csv_file)
        self.rootdir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor()
        ])
        self.transform_discriminate = transforms.Compose([
            transforms.RandomCrop(400),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(30, 120)),
            transforms.ColorJitter(brightness=0.5),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.image_index_label)

    def __getitem__(self, idx):
        image_path = os.path.join(self.rootdir, self.image_index_label.iloc[idx, 0])
        label_class = self.image_index_label.iloc[idx, 1]
        if label_class == 'A':
            label = torch.ones(size=(1,), dtype=torch.uint8)
        elif label_class == 'B':
            label = torch.zeros(size=(1,), dtype=torch.uint8)
        slide = openslide.open_slide(image_path)
        level_count = slide.level_count
        [m, n] = slide.dimensions
        region = np.array(slide.read_region((0, 0), (level_count - 1), (m, n)))
        region = transforms.ToPILImage()(region).convert('RGB')
        region_patch = self.transform(region)
        grid_patch = torch.stack([region_patch[:, 0:224, 0:224], region_patch[:, 0:224, 224:448],
                                  region_patch[:, 224:448, 0:224], region_patch[:, 224:448, 224:448]], 0)

        # br_patch = self.transform_br(region)
        # channels, height, width = br_patch.size()
        # temp_1_1 = torch.mul(torch.full((height, width), 100).float(), br_patch[2, :, :])
        # temp_1_2 = torch.ones((height, width)) + br_patch[0, :, :] + br_patch[1, :, :]
        # temp_1 = torch.div(temp_1_1, temp_1_2)
        # temp_2_1 = torch.full((height, width), 256).float()
        # temp_2_2 = torch.ones((height, width)) + br_patch[0, :, :] + br_patch[1, :, :] + br_patch[2, :, :]
        # temp_2 = torch.div(temp_2_1, temp_2_2)
        # br = torch.mul(temp_1, temp_2)
        # br = br.unsqueeze(dim=0)

        discriminate_patch = self.transform_discriminate(region)

        region_left_top = torch.full((1,), float(torch.sum(region_patch[:, 0:224, 0:224])))
        region_left_bottom = torch.full((1,), float(torch.sum(region_patch[:, 0:224, 224:448])))
        region_right_top = torch.full((1,), float(torch.sum(region_patch[:, 224:448, 0:224])))
        region_right_bottom = torch.full((1,), float(torch.sum(region_patch[:, 224:448, 224:448])))
        region_sum = torch.cat((region_left_top, region_left_bottom, region_right_top, region_right_bottom), dim=0)
        _, index = torch.sort(region_sum, descending=True)
        grid_score = torch.zeros((4,), dtype=torch.float)
        grid_score[index[0]] = 0.1
        grid_score[index[1]] = 0.1
        grid_score[index[2]] = 0.4
        grid_score[index[3]] = 0.4

        return grid_patch, grid_score, discriminate_patch, label


if __name__ == '__main__':
    dataset = DiscriminatePatchDataset(csv_file="/home/student2/class/coords/train.csv",
                                       root_dir='/mnt/data/students/traindata_patch/10x')
    print(len(dataset))
    sample, grid_score, dicriminate_patch, label = dataset[68]
    print(sample.size())
    print(grid_score.size())
    print(dicriminate_patch.size())
    print(label.size())
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(sample[0].permute(1, 2, 0).numpy())
    plt.subplot(2, 3, 2)
    plt.imshow(sample[1].permute(1, 2, 0).numpy())
    plt.subplot(2, 3, 3)
    plt.imshow(sample[2].permute(1, 2, 0).numpy())
    plt.subplot(2, 3, 4)
    plt.imshow(sample[3].permute(1, 2, 0).numpy())
    plt.subplot(2, 3, 5)
    plt.imshow(dicriminate_patch[0, :, :].numpy(), cmap='gray')
    plt.show()