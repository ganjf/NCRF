# %load /home/student2/classification/dataset/TrainDataset.py
import os
import csv
import torch
import pandas as pd
import openslide
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class GridPatchTrainDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.image_index_label = pd.read_csv(csv_file)
        self.rootdir = root_dir
        self.transform = transforms.Compose([
            transforms.RandomCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.ToTensor()])   
        
    def __len__(self):
            return len(self.image_index_label)
        
    def __getitem__(self, idx):
        image_path = os.path.join(self.rootdir, self.image_index_label.iloc[idx,0])
        label_class = self.image_index_label.iloc[idx, 1]
        if label_class == 'A':
            label = torch.ones(size=(4,), dtype=torch.uint8)
        elif label_class == 'B':
            label = torch.zeros(size=(4,), dtype=torch.uint8)
        slide = openslide.open_slide(image_path)
        level_count = slide.level_count
        [m, n] = slide.dimensions
        region = np.array(slide.read_region((0,0),(level_count-1),(m,n)))
        region = transforms.ToPILImage()(region).convert('RGB')
        region = self.transform(region)
        patch = torch.stack([region[:,0:224,0:224], region[:,0:224,224:448],
                    region[:,224:448,0:224], region[:,224:448,224:448]], 0)
        return patch, label

if __name__ == '__main__': 
    dataset = GridPatchTrainDataset(csv_file="/home/student2/classification/dataset/train.csv", root_dir='/mnt/data/students/traindata_patch/10x')
    print(len(dataset))
    sample, label = dataset[68]
    print(sample.size())
    print(label)
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(sample[0].permute(1,2,0).numpy())
    plt.subplot(2,2,2)
    plt.imshow(sample[1].permute(1,2,0).numpy())
    plt.subplot(2,2,3)
    plt.imshow(sample[2].permute(1,2,0).numpy())
    plt.subplot(2,2,4)
    plt.imshow(sample[3].permute(1,2,0).numpy())
    plt.show()
