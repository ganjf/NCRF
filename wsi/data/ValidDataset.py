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


class GridPatchValidnDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.image_index_label = pd.read_csv(csv_file)
        self.rootdir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.image_index_label)

    def __getitem__(self, idx):
        image_path = os.path.join(self.rootdir, self.image_index_label.iloc[idx, 0])
        label_class = self.image_index_label.iloc[idx, 1]
        if label_class == 'A':
            label = torch.ones(size=(4,), dtype=torch.uint8)
        elif label_class == 'B':
            label = torch.zeros(size=(4,), dtype=torch.uint8)
        slide = openslide.open_slide(image_path)
        level_count = slide.level_count
        [m, n] = slide.dimensions
        region = np.array(slide.read_region((0, 0), (level_count - 1), (m, n)))
        region = transforms.ToPILImage()(region).convert('RGB')
        region = self.transform(region)
        patch = torch.stack([region[:, 0:224, 0:224], region[:, 0:224, 224:448],
                             region[:, 224:448, 0:224], region[:, 224:448, 224:448]], 0)
        return patch, label