import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
import openslide
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class ScoringTestDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.image_index_label = pd.read_csv(csv_file)
        self.rootdir = root_dir
        self._files_Name = None
        self.testTransforms = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor()])
        self.testTransformsAuxiliary = transforms.Compose([
            transforms.Resize(400),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.image_index_label)

    def __getitem__(self, idx):
        index_path = os.path.join(self.rootdir, self.image_index_label.iloc[idx, 0])
        self._files_Name = self._list_all_files(index_path)
        label_class = self.image_index_label.iloc[idx, 1]
        image_list = []
        image_score_list = []
        image_auxiliary_list = []
        for file_index in self._files_Name:
            image_path = os.path.join(index_path, file_index)
            slide = openslide.open_slide(image_path)
            level_count = slide.level_count
            [m, n] = slide.dimensions
            region = np.array(slide.read_region((0, 0), (level_count - 1), (m, n)))
            region = transforms.ToPILImage()(region).convert('RGB')
            region_patch = self.testTransforms(region)
            if (int(torch.sum(region_patch)) < int(0.85 * 3 * 448 * 448)):
                region_left_top = torch.full((1,), float(torch.sum(region_patch[:, 0:224, 0:224])))
                region_left_bottom = torch.full((1,), float(torch.sum(region_patch[:, 0:224, 224:448])))
                region_right_top = torch.full((1,), float(torch.sum(region_patch[:, 224:448, 0:224])))
                region_right_bottom = torch.full((1,), float(torch.sum(region_patch[:, 224:448, 224:448])))
                region_sum = torch.cat((region_left_top,region_left_bottom,region_right_top,region_right_bottom),dim=0)
                _, index = torch.sort(region_sum,descending=True)
                image_score = torch.zeros((4,), dtype=torch.float)
                image_score[index[0]] = 0.1
                image_score[index[1]] = 0.1
                image_score[index[2]] = 0.4
                image_score[index[3]] = 0.4
                patch = torch.stack([region_patch[:, 0:224, 0:224], region_patch[:, 0:224, 224:448], region_patch[:, 224:448, 0:224],
                                     region_patch[:, 224:448, 224:448]], 0)
                discriminate_patch = self.testTransformsAuxiliary(region)
                image_list.append(patch)
                image_score_list.append(image_score)
                image_auxiliary_list.append(discriminate_patch)
        if label_class == 'A':
            label = torch.ones((1,), dtype=torch.uint8)
        elif label_class == 'B':
            label = torch.zeros((1,), dtype=torch.uint8)
        return image_list, image_auxiliary_list, image_score_list, label

    def _list_all_files(self, root):
        import os
        _files = []
        list = os.listdir(root)
        for list_index in list:
            path = os.path.join(root, list_index)
            if os.path.isfile(path):
                _files.append(list_index)
            else:
                continue
        return _files