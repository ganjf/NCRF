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

    def __len__(self):
        return len(self.image_index_label)

    def __getitem__(self, idx):
        index_path = os.path.join(self.rootdir, self.image_index_label.iloc[idx, 0])
        self._files_Name = self._list_all_files(index_path)
        label_class = self.image_index_label.iloc[idx, 1]
        image_list = []
        image_score_list = []
        for file_index in self._files_Name:
            image_path = os.path.join(index_path, file_index)
            slide = openslide.open_slide(image_path)
            level_count = slide.level_count
            [m, n] = slide.dimensions
            region = np.array(slide.read_region((0, 0), (level_count - 1), (m, n)))
            region = transforms.ToPILImage()(region).convert('RGB')
            region = self.testTransforms(region)
            if (int(torch.sum(region)) < int(0.85 * 3 * 448 * 448)):
                region_left_top = torch.full((1,), float(torch.sum(region[:, 0:224, 0:224])))
                region_left_bottom = torch.full((1,), float(torch.sum(region[:, 0:224, 224:448])))
                region_right_top = torch.full((1,), float(torch.sum(region[:, 224:448, 0:224])))
                region_right_bottom = torch.full((1,), float(torch.sum(region[:, 224:448, 224:448])))
                region_sum = torch.cat((region_left_top,region_left_bottom,region_right_top,region_right_bottom),dim=0)
                _, index = torch.sort(region_sum,descending=True)
                image_score = torch.zeros((4,), dtype=torch.float)
                image_score[index[0]] = 0.1
                image_score[index[1]] = 0.1
                image_score[index[2]] = 0.4
                image_score[index[3]] = 0.4
                patch = torch.stack([region[:, 0:224, 0:224], region[:, 0:224, 224:448], region[:, 224:448, 0:224],
                                     region[:, 224:448, 224:448]], 0)
                image_list.append(patch)
                image_score_list.append(image_score)
        if label_class == 'A':
            label = torch.ones((1,), dtype=torch.uint8)
        elif label_class == 'B':
            label = torch.zeros((1,), dtype=torch.uint8)
        return image_list, image_score_list, label

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


if __name__ == '__main__':
    dataset = ImageTestDataset(csv_file='/mnt/data/students/trainlabel/trainlabel.csv',
                               root_dir='/mnt/data/students/traindata_patch/10x')
    print(len(dataset))
    element, score, label = dataset[0]
    print(score)
    print(torch.sum(element[0][0, :, :]))
    print(torch.sum(element[0][1, :, :]))
    print(torch.sum(element[0][2, :, :]))
    print(torch.sum(element[0][3, :, :]))
    print(len(element))
    print(label.size())
    image = element[0][0, :, :].permute(1, 2, 0)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(element[0][0, :, :].permute(1, 2, 0).numpy())
    plt.subplot(2, 2, 2)
    plt.imshow(element[0][1, :, :].permute(1, 2, 0).numpy())
    plt.subplot(2, 2, 3)
    plt.imshow(element[0][2, :, :].permute(1, 2, 0).numpy())
    plt.subplot(2, 2, 4)
    plt.imshow(element[0][3, :, :].permute(1, 2, 0).numpy())
    plt.show()
    batch_size = 32
    batch_num = int(len(element) / batch_size)
    for index in range(batch_num):
        image_list_set = element[index * batch_size : (index + 1) * batch_size]
        image_set = torch.stack(image_list_set, 0)
        print(image_set.size())
    image_list_set = element[batch_num * batch_size: ]
    print(len(image_list_set))
    image_set = torch.stack(image_list_set, 0)
    print(image_set.size())