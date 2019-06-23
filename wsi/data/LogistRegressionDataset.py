from torch.utils.data import Dataset
import pandas as pd
import torch


class LogistRegressionDataset(Dataset):

    def __init__(self, csv_file):
        self.histogram_index = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.histogram_index)

    def __getitem__(self, item):
        number_A = self.histogram_index.iloc[item, 0]
        number_B = self.histogram_index.iloc[item, 1]
        image_label = self.histogram_index.iloc[item, 2]
        A = torch.full((1,), int(number_A))
        B = torch.full((1,), int(number_B))
        histogram = torch.cat((A,B), dim=0)
        if image_label == 'A':
            label = torch.ones((1,), dtype=torch.uint8)
        else:
            label = torch.zeros((1,), dtype=torch.uint8)
        return histogram, label

if __name__ == '__main__':
    data = LogistRegressionDataset('/home/student2/class_1/coords/train_second.csv')
    histogram, label = data[1]
    print(histogram)
    print(label)




