import pandas as pd
import openslide
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import csv
import logging
from torchvision import transforms
def list_all_files(rootdir):
    import os 
    _files = []
    list = os.listdir(rootdir)
    for list_index in list:
        path = os.path.join(rootdir, list_index)
        if os.path.isfile(path):
            _files.append(list_index)
        else:
            continue
    return _files
if __name__ == '__main__':
  base_path = '/mnt/data/students/traindata_patch/10x'
  image_index_label = pd.read_csv("/mnt/data/students/trainlabel/trainlabel.csv")
  trainTransform = transforms.Compose([
                  transforms.ToTensor()])
  count = 0
  logger = logging.getLogger("file")
  logger.setLevel(logging.DEBUG)
  fileHanlder = logging.FileHandler('file.log')
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  fileHanlder.setFormatter(formatter)
  logger.addHandler(fileHanlder)
  
  for i in range(len(image_index_label)):
      label = image_index_label.iloc[i,1]
      path = os.path.join(base_path, image_index_label.iloc[i,0])
      files = list_all_files(path)
      with open('/mnt/data/students/student2/train_mx.csv', 'a+') as csvfile:
          csv_writer = csv.writer(csvfile)
          for file in files:
              file_path = os.path.join(image_index_label.iloc[i,0], file)
              path = os.path.join(base_path, file_path)
              slide = openslide.open_slide(path)
              level_count = slide.level_count
              [m, n] = slide.dimensions
              region = np.array(slide.read_region((0,0),(level_count-1),(m,n)))
              region = transforms.ToPILImage()(region).convert('RGB')
              region = trainTransform(region)
              if(int(region.sum()) < int(0.85*3*512*512)):
                  count += 1
                  logger.info(str(count) + '\t' + str(int(region.sum())))
                  data_row = [file_path, label]
                  csv_writer.writerow(data_row)
              else:
                  continue
  logger.info('11111111111111111111111111111111111111111')