import torch
import json
from wsi.data.ImageTestDataset import ImageTestDataset
from torch.nn import DataParallel
from wsi.model import MODELS
import argparse
import logging
import csv

def get_parser():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--load_path', default='/mnt/data/students/student2/ckpt/best.pkl',
                        type=str,
                        help='Path to the load models')
    parser.add_argument('--test_root', default='/home/student2/class_1/coords/train_second_resource.csv',
                        type=str,
                        help='Path to load image for test(/mnt/data/students/trainlabel/trainlabel.csv')
    parser.add_argument('--base_root', default='/mnt/data/students/traindata_patch/10x',
                        type=str,
                        help='base root for image(/mnt/data/students/traindata_patch/10x)')
    parser.add_argument('--save_path', default='/home/student2/class_1/coords/train_second.csv',
                        type=str,
                        help='Path to the saved result of patch classify')
    parser.add_argument('--batch_size', default=32,
                        type=int,
                        help='the batch size to calculate the image with model')
    parser.add_argument('--cfg_path', default='/home/student2/class_2/configs/resnet18_crf.json',
                        type=str,
                        help='Path to the config file in json format')
    args = parser.parse_args()
    return args

def run():
    args = get_parser()
    with open(args.cfg_path) as f:
        cfg = json.load(f)

    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    fileHanlder = logging.FileHandler('getGridClass.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHanlder.setFormatter(formatter)
    logger.addHandler(fileHanlder)

    dataset = ImageTestDataset(csv_file=args.test_root, root_dir=args.base_root)

    model = MODELS[cfg['model']](num_nodes=cfg['grid_size'], use_crf=cfg['use_crf'])
    model = DataParallel(model, device_ids=None)
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()
    
    count = 0;

    for iteration, (image_list, label) in enumerate(dataset):
        pred_label = []
        batch_num = int(len(image_list) / args.batch_size)
        remain = int(len(image_list) % args.batch_size)
        for index in range(batch_num):
            image_list_set = image_list[index * args.batch_size : (index + 1) * args.batch_size]
            image_set = torch.stack(image_list_set, 0)
            image_set = image_set.cuda()
            outputs = model(image_set)
            probs = outputs.sigmoid()
            prediction = probs.ge(0.5)
            pred_label.append(prediction.cpu())
        if remain != 0:
            image_list_set = image_list[batch_num * args.batch_size : ]
            image_set = torch.stack(image_list_set, 0)
            image_set = image_set.cuda()
            outputs = model(image_set)
            probs = outputs.sigmoid()
            prediction = probs.ge(0.5)
            if remain == 1:
                prediction = prediction.unsqueeze(0)
            pred_label.append(prediction.cpu())

        pred_cls_label = torch.cat(pred_label, dim=0)
        grid_num_sum = pred_cls_label.size()[0] * pred_cls_label.size()[1]
        score = torch.sum(pred_cls_label)

        if torch.equal(label, torch.ones((1,), dtype=torch.uint8)):
            image_label = 'A'
        else:
            image_label = 'B'

        number_A = int(score)
        number_B = grid_num_sum - number_A

        with open(args.save_path, 'a+') as csvfile:
            csv_writer = csv.writer(csvfile)
            data_row = [number_A, number_B, image_label]
            csv_writer.writerow(data_row)
            count += 1
            logger.info(count)
            


if __name__ == '__main__':
    run()
