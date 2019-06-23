import argparse
import json
import torch
import time
import logging
from wsi.data.ImageTestDataset import ImageTestDataset
from wsi.model import MODELS
from wsi.model.logistRegression import LogistRegression
from torch.nn import DataParallel
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy



def get_parser():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--load_path_cnn', default=None,
                        type=str,
                        help='Path to the load first models')
    parser.add_argument('--load_path_lr', default=None,
                        type=str,
                        help='Path to the load second models')
    parser.add_argument('--test_root', default=None,
                        type=str,
                        help='Path to load image for test')
    parser.add_argument('--base_root', default=None,
                        type=str,
                        help='base root for image')
    parser.add_argument('--batch_size', default=32,
                        type=int,
                        help='the batch size to calculate the image with model')
    parser.add_argument('--cfg_path', default=None,
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
    fileHanlder = logging.FileHandler('test_lr_auc.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHanlder.setFormatter(formatter)
    logger.addHandler(fileHanlder)

    dataset = ImageTestDataset(csv_file=args.test_root, root_dir=args.base_root)

    model_cnn = MODELS[cfg['model']](num_nodes=cfg['grid_size'], use_crf=cfg['use_crf'])
    model_cnn = DataParallel(model_cnn, device_ids=None)
    checkpoint_cnn = torch.load(args.load_path_cnn)
    model_cnn.load_state_dict(checkpoint_cnn['state_dict'])
    model_cnn = model_cnn.cuda()
    model_cnn.eval()
    model_lr = LogistRegression()
    model_lr = DataParallel(model_lr, device_ids=None)
    checkpoint_lr = torch.load(args.load_path_lr)
    model_lr.load_state_dict(checkpoint_lr['state_dict'])
    model_lr.cuda()
    model_lr.eval()

    summary = {'count': 0, 'correct': 0, 'acc': 0}
    y_pred = []
    y_label = []


    for iteration, (image_list, label) in enumerate(dataset):
        time_now = time.time()
        pred_label = []
        label = label.cuda()
        batch_num = int(len(image_list) / args.batch_size)
        remain = int(len(image_list) % args.batch_size)
        for index in range(batch_num):
            image_list_set = image_list[index * args.batch_size : (index + 1) * args.batch_size]
            image_set = torch.stack(image_list_set, 0)
            image_set = image_set.cuda()
            outputs = model_cnn(image_set)
            probs = outputs.sigmoid()
            prediction = probs.ge(0.5)
            pred_label.append(prediction.cpu())
        if remain != 0:
            image_list_set = image_list[batch_num * args.batch_size : ]
            image_set = torch.stack(image_list_set, 0)
            image_set = image_set.cuda()
            outputs = model_cnn(image_set)
            probs = outputs.sigmoid()
            prediction = probs.ge(0.5)
            if remain == 1:
                prediction = prediction.unsqueeze(0)
            pred_label.append(prediction.cpu())

        pred_cls_label = torch.cat(pred_label, dim=0)
        grid_num_sum = pred_cls_label.size()[0] * pred_cls_label.size()[1]
        score = torch.sum(pred_cls_label)

        number_A = int(score)
        number_B = grid_num_sum - number_A

        A = torch.full((1,), number_A)
        B = torch.full((1,), number_B)
        histogram = torch.cat((A,B), dim=0)

        output = model_lr(histogram)
        pred_cls = output.ge(0.5)
        summary['count'] += 1
        if torch.equal(pred_cls, label):
            summary['correct'] += 1
        time_spent = time.time() - time_now
        summary['acc'] = float(summary['correct']) / float(summary['count'])
        logger.info('{}, Numbers of all WSI: {}, Number of the correct WSI classification: {}, '
                    'Accuracy: {:.4f}, Running time: {:.4f}'.format(
            time.strftime("%Y-%m-%d %H:%M:%S"), summary['count'], summary['correct'], summary['acc'], time_spent))

        y_pred.append(float(output.cpu()))
        y_label.append(label.cpu())
        
    y_pred_array = numpy.array(y_pred)
    y_label_array = numpy.array(y_label)

    fpr,tpr, threshold = metrics.roc_curve(y_true=y_label_array, y_score=y_pred_array, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    logger.info('AUC = {:.4f}'.format(auc))
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC Curve(area = %0.4f'% auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

if __name__ == '__main__':
    run()

