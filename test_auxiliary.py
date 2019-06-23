import torch
import json
from wsi.data.ScoringTestDatasetAuxiliary import ScoringTestDataset
from torch.nn import DataParallel
from wsi.model import MODELS
from wsi.model.DiscriminatePatch import DiscriminatePatch
import argparse
import time
import logging
from tensorboardX import SummaryWriter
from sklearn import metrics
import numpy
import matplotlib.pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--load_path_cls', default=None,
                        type=str,
                        help='Path to the load models')
    parser.add_argument('--load_path_auxiliary', default=None,
                        type=str,
                        help='Path to the load models')
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
    fileHanlder = logging.FileHandler('test_auxiliary.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHanlder.setFormatter(formatter)
    logger.addHandler(fileHanlder)

    dataset = ScoringTestDataset(csv_file=args.test_root, root_dir=args.base_root)

    summary = {'count': 0, 'correct': 0, 'acc': 0}
    summary_writer = SummaryWriter(comment='TEST_AUXILIARY')


    model_cls = MODELS[cfg['model']](num_nodes=cfg['grid_size'], use_crf=cfg['use_crf'])
    model_cls = DataParallel(model_cls, device_ids=None)
    checkpoint = torch.load(args.load_path_cls)
    model_cls.load_state_dict(checkpoint['state_dict'])
    model_cls = model_cls.cuda()
    model_cls.eval()

    model_auxiliary = DiscriminatePatch()
    model_auxiliary = DataParallel(model_auxiliary, device_ids=None)
    checkpoint = torch.load(args.load_path_auxiliary)
    model_auxiliary.load_state_dict(checkpoint['state_dict'])
    model_auxiliary = model_auxiliary.cuda()
    model_auxiliary.eval()

    time_now = time.time()
    y_label = []
    y_pred = []

    for iteration, (image_list, image_auxiliary_list, score_list, label) in enumerate(dataset):
        pred_label = []
        auxiliary_score = []
        batch_num = int(len(image_list) / args.batch_size)
        remain = int(len(image_list) % args.batch_size)
        image_cls_score = torch.stack(score_list)
        for index in range(batch_num):
            image_list_set = image_list[index * args.batch_size : (index + 1) * args.batch_size]
            image_set = torch.stack(image_list_set, 0)
            image_set = image_set.cuda()
            outputs = model_cls(image_set)
            probs = outputs.sigmoid()
            prediction = probs.ge(0.5)
            pred_label.append(prediction.cpu())
            image_auxiliary_list_set = image_auxiliary_list[index * args.batch_size: (index + 1) * args.batch_size]
            image_auxiliary_set = torch.stack(image_auxiliary_list_set, 0)
            image_auxiliary_set = image_auxiliary_set.cuda()
            patch_score = model_auxiliary(image_auxiliary_set)
            patch_score = patch_score.ge(0.5)
            auxiliary_score.append((patch_score.cpu()))
        if remain != 0:
            image_list_set = image_list[batch_num * args.batch_size : ]
            image_set = torch.stack(image_list_set, 0)
            image_set = image_set.cuda()
            outputs = model_cls(image_set)
            probs = outputs.sigmoid()
            prediction = probs.ge(0.5)
            image_auxiliary_list_set = image_auxiliary_list[batch_num * args.batch_size : ]
            image_auxiliary_set = torch.stack(image_auxiliary_list_set, 0)
            image_auxiliary_set = image_auxiliary_set.cuda()
            patch_score = model_auxiliary(image_auxiliary_set)
            patch_score = patch_score.ge(0.5)
            if remain == 1:
                prediction = prediction.unsqueeze(0)
                patch_score = patch_score.unsqueeze(0)
            pred_label.append(prediction.cpu())
            auxiliary_score.append((patch_score.cpu()))

        pred_cls_label = torch.cat(pred_label, dim=0).float()
        patch_auxiliary_score = torch.cat(auxiliary_score, dim=0).float()
        patch_discriminate_num = float(torch.sum(patch_auxiliary_score))
        patch_score = torch.mul(pred_cls_label, image_cls_score)
        score = torch.sum(patch_score, dim=1)
        score = score.unsqueeze(dim=1)
        finally_score = torch.sum(torch.mul(score, patch_auxiliary_score))
        ratio = float(finally_score) / float(patch_discriminate_num)

        if ratio >= 0.5:
            cls_label = torch.ones((1,), dtype=torch.uint8)
        else:
            cls_label = torch.zeros((1,), dtype=torch.uint8)
        if torch.equal(cls_label, label):
            summary['correct'] += 1

        logger.info(cls_label)
        logger.info(label)
        logger.info('score: {:.4f} / patch_num: {} = {:.4f}'.format(float(finally_score), patch_discriminate_num, ratio))

        summary['count'] += 1
        summary['acc'] = float(summary['correct']) / float(summary['count'])
        summary_writer.add_scalar(
            'test/acc', summary['acc'], summary['count'])
        logger.info('{}, Numbers of all WSI: {}, Number of the correct WSI classification: {}, '
                    'Accuracy: {:.4f}'.format(
            time.strftime("%Y-%m-%d %H:%M:%S"), summary['count'], summary['correct'], summary['acc']))
        y_label.append(label.cpu())
        y_pred.append(float(ratio))

    y_label_array = numpy.array(y_label)
    y_pred_array = numpy.array(y_pred)

    fpr, tpr, threshold = metrics.roc_curve(y_true=y_label_array, y_score=y_pred_array, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    logger.info('AUC = {:.4f}'.format(auc))
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC Curve(area = %0.4f' % auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()



if __name__ == '__main__':
    run()
