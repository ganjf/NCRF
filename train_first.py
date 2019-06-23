import sys
import os
import argparse
import logging
import json
import time

import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.optim import SGD

from tensorboardX import SummaryWriter

from wsi.data.TrainDataset import GridPatchTrainDataset
from wsi.data.ValidDataset import GridPatchValidnDataset
from wsi.model import MODELS

def get_parser():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--cfg_path', default='/configs/resnet18_crf.json',
                        type=str,
                        help='Path to the config file in json format')
    parser.add_argument('--save_path', default=None,
                        type=str,
                        help='Path to the saved models')
    parser.add_argument('--load_path', default=None,
                        type=str,
                        help='Path to the load models')
    parser.add_argument('--num_workers', default=2, type=int, help='number of'
                        ' workers for each data loader, default 2.')
    args = parser.parse_args()
    return args


def train_epoch(summary, summary_writer, cfg, model, loss_fn, optimizer, dataloader):

    model.train()
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    fileHanlder = logging.FileHandler('train.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHanlder.setFormatter(formatter)
    logger.addHandler(fileHanlder)

    batch_size = cfg['batch_size']
    grid_size = cfg['grid_size']
    time_now = time.time()
    for step,(patch ,label) in enumerate(dataloader):
        patch, label = patch.float().cuda(), label.cuda()

        output = model(patch)
        loss = loss_fn(output, label.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = output.sigmoid()
        predicts = probs.ge(0.5)
        acc_data = float((predicts == label).sum()) * 1.0 / (
            batch_size * grid_size )

        loss_data = loss.item()

        time_spent = time.time() - time_now
        time_now = time.time()
        logger.info(
            '{}, Epoch : {}, Step : {}, Training Loss : {:.5f}, '
            'Training Acc : {:.3f}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), summary['epoch'] + 1,
                step, loss_data, acc_data, time_spent))

        summary['step'] += 1

        if summary['step'] % cfg['log_every'] == 0:
            summary_writer.add_scalar('train/loss', loss_data, summary['step'])
            summary_writer.add_scalar('train/acc', acc_data, summary['step'])

    summary['epoch'] += 1
    
    

    return summary


def valid_epoch(summary, cfg, model, loss_fn,dataloader):

    model.eval()
    grid_size = cfg['grid_size']
    crop_size = cfg['crop_size']
    steps = len(dataloader)

    loss_sum = 0
    acc_sum = 0

    for step, (image, label) in enumerate(dataloader):

        image = image.view(-1, grid_size, 3, crop_size, crop_size)
        label = label.view(-1, grid_size)
        image, label = image.float().cuda(), label.cuda()

        output = model(image)
        loss = loss_fn(output, label.float())

        probs = output.sigmoid()
        predicts = probs.ge(0.5)
        acc_data = float((predicts == label).sum()) * 1.0 / (
                image.size()[0] * grid_size)
        loss_data = loss.item()

        loss_sum += loss_data
        acc_sum += acc_data
        print('----------------------------')

    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps

    return summary


def run(args):
    with open(args.cfg_path) as f:
        cfg = json.load(f)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    batch_size_train = cfg['batch_size']
    batch_size_valid = cfg['batch_size'] * 3
    num_workers = args.num_workers
    grid_size = cfg['grid_size']

    logger = logging.getLogger("valid")
    logger.setLevel(logging.DEBUG)
    fileHanlder = logging.FileHandler('valid.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHanlder.setFormatter(formatter)
    logger.addHandler(fileHanlder)

    model = MODELS[cfg['model']](num_nodes=grid_size, use_crf=cfg['use_crf'])
    model = DataParallel(model, device_ids=None)
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()

    loss_fn = BCEWithLogitsLoss().cuda()
    optimizer = SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])

    dataset_train = GridPatchTrainDataset(csv_file=cfg['data_path_train'],
                                           root_dir=cfg['root_dir'])

    dataset_valid = GridPatchValidnDataset(csv_file=cfg['data_path_valid'],
                                        root_dir=cfg['root_dir'])


    dataloader_train = DataLoader(dataset_train,
                                batch_size=batch_size_train,
                                  shuffle=True,
                                num_workers=num_workers)
    dataloader_valid = DataLoader(dataset_valid,
                                        batch_size=batch_size_valid,
                                        num_workers=num_workers)


    summary_train = {'epoch': 0, 'step': 0}
    summary_valid = {'loss': float('inf'), 'acc': 0}
    summary_writer = SummaryWriter(comment='NCRF')
    loss_valid_best = 1000.0

    for epoch in range(cfg['epoch']):
        if epoch == 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg['lr_1']
        if epoch == 10:
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg['lr_2']
        if epoch == 15:
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg['lr_3']
    
        summary_train = train_epoch(summary_train, summary_writer, cfg, model,
                                    loss_fn, optimizer,
                                    dataloader_train)
        fileName = 'train_' + str(summary_train['epoch']) + '.pkl'
        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'state_dict': model.state_dict()},
                   os.path.join(args.save_path, fileName))

        time_now = time.time()
        summary_valid = valid_epoch(summary_valid, cfg, model, loss_fn,
                                    dataloader_valid)
        time_spent = time.time() - time_now

        logger.info(
            '{}, Epoch : {}, Step : {}, Validation Loss : {:.5f}, '
            'Validation Acc : {:.3f}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'],
                summary_train['step'], summary_valid['loss'],
                summary_valid['acc'], time_spent))

        summary_writer.add_scalar(
            'valid/loss', summary_valid['loss'], summary_train['step'])
        summary_writer.add_scalar(
            'valid/acc', summary_valid['acc'], summary_train['step'])

        if summary_valid['loss'] < loss_valid_best:
            loss_valid_best = summary_valid['loss']

            torch.save({'epoch': summary_train['epoch'],
                        'step': summary_train['step'],
                        'state_dict': model.state_dict()},
                       os.path.join(args.save_path, 'best.pkl'))

    summary_writer.close()


def main():
    logging.basicConfig(level=logging.INFO)

    args = get_parser()
    run(args)


if __name__ == '__main__':
    main()
