import torch
import argparse
import logging
from tensorboardX import SummaryWriter
import json
import os
from wsi.data.LogistRegressionDataset import LogistRegressionDataset
from wsi.model.logistRegression import LogistRegression
from torch.utils.data import DataLoader
from torch.nn import BCELoss, DataParallel
from torch.optim import SGD
import time

def get_parser():
    parser = argparse.ArgumentParser(description='Train model second stage')
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

def train_epoch(summary, summaryWriter, cfg, model, optimizer, loss_fn, dataloader, ):

    model.train()
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    fileHanlder = logging.FileHandler('train_second.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHanlder.setFormatter(formatter)
    logger.addHandler(fileHanlder)

    time_now = time.time()
    for step, (histogram, label) in enumerate(dataloader):
        histogram, label = histogram.cuda(), label.cuda()
        batch_size = histogram.size()[0]

        output = model(histogram)
        loss = loss_fn(output, label.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicts = output.ge(0.5)
        acc_data = float((predicts == label).sum()) * 1.0 / (batch_size)
        loss_data = loss.item()

        time_spent = time.time() - time_now
        time_now = time.time()
        logger.info(
            '{}, Epoch : {}, Step : {}, Training Loss : {:.5f}, '
            'Training Acc : {:.3f}, Run Time : {:.2f}'
                .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), summary['epoch'] + 1,
                step, loss_data, acc_data, time_spent))

        summary['iteration'] += 1

        if summary['iteration'] % cfg['log_every_second'] == 0:
            summaryWriter.add_scalar('train_second/loss', loss_data, summary['iteration'])
            summaryWriter.add_scalar('train_second/acc', acc_data, summary['iteration'])

    summary['epoch'] += 1

    return summary

def valid_epoch(summary, model, loss_fn, dataloader):

    model.eval()
    steps = len(dataloader)

    loss_sum = 0
    acc_sum = 0

    for step, (histogram, label) in enumerate(dataloader):

        histogram, label = histogram.cuda(), label.cuda()

        output = model(histogram)
        loss = loss_fn(output, label.float())

        predicts = output.ge(0.5)
        acc_data = float((predicts == label).sum()) * 1.0 / (histogram.size()[0])
        loss_data = loss.item()

        loss_sum += loss_data
        acc_sum += acc_data

    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps

    return summary


def run():
    args = get_parser()
    with open(args.cfg_path) as f:
        cfg = json.load(f)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    logger = logging.getLogger("train_second")
    logger.setLevel(logging.DEBUG)
    fileHanlder = logging.FileHandler('valid_second.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHanlder.setFormatter(formatter)
    logger.addHandler(fileHanlder)

    dataset_train = LogistRegressionDataset(csv_file=cfg['data_path_second_train'])
    dataset_valid = LogistRegressionDataset(csv_file=cfg['data_path_second_valid'])
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=cfg['batch_second_size'],
                                  shuffle=True, num_workers=args.num_workers)
    dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=1,
                                  shuffle=False, num_workers=args.num_workers)

    model = LogistRegression()
    model = DataParallel(model, device_ids=None)
    model = model.cuda()

    optimizer = SGD(model.parameters(), lr=cfg['lr_second'], momentum=cfg['momentum'])
    loss_fn = BCELoss().cuda()

    summary_train = {'epoch': 0, 'iteration': 0}
    summary_valid = {'loss': float('inf'), 'acc': 0}
    summary_writer = SummaryWriter(comment='LR')
    loss_valid_best = 1000.0

    for epoch in range(100):
        summary_train = train_epoch(summary=summary_train, summaryWriter=summary_writer, cfg=cfg,
                                    model=model, optimizer=optimizer,
                                    loss_fn=loss_fn, dataloader=dataloader_train)
        time_now = time.time()
        summary_valid = valid_epoch(summary=summary_valid, model=model, loss_fn=loss_fn,
                                    dataloader=dataloader_valid)
        time_spent = time.time() - time_now

        logger.info(
            '{}, Epoch : {}, Step : {}, Validation Loss : {:.5f}, '
            'Validation Acc : {:.3f}, Run Time : {:.2f}'
                .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'],
                summary_train['iteration'], summary_valid['loss'],
                summary_valid['acc'], time_spent))
        fileName = 'train_' + str(summary_train['epoch']) + '.pkl'
        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['iteration'],
                    'state_dict': model.state_dict()},
                   os.path.join(args.save_path, fileName))
        summary_writer.add_scalar(
            'valid_second/loss', summary_valid['loss'], summary_train['iteration'])
        summary_writer.add_scalar(
            'valid/_second_acc', summary_valid['acc'], summary_train['iteration'])

        if summary_valid['loss'] <= loss_valid_best:
            loss_valid_best = summary_valid['loss']
            torch.save({'epoch': summary_train['epoch'],
                        'step': summary_train['iteration'],
                        'state_dict': model.state_dict()},
                       os.path.join(args.save_path, 'best_second.pkl'))
    summary_writer.close()

if __name__ == '__main__':
    run()

