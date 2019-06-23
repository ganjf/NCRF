import os
import argparse
import logging
import json
import time

import torch
from torch.utils.data import DataLoader
from torch.nn import BCELoss, DataParallel
from torch.optim import SGD

from tensorboardX import SummaryWriter

from wsi.data.DiscriminatePatchDataset import DiscriminatePatchDataset
from wsi.model.DiscriminatePatch import DiscriminatePatch
from wsi.model import MODELS

def get_parser():
    parser = argparse.ArgumentParser(description='Train model first stage')
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


def train_epoch(summary, summary_writer, cfg, model_classification, model_auxiliary, loss_fn, optimizer, dataloader):

    model_auxiliary.train()
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    fileHanlder = logging.FileHandler('train_auxiliary.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHanlder.setFormatter(formatter)
    logger.addHandler(fileHanlder)

    batch_size = cfg['batch_size_auxiliary']
    grid_size = cfg['grid_size']
    time_now = time.time()
    for step,(grid_patch, grid_score, br, label) in enumerate(dataloader):

        grid_patch, grid_score, br, label = grid_patch.float().cuda(), grid_score.cuda(), br.float().cuda(), label.cuda()
        output = model_classification(grid_patch)
        grid_probs = output.sigmoid()
        grid_predicts = grid_probs.ge(0.5)
        patch_probs = torch.mul(grid_predicts.float(), grid_score)
        patch_probs = torch.sum(patch_probs, dim=1)
        patch_probs = patch_probs.unsqueeze(dim=1)
        patch_predicts = patch_probs.ge(0.5)
        discriminate_label = (patch_predicts == label)

        patch_score_discriminate = model_auxiliary(br)
        patch_score_discriminate_predicts = patch_score_discriminate.ge(0.5)
        loss = loss_fn(patch_score_discriminate, discriminate_label.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_data = float((patch_score_discriminate_predicts == discriminate_label).sum()) * 1.0 / (batch_size)

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


def run(args):
    with open(args.cfg_path) as f:
        cfg = json.load(f)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    batch_size_train = cfg['batch_size_auxiliary']
    num_workers = args.num_workers
    grid_size = cfg['grid_size']

    model_classification = MODELS[cfg['model']](num_nodes=grid_size, use_crf=cfg['use_crf'])
    model_classification = DataParallel(model_classification, device_ids=None)
    model_classification = model_classification.cuda()
    model_classification.eval()

    model_auxiliary = DiscriminatePatch()
    model_auxiliary = DataParallel(model_auxiliary, device_ids=None)
    model_auxiliary = model_auxiliary.cuda()

    loss_fn = BCELoss().cuda()
    optimizer = SGD(model_auxiliary.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])

    dataset_train = DiscriminatePatchDataset(csv_file=cfg['data_path_train'],
                                           root_dir=cfg['root_dir'])

    dataloader_train = DataLoader(dataset_train,
                                batch_size=batch_size_train,
                                shuffle=True,
                                num_workers=num_workers)

    summary_train = {'epoch': 0, 'step': 0}
    summary_writer = SummaryWriter(comment='NCRF_Auxiliary')

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

        summary_train = train_epoch(summary=summary_train, summary_writer=summary_writer, cfg=cfg,
                                    model_classification=model_classification, model_auxiliary=model_auxiliary,
                                    loss_fn=loss_fn, optimizer=optimizer,
                                    dataloader=dataloader_train)
        fileName = 'train_' + str(summary_train['epoch']) + '.pkl'
        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'state_dict': model_auxiliary.state_dict()},
                   os.path.join(args.save_path, fileName))
    summary_writer.close()


def main():
    logging.basicConfig(level=logging.INFO)

    args = get_parser()
    run(args)


if __name__ == '__main__':
    main()
