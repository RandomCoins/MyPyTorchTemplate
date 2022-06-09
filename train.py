import torch
import argparse
import os
import builtins
from utils import setup_seed, get_logger, AverageMeter, get_lr, save_model, get_remain_time, ProgressMeter
from model import Model
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from dataset import TrainSet, ValidationSet
from tensorboardX import SummaryWriter
from scipy.stats import spearmanr
import time


def train(args, model, optimizer, train_loader, epoch, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    loss_meter = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, loss_meter],
        prefix='Epoch: [{}]'.format(epoch)
    )

    model.train()
    max_iter = len(train_loader)

    end = time.time()
    for idx, (image, label) in enumerate(train_loader):
        n = image.size(0)

        if args.gpu:
            image = image.cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)
        else:
            image = image.cuda()
            label = label.cuda()

        output = model(image)
        optimizer.zero_grad()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % 5 == 0:
            progress.display(idx)

    return loss_meter.avg


def validation(model, test_loader, criterion):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    loss_meter = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, loss_meter],
        prefix='Validation: '
    )

    predict = []
    labels = []

    end = time.time()
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image = image.cuda()
            label = label.cuda()
            output = model(image)
            loss = criterion(output, label)
            loss_meter.update(loss.item())

            predict.append(output.view(-1).detach().item())
            labels.append(label.view(-1).detach().item())

            batch_time.update(time.time() - end)
            end = time.time()
            if i % 5 == 0:
                progress.display(i)

    corr, _ = spearmanr(predict, labels)
    return loss_meter.avg, corr


def main():
    args = parser.parse_args()
    logger = get_logger(os.path.join(args.log_path, args.name, 'log.txt'))
    logger.info(vars(args))
    setup_seed(args.seed)
    torch.set_num_threads(1)
    os.makedirs(os.path.join(args.log_dir, args.name, 'tensorboard'), exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, args.name, 'checkpoint'), exist_ok=True)

    ngpus_per_node = torch.cuda.device_count()
    args.distributed = ngpus_per_node > 1 and args.multiprocess_distributed
    if args.distributed:
        args.world_size = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, logger, ngpus_per_node))
    else:
        main_worker(args.gpu, args, logger, ngpus_per_node)


def main_worker(rank, args, logger, ngpus_per_node):
    args.gpu = rank
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    if args.gpu:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * torch.cuda.device_count() + rank
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:10001', world_size=args.world_size, rank=args.rank)

    print('==> Building model..')
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    start_epoch = 0

    if args.distributed:
        if args.gpu:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model = model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), f'No checkpoint found at {args.resume}'
        checkpoint = torch.load(args.resume, map_location='cuda')
        if args.distributed:
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

    print('==> Preparing data..')
    train_set = TrainSet(args.train_data)
    batch_size = args.batch_size
    if args.distributed:
        batch_size = batch_size // torch.cuda.device_count()
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=args.num_workers,
                              shuffle=(train_sampler is None), drop_last=True, pin_memory=False, sampler=train_sampler)

    val_set = ValidationSet(args.val_data)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=args.num_workers)

    writer = SummaryWriter(args.tensorboard)
    criterion = torch.nn.MSELoss().cuda()

    for epoch in range(start_epoch, start_epoch + args.epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss = train(args, model, optimizer, train_loader, epoch, criterion)
        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            writer.add_scalar('Train/Loss', train_loss, global_step=epoch)
            writer.add_scalar('Train/LearningRate', scalar_value=get_lr(optimizer), global_step=epoch)
            logger.info('train - epoch: {}, loss: {}'.format(epoch, train_loss))
        if epoch % 5 == 0:
            val_loss, val_srocc = validation(model, val_loader, criterion)
            if not args.multiprocessing_distributed or (
                    args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                writer.add_scalar('Test/Loss', val_loss, global_step=epoch)
                writer.add_scalar('Test/SROCC', val_srocc, global_step=epoch)
                logger.info('validation - epoch: {}, loss: {:.4f}, SROCC: {:.4f}'.format(epoch, val_loss, val_srocc))
        writer.flush()
        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_model(model, optimizer, epoch, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data', type=str)
    parser.add_argument('--val_data', type=str)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--lr', type=float)

    parser.add_argument('--log_dir', type=str, default='experiments', help='The log path saved')
    parser.add_argument('--resume', type=str, default=None,
                        help='The checkpoint path used to continue the train')
    parser.add_argument('--num_workers', type=int, default=4, help='The num of thread used to load data')

    parser.add_argument('--rank', type=int, default=-1)
    parser.add_argument('--multiprocess_distributed', action='store_true')
    main()
