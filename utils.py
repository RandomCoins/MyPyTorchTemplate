import logging
import torch
import numpy as np
import random
import os


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def get_logger(logpath, resume=False):
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s', "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if resume:
        mode = 'a'
    else:
        mode = 'w'
    info_file_handler = logging.FileHandler(logpath, mode=mode)
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(formatter)
    logger.addHandler(info_file_handler)
    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_lr(optimizer):
    state_dict = optimizer.state_dict()
    lr = state_dict['param_groups'][0]['lr']
    return lr


def save_model(model, optimizer, epoch, args, is_last=False):
    save_path = os.path.join(args.log_dir, args.name, 'checkpoint')
    if is_last:
        save_path = os.path.join(save_path, 'final.pth')
    else:
        save_path = os.path.join(save_path, 'epoch_{}.pth'.format(epoch))
    state = {
        'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    logging.info(f'model save to {save_path}')
    torch.save(state, save_path)