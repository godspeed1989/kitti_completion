import os
import logging
import argparse
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from kitti_dataset import use_norm_depth, readlines
from kitti_dataset import KITTIRAWDataset, DAT_VAL_TEST, _merge_batch
from options import CompletionOptions
from sparse_model import Model
from metric import Metrics, AverageMeter

def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'

    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def define_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("The requested optimizer: {} is not implemented".format(optim))
    return optimizer


def define_lr_scheduler(optimizer, args):
    if args.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - args.niter) / float(args.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=args.lr_decay_iters, gamma=args.gamma)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   factor=args.gamma,
                                                   threshold=0.0001,
                                                   patience=args.lr_decay_iters)
    elif args.lr_policy == 'none':
        scheduler = None
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def save_model(model, path, epoch, mse, rmse):
    save_folder = os.path.join(path, "models")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = os.path.join(save_folder, "weights_{}.pth".format(epoch))
    to_save = {
        'mae': mae,
        'rmse': rmse,
        'state_dict': model.state_dict()
    }
    torch.save(to_save, save_path)

def log_time(epoch, batch_idx,
             start_time, duration,
             total_batches, batch_size, print_dict=None):
    def sec_to_hm(t):
        """Convert time in seconds to time in hours, minutes and seconds
        e.g. 10239 -> (2, 50, 39)
        """
        t = int(t)
        s = t % 60
        t //= 60
        m = t % 60
        t //= 60
        return t, m, s
    def sec_to_hm_str(t):
        """Convert time in seconds to a nice string
        e.g. 10239 -> '02h50m39s'
        """
        h, m, s = sec_to_hm(t)
        return "{:02d}h{:02d}m{:02d}s".format(h, m, s)
    samples_per_sec = batch_size / duration
    time_sofar = time.time() - start_time
    step = batch_idx + 1
    training_time_left = (total_batches / step - 1.0) * time_sofar
    print_string = "epoch {:>3} | batch {}/{} | examples/s: {:5.1f} | " + \
                   "time elapsed: {} | time left: {}"
    print(print_string.format(epoch, batch_idx, total_batches, samples_per_sec,
                              sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)), end=' ')
    if print_dict:
        for k,v in print_dict.items(): print("| {}: {}".format(k,v), end=' ')
    print('')


def MSE_loss(prediction, gt, weighted_map=None):
    err = prediction[:,0:1] - gt
    mask = (gt > 0).detach()
    if weighted_map is not None:
        weighted_mse_loss = 0
        max_val = torch.max(weighted_map).int()
        for i in range(1, max_val + 1):
            weighted_mse_loss += torch.mean(err[weighted_map == i]**2)
        return weighted_mse_loss / i
    mse_loss = torch.sum((err[mask])**2) / mask.sum()
    return mse_loss

def smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # don't ask me why it works better
    return loss

def train_epoch(epoch, max_epoch, G_model: Model,
                loader, optimizer, device, tb_log, cfg):
    start_time = time.time()
    batch_size = loader.batch_size

    G_model.train()
    for batch_idx, input in enumerate(loader):
        it = epoch * len(loader) + batch_idx
        before_op_time = time.time()

        optimizer.zero_grad()

        rgb = input['color_aug'].to(device) * 255.
        depth = input['depth_aug'].to(device)
        gtdepth = input['depth_sd_gt'].to(device)

        if cfg.colorize:
            depth_color = input['depth_color'].to(device) * 255.
            depth_in = torch.cat([depth, depth_color], 1)
        else:
            depth_in = depth

        mask = (depth > 0).float()
        pred_depth, pred_img = G_model(depth_in, mask, rgb)

        weight_map = torch.clamp(torch.ceil(gtdepth / cfg.max_depth), min=0, max=1)
        loss_depth = MSE_loss(pred_depth, gtdepth, weight_map)
        loss_smooth = smooth_loss(pred_depth) * cfg.weight_smooth_loss
        if cfg.reconstruction_loss:
            gray = input['gray'].to(device)
            loss_img = F.l1_loss(pred_img, gray)
            tb_log.add_scalar('loss_img', loss_img.item(), it)
        else:
            loss_img = 0

        loss = loss_depth + loss_smooth + loss_img
        tb_log.add_scalar('loss', loss.item(), it)
        tb_log.add_scalar('loss_depth', loss_depth.item(), it)
        tb_log.add_scalar('loss_smooth', loss_smooth.item(), it)
        loss.backward()

        optimizer.step()

        duration = time.time() - before_op_time

        if batch_idx % 10 == 0:
            print_dict = {}
            print_dict['loss'] = '{:.5f}'.format(loss.item())
            log_time(epoch, batch_idx, start_time, duration,
                    len(loader), batch_size, print_dict)
        if batch_idx % cfg.log_frequency == 0:
            def to_img(tensor, itype):
                # convert float tensor to tensorboardX supported image
                img = tensor.detach().cpu().numpy()
                if itype == 'depth':
                    img = img[0]
                    if use_norm_depth:
                        img = np.clip(img, 0, 1.0)
                    else:
                        img = np.clip(img, cfg.min_depth, cfg.max_depth)
                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                    img = 255 * plt.cm.jet(img)[:, :, :3]  # H, W, C
                    img = np.transpose(img, [2, 0, 1])
                    return img.astype('uint8')
                elif itype == 'gray':
                    return np.clip(img, 0, 1)
                elif itype == 'color':
                    return np.clip(img, 0, 255) / 255.0
            tb_log.add_image('rgb', to_img(rgb[0], 'color'), it)
            tb_log.add_image('sparse_depth', to_img(depth[0], 'depth'), it)
            tb_log.add_image('pred_depth', to_img(pred_depth[0], 'depth'), it)
            tb_log.add_image('gt_depth', to_img(gtdepth[0], 'depth'), it)
            if cfg.reconstruction_loss:
                tb_log.add_image('pred_img', to_img(pred_img[0], 'gray'), it)


def validate(model, val_loader, device, min_depth, max_depth, cfg):
    model.eval()

    metric = Metrics(max_depth=max_depth)
    score = AverageMeter()
    score_1 = AverageMeter()
    with torch.no_grad():
        for _, inputs in tqdm(enumerate(val_loader)):
            rgb = inputs['color'].to(device) * 255.
            sdepth = inputs['depth_gt'].to(device)

            if cfg.colorize:
                depth_color = inputs['depth_color'].to(device) * 255.
                depth_in = torch.cat([sdepth, depth_color], 1)
            else:
                depth_in = sdepth

            mask = (sdepth > 0).float()
            output, _ = model(depth_in, mask, rgb)
            if use_norm_depth:
                output = torch.clamp(output, 0, 1.0)
                output = min_depth + output * (max_depth - min_depth)
            else:
                output = torch.clamp(output, min_depth, max_depth)
            output = output[:,0:1].detach().cpu()

            gt = inputs['depth_sd_gt']
            metric.calculate(output, gt)
            score.update(metric.get_metric('mae'), metric.num)
            score_1.update(metric.get_metric('rmse'), metric.num)

    model.train()
    return score.avg, score_1.avg


if __name__ == "__main__":
    timestr = time.strftime("%b-%d_%H-%M-%S", time.localtime())
    root_result_dir = os.path.join('./output-' + timestr)
    os.makedirs(root_result_dir, exist_ok=True)
    # log to file
    log_file = os.path.join(root_result_dir, 'log_train.txt')
    logger = create_logger(log_file)
    logger.info('***************Start logging***************')

    options = CompletionOptions()
    args = options.parse()

    # copy important files to backup
    os.system('cp *.py %s/' % root_result_dir)
    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))

    device = torch.device("cpu" if args.no_cuda else "cuda")

    G_mod = Model(scales=4, base_width=32, dec_img=args.reconstruction_loss, colorize=args.colorize)
    G_mod = G_mod.to(device)

    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path)
        G_mod.load_state_dict(checkpoint['state_dict'])
        logger.info("use save model: {}".format(args.weight_path))
        logger.info("saved model: mae {} rmse {}".format(checkpoint['mae'], checkpoint['rmse']))

    # optimizer & lr scheduler
    optimizer = define_optim(args.optimizer,
                            [{'params':G_mod.parameters()},],
                            args.learning_rate, args.weight_decay)
    scheduler = define_lr_scheduler(optimizer, args)

    img_ext = '.png' if args.png else '.jpg'
    if args.split == 'tiny':
        fpath = os.path.join(os.path.dirname(__file__), "splits/tiny", "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        train_dataset = KITTIRAWDataset(
            args.data_path, train_filenames, args.crop_h, args.crop_w,
            img_ext=img_ext, min_depth=args.min_depth, max_depth=args.max_depth)
        val_dataset = KITTIRAWDataset(
            args.data_path, val_filenames, args.crop_h, args.crop_w,
            img_ext=img_ext, min_depth=args.min_depth, max_depth=args.max_depth)
    elif args.split == 'full':
        fpath = os.path.join(os.path.dirname(__file__), "splits/", "{}.txt")
        train_filenames = readlines(fpath.format("train"))
        train_dataset = KITTIRAWDataset(
            args.data_path, train_filenames, args.crop_h, args.crop_w,
            img_ext=img_ext, min_depth=args.min_depth, max_depth=args.max_depth)
        val_dataset = DAT_VAL_TEST(
            args.data_path, is_test=False, crop_h=args.crop_h, crop_w=args.crop_w,
            min_depth=args.min_depth, max_depth=args.max_depth)
    logger.info("Using split:  {}".format(args.split))
    logger.info("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

    # dataloader
    loaders = {}
    train_loader = DataLoader(
            train_dataset, args.batch_size, shuffle=True, collate_fn=_merge_batch,
            num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=None)
    val_loader = DataLoader(
            val_dataset, args.batch_size, shuffle=True, collate_fn=_merge_batch,
            num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=None)

    tb_log = SummaryWriter(logdir=os.path.join(root_result_dir, 'tensorboard'))
    step = 0
    start_time = time.time()
    for e in range(args.num_epochs):
        logging.info("Epoch {}".format(e+1))
        train_epoch(e, args.num_epochs, G_mod,
                    train_loader, optimizer, device, tb_log, args)
        mae, rmse = validate(G_mod, val_loader, device, args.min_depth, args.max_depth, args)
        logger.info("Epoch {} MAE:{:.4f} RMSE:{:.4f}".format(e+1, mae, rmse))
        tb_log.add_scalar('MAE', mae, e+1)
        tb_log.add_scalar('RMSE', rmse, e+1)

        if (e + 1) % args.save_frequency == 0:
            logger.info("save weights after {} epochs".format(e+1))
            save_model(G_mod, root_result_dir, e + 1, mae, rmse)

        if args.lr_policy != 'plateau':
            scheduler.step()
        elif args.lr_policy == 'plateau':
            scheduler.step(rmse)
        logger.info("lr is set to {}".format(optimizer.param_groups[0]['lr']))
