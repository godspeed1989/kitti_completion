from __future__ import absolute_import, division, print_function

import os
import time
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in
timestr = time.strftime("%b-%d_%H-%M-%S", time.localtime())

class CompletionOptions:
    def __init__(self, name='output'):
        self.parser = argparse.ArgumentParser(description="CMD options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data_tiny"))

        # DATA options
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["tiny", "full"],
                                 default="tiny")
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="which dataset to use",
                                 choices=["kitti", "nyudepth"],
                                 default="kitti")
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--crop_h",
                                 type=int,
                                 help="input image height",
                                 default=256)
        self.parser.add_argument("--crop_w",
                                 type=int,
                                 help="input image width",
                                 default=1216)
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=85.0)

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=2)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-3)
        self.parser.add_argument('--optimizer',
                                 type=str,
                                 default='adam',
                                 help='adam or sgd')
        self.parser.add_argument('--weight_decay',
                                 type=float,
                                 default=0,
                                 help='L2 weight decay/regularisation on?')
        self.parser.add_argument('--lr_policy',
                                 type=str,
                                 default='plateau',
                                 help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--niter',
                                 type=int,
                                 default=50,
                                 help='Number of iter at starting learning rate (for lambda lr policy)')
        self.parser.add_argument('--lr_decay_iters',
                                 type=int,
                                 default=3,
                                 help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--gamma',
                                 type=float, default=0.5,
                                 help='factor to decay learning rate every lr_decay_iters with')
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=80)
        self.parser.add_argument("--weight_init",
                                 type=str,
                                 default="kaiming",
                                 help="normal, xavier, kaiming, orhtogonal weights initialisation")

        # ABLATION options
        self.parser.add_argument("--reconstruction_loss",
                                 help="if set enable img reconstruction loss",
                                 action="store_true")
        self.parser.add_argument("--weight_smooth_loss",
                                 type=float, default=0.1,
                                 help='weight of smooth loss')

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=4)
        # For multiple GPUs training on a single machine with SyncBatchnorm
        self.parser.add_argument("--mgpus",
                                 help="if set enable multi GPU",
                                 action="store_true")
        self.parser.add_argument("--ngpu",
                                 type=int,
                                 help="number of GPUs",
                                 default=2)
        self.parser.add_argument("--local_rank",
                                 type=int,
                                 help="the master GPU number",
                                 default=0)

        # LOADING options
        self.parser.add_argument("--weight_path",
                                 type=str,
                                 help="path of model to load")
        self.parser.add_argument("--dump",
                                 help="if set dump test data",
                                 action="store_true")

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=500)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

    def parse(self, args=None):
        self.options = self.parser.parse_args(args=args)
        return self.options
