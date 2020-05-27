from __future__ import absolute_import, division, print_function

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from options import CompletionOptions
from kitti_dataset import DAT_VAL_TEST, plt_img, restore_depth, use_norm_depth
from sparse_model import Model


class TestC:
    def __init__(self, opt):
        super(TestC, self).__init__()
        self.opt = opt

        self.opt.batch_size = 1
        test_dataset = DAT_VAL_TEST(
            self.opt.data_path, is_test=True,
            min_depth=self.opt.min_depth, max_depth=self.opt.max_depth,
            crop_h=352, crop_w=1216)
        self.test_loader = DataLoader(
            test_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        print("There are {:d} test items".format(len(test_dataset)))

        self.crop_h = 256
        self.crop_w = 1216

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.net = Model(scales=4, base_width=32, dec_img=False)
        if self.opt.mgpus:
            self.net = torch.nn.DataParallel(self.net)
        self.net.to(self.device)
        print("=> loading checkpoint '{}'".format(self.opt.weight_path))
        checkpoint = torch.load(self.opt.weight_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        print("mae {} rmse {}".format(checkpoint['mae'], checkpoint['rmse']))

        if self.opt.dump:
            self.dump_dir = './completion_test_dump'
            os.makedirs(self.dump_dir, exist_ok=True)

    def run(self):
        self.net.eval()
        for loader in [self.test_loader]:
            for batch_idx, inputs in tqdm(enumerate(loader)):
                raw = inputs['depth_gt'].float().to(self.device)
                rgb = inputs['color'].float().to(self.device)
                rgb = rgb*255.0
                # crop
                assert raw.size()[2:] == rgb.size()[2:]
                h, w = raw.size()[2:]
                assert h >= self.crop_h
                assert w == self.crop_w  # 1216 don't need crop w
                h_cropped = h - self.crop_h
                raw = raw[:,:, h_cropped:h, 0:self.crop_w]
                rgb = rgb[:,:, h_cropped:h, 0:self.crop_w]

                mask = (raw > 0).float()
                output, _ = self.net(raw, mask, rgb)

                if use_norm_depth == False:
                    output = torch.clamp(output, min=self.opt.min_depth, max=self.opt.max_depth)
                else:
                    output = torch.clamp(output, min=0, max=1.0)
                    output = restore_depth(output, self.opt.min_depth, self.opt.max_depth)
                output = output[0][0:1].detach().cpu()

                if h_cropped != 0:
                    padding = (0, 0, h_cropped, 0)
                    output = torch.nn.functional.pad(output, padding, "constant", 0)
                    output[:, 0:h_cropped] = output[:, h_cropped].repeat(h_cropped, 1)

                if self.opt.dump:
                    to_pil = transforms.ToPILImage()
                    arr = output * 256.
                    arr = arr.detach().cpu()
                    pil_img = to_pil(arr.int())
                    pil_img.save(os.path.join(self.dump_dir, '{:010d}.png'.format(batch_idx)))
                else:
                    fig = plt.figure(num=batch_idx, figsize=(8, 10))
                    plt_img(fig, 3, 1, 1, plt, inputs['color'][0], 'color')
                    plt_img(fig, 3, 1, 2, plt, inputs['depth_gt'][0], 'depth')
                    plt_img(fig, 3, 1, 3, plt, output, 'depth')
                    plt.tight_layout()
                    plt.show()


if __name__ == "__main__":
    options = CompletionOptions()
    opt = options.parse()

    test = TestC(opt)
    test.run()
