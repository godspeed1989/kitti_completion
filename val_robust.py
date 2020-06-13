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
from metric import AverageMeter, Metrics


__R0 = np.array([[ 0.9999239,   0.00983776, -0.00744505],
                 [-0.0098698,   0.9999421,  -0.00427846],
                 [ 0.00740253,  0.00435161,  0.9999631 ]], dtype=np.float32)

__velo2cam = np.array([[ 7.023745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                       [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
                       [ 9.998621e-01,  7.523790e-03,  2.250755e-03, -2.717806e-01]], dtype=np.float32)

__P_rect = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
                     [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                     [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]], dtype=np.float32)


class Calib(object):
    @staticmethod
    def inverse_rigid_trans(Tr):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    def __init__(self, R0, V2C, P):
        self.R0 = R0    # R0_rect
        self.V2C = V2C  # Tr_velo_to_cam
        self.P = P      # P2 or P3

        self.C2V = self.inverse_rigid_trans(self.V2C)
        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))
    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))
    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))
    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)

def project_depth_to_points(depth):
    calib = Calib(__R0, __velo2cam, __P_rect)

    rows, cols = depth.shape
    mask = depth > 0

    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    valid_points = points[mask.reshape(-1)]
    cloud = calib.project_image_to_velo(valid_points)

    # the image coordinate and depth of each point
    # coordinate:
    coords = np.stack([c, r])
    coords = coords.reshape((2, -1))
    coords = coords.T
    loc = coords[mask.reshape(-1)]
    # swapc from col,row to row,col
    loc = np.roll(loc, 1, axis=1)
    # depth:
    dep = depth.reshape((-1, 1))
    dep = dep[mask.reshape(-1)]
    ret_loc = np.concatenate([loc, dep], 1)

    return cloud, ret_loc

import pyrender
import trimesh

def plot3d(pc):
    colors = np.zeros(pc.shape)
    colors[:,1] = 0.85

    cloud = pyrender.Mesh.from_points(pc, colors=colors)

    scene = pyrender.Scene()
    scene.add(cloud)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2.5)
    viewer.close_external()

def restore_scan_line(pc, coord, n_lines=64, pitch_start=-15.7, pitch_step=0.4, verbose=False):
    '''pc [N, 3]
    return: mask [N,]
    '''
    assert pc.ndim == 2 and pc.shape[1] >= 3
    x = pc[:,0]
    y = pc[:,1]
    z = pc[:,2]
    p = np.arcsin(z / np.sqrt(x**2 + y**2 + z**2))
    pitch = np.rad2deg(p)
    if verbose:
        print('\npitch actual', np.min(pitch), '->', np.max(pitch))
        print('\npitch expect', pitch_start, '->', pitch_start+pitch_step*n_lines)

    mask = np.full((pitch.shape[0],), -1)
    for i in range(n_lines):
        begin = pitch_start + i * pitch_step
        end = begin + pitch_step
        mask[np.logical_and(pitch > begin,  pitch < end)] = i

    reorg_pc = []
    reorg_coord = []
    for i in range(n_lines):
        reorg_pc.append(pc[np.where(mask == i)])
        reorg_coord.append(coord[np.where(mask == i)])
        if verbose:
            print('[{}]{};'.format(i, reorg_pc[-1].shape[0]), end='')
    if verbose:
        print('')

    return reorg_pc, reorg_coord

def reduce_scan_line(reorg_pc, reorg_coord, step=2):
    lines = len(reorg_pc)
    all_points = None
    all_coords = None
    for i in range(0, lines, step):
        pc_in = reorg_pc[i].copy()
        coord_in = reorg_coord[i].copy()
        if pc_in.shape[0] == 0:
            continue
        points = pc_in[:,:3]
        coords = coord_in[:,:3]
        if all_points is None:
            all_points = points
            all_coords = coords
        else:
            all_points = np.concatenate([all_points, points], 0)
            all_coords = np.concatenate([all_coords, coords], 0)
    return all_points, all_coords

def sample_scan_line(reorg_pc, reorg_coord, ratio=0.5):
    lines = len(reorg_pc)
    all_points = None
    all_coords = None
    for i in range(0, lines):
        pc_in = reorg_pc[i].copy()
        coord_in = reorg_coord[i].copy()
        if pc_in.shape[0] == 0:
            continue
        points = pc_in[:,:3]
        coords = coord_in[:,:3]
        cnt = points.shape[0]
        sampled = np.random.choice(cnt, (int)(cnt * ratio), replace=False)
        if all_points is None:
            all_points = points[sampled]
            all_coords = coords[sampled]
        else:
            all_points = np.concatenate([all_points, points[sampled]], 0)
            all_coords = np.concatenate([all_coords, coords[sampled]], 0)
    return all_points, all_coords

def restore_depth_map(pc, pc_coord, img_shape):
    depth_map = np.zeros(img_shape, dtype=np.float)
    coords = pc_coord[:,:2].astype(np.int)
    depths = pc_coord[:,2]
    depth_map[coords[:,0], coords[:,1]] = depths
    return depth_map


class TestC:
    def __init__(self, opt):
        super(TestC, self).__init__()
        self.opt = opt

        self.opt.batch_size = 1
        self.opt.crop_h = 352
        self.opt.crop_w = 1216
        test_dataset = DAT_VAL_TEST(
            self.opt.data_path, is_test=False, crop_h=self.opt.crop_h, crop_w=self.opt.crop_w,
            min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
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
                depth_gt = inputs['depth_gt'].float()
                depth_sd_gt = inputs['depth_sd_gt'].float()

                # reduce scan lines
                depth_map = depth_gt[0][0].numpy()
                pc, coord = project_depth_to_points(depth_map)
                reorg_pc, reorg_coord = restore_scan_line(pc, coord, verbose=not self.opt.dump)
                reduced_pc, reduced_coord = reduce_scan_line(reorg_pc, reorg_coord, step=4)
                # reduced_pc, reduced_coord = sample_scan_line(reorg_pc, reorg_coord, ratio=0.1)

                print('pc:', pc.shape[0], '->', reduced_pc.shape[0])

                reduced_depth = restore_depth_map(reduced_pc, reduced_coord, [self.opt.crop_h, self.opt.crop_w])
                reduced_depth = torch.from_numpy(reduced_depth).float()
                reduced_depth = reduced_depth.unsqueeze(0).unsqueeze(0)

                raw = reduced_depth.to(self.device)
                rgb = inputs['color'].float().to(self.device)
                rgb = rgb*255.0
                # crop
                assert raw.size()[2:] == rgb.size()[2:]
                h, w = raw.size()[2:]
                assert h >= self.crop_h
                assert w == self.crop_w  # 1216 don't need crop w
                h_cropped = h - self.crop_h
                depth_gt = depth_gt[:,:, h_cropped:h, 0:self.crop_w]
                depth_sd_gt = depth_sd_gt[:,:, h_cropped:h, 0:self.crop_w]
                raw = raw[:,:, h_cropped:h, 0:self.crop_w]
                rgb = rgb[:,:, h_cropped:h, 0:self.crop_w]

                mask = (raw > 0).float()
                output, _ = self.net(raw, mask, rgb)

                if use_norm_depth == False:
                    output = torch.clamp(output, min=self.opt.min_depth, max=self.opt.max_depth)
                else:
                    output = torch.clamp(output, min=0, max=1.0)
                    output = restore_depth(output, self.opt.min_depth, self.opt.max_depth)
                output = output[:,0:1].detach().cpu()

                metric = Metrics(max_depth=self.opt.max_depth)
                mae = AverageMeter()
                rmse = AverageMeter()
                metric.calculate(output, depth_sd_gt)
                mae.update(metric.get_metric('mae'), metric.num)
                rmse.update(metric.get_metric('rmse'), metric.num)
                print("model: mae {} rmse {}".
                      format(int(1000*mae.avg), int(1000*rmse.avg)))

                if not self.opt.dump:
                    plot3d(reduced_pc)
                    fig = plt.figure(num=batch_idx, figsize=(8, 10))
                    plt_img(fig, 4, 1, 1, plt, inputs['color'][0], 'color')
                    plt_img(fig, 4, 1, 2, plt, depth_gt[0], 'depth')
                    plt_img(fig, 4, 1, 3, plt, raw.cpu()[0], 'depth')
                    plt_img(fig, 4, 1, 4, plt, output[0], 'depth')
                    plt.tight_layout()
                    plt.show()


if __name__ == "__main__":
    options = CompletionOptions()
    opt = options.parse()

    test = TestC(opt)
    test.run()
