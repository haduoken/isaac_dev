import typing

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Voxelization:
    @classmethod
    def in_range(cls, pt):
        return -6 < pt[0] and pt[0] < 6 and -6 < pt[1] and pt[1] < 6 and pt[2] <= 2

    @classmethod
    def vox(cls, points_arr: typing.List[np.ndarray], device="cpu"):
        """
        :param points_arr:
        :param device:
        :return:
        """
        bs = len(points_arr)

        pillar_arr, non_empty_mask_arr = [], []

        for N in range(bs):
            # 进行体素化
            pillar_feat = np.zeros([1, 24, 24, 4, 100], dtype=np.float32)
            pt_cnt = np.zeros([24, 24], dtype=np.int8)
            non_empty_mask = np.zeros([1, 24, 24], dtype=bool)

            points = points_arr[N]
            for pt in points:
                tmp_x, tmp_y = 6 - pt[1], 6 - pt[0]
                ix = int(tmp_x / 0.5)
                iy = int(tmp_y / 0.5)
                cnt = int(pt_cnt[iy, ix])
                non_empty_mask[0, iy, ix] = True
                if cnt < 100:
                    pillar_feat[0, iy, ix, :, cnt] = pt
                    pt_cnt[iy, ix] += 1

            pillar_arr.append(pillar_feat)
            non_empty_mask_arr.append(non_empty_mask)

        # 将数据进行合并
        pillar_feat = np.concatenate(pillar_arr, axis=0)
        non_empty_mask = np.concatenate(non_empty_mask_arr, axis=0)

        # 转换成tensor
        pillar_feat = torch.from_numpy(pillar_feat).to(device)

        return pillar_feat, non_empty_mask


# 进入到pilldar encoder里面
class PillarEncoder(nn.Module):
    def __init__(
        self, voxel_size, point_cloud_range, out_channel, batch_size=4, device="cpu"
    ):
        super().__init__()
        self.batch_size = batch_size
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        # 得到pillar center 也就是 24*24*2的tensor，最后的两个维度代表了中心点的xy坐标
        x = torch.arange(
            self.vx / 2, point_cloud_range[3] - point_cloud_range[0], self.vx
        )
        y = torch.arange(
            self.vy / 2, point_cloud_range[4] - point_cloud_range[1], self.vy
        )

        # (2, 24, 24)  2代表x-center,y-center，第一个24代表列数, 第二个24代表行数
        self.pillar_center = torch.stack(torch.meshgrid(x, y, indexing="ij")).to(device)
        # (24, 24, 2) 2代表x-center,y-center, 第一个24代表行数, 第二个24代表列数
        self.pillar_center = self.pillar_center.permute(2, 1, 0).contiguous()
        self.pillar_center = self.pillar_center.unsqueeze(3)
        self.pillar_center = self.pillar_center.repeat(1, 1, 1, 100)

        self.bn0 = nn.BatchNorm1d(4)
        self.conv = nn.Conv1d(4, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def pos_enc(self, x, layer=6):
        rets = [x]
        for i in range(layer):
            for fn in [torch.sin, torch.cos]:
                rets.append(fn(2**i * x))
        return torch.concatenate(rets, -1)

    def forward(self, feat):
        """
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  (bs, out_channel, y_l, x_l)
        """
        feat[:, :, :, :2, :] = feat[:, :, :, :2, :] - self.pillar_center
        feat[:, :, :, :3, :] = (feat[:, :, :, :3, :] - 126.0) / 126.0

        feat = feat.reshape(-1, 4, 100)

        feat = self.bn0(feat)

        feat = F.relu(
            self.bn(self.conv(feat))
        )  # (p1 + p2 + ... + pb, out_channels, num_points)

        feat = torch.max(feat, dim=-1)[0]  # (p1 + p2 + ... + pb, out_channels)

        feat = feat.reshape(self.batch_size, 24, 24, 64)

        features = feat.permute(0, 3, 1, 2).contiguous()

        return features


class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[2, 2]):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)

        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(
                nn.Conv2d(
                    in_channel,
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    bias=False,
                    padding=1,
                )
            )
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]):
                blocks.append(
                    nn.Conv2d(
                        out_channels[i], out_channels[i], 3, bias=False, padding=1
                    )
                )
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        """
        x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        """
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs


class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(
                nn.ConvTranspose2d(
                    in_channels[i],
                    out_channels[i],
                    upsample_strides[i],
                    stride=upsample_strides[i],
                    bias=False,
                )
            )
            decoder_block.append(
                nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01)
            )
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        """
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        """
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i])  # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out


class MaskHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)
