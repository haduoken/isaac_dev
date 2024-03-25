import typing

import torch
from torch import nn
import os
import numpy as np
from parse_seg_map import *

from torch.utils.data import DataLoader, Dataset, random_split


class CustomDataSet(Dataset):
    def __init__(self, dir_name, specify_idx: list = None):
        super().__init__()
        self.dir_name = dir_name
        if specify_idx is not None:
            self.idx_arr = specify_idx
        else:
            self.idx_arr = []
            # 遍历
            for f in os.listdir(os.path.join(dir_name, 'RenderProduct_cam_l', "camera_params")):
                f_name = os.path.splitext(f)[0]
                id = f_name.split('_')[2]
                self.idx_arr.append(id)

    def __len__(self):
        return len(self.idx_arr)

    def __getitem__(self, idx):
        return self.dir_name, self.idx_arr[idx]


class SpliteDataset(Dataset):
    def __init__(self, file="train.txt"):
        super().__init__()
        with open(os.path.join(file), "r") as f:
            self.data = f.read().split("\n")[:-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx].split(",")
        return data[0], data[1]


def get_loader(file: str, batch_size=4):
    set = SpliteDataset(file)
    loader = DataLoader(set, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader


def get_custom_loader(folder, specify_idx: list = None, batch_size=4):
    set = CustomDataSet(folder, specify_idx)
    loader = DataLoader(set, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader


def read_pcd_bin(pcd_file):
    with open(pcd_file, "rb") as f:
        data = f.read()
        data_binary = data[data.find(b"DATA binary") + 12:]
        points = np.frombuffer(data_binary, dtype=np.float32).reshape(-1, 4)
        points = points.astype(np.float32)
    return points


def read_label(label_file):
    with open(label_file, "r") as f:
        data = f.read(24 * 24)
        data = [int(x) for x in data]
        data = np.array(data).reshape((24, 24))
        data = data[None, :, :]
        return data


def load_next_data(dir: str, idx) -> typing.Tuple[np.ndarray, np.ndarray]:
    # 加载点云
    pcd_file = os.path.join(dir, "cloud", f"{idx}.pcd")
    label_file = os.path.join(dir, "label", f"{idx}.txt")

    x = read_pcd_bin(pcd_file)
    label = read_label(label_file)
    return x, label


def load_batch(dir_name: list, idx: list) -> typing.Tuple[list, np.ndarray]:
    # 加载点云
    frame_arr = []
    for i in range(len(dir_name)):
        frame = Frame(folder=dir_name[i], frame_id=idx[i])
        frame_arr.append(frame)

    return frame_arr


class DataLoaderTools:
    @classmethod
    def gen_train_val_test_splite(cls):
        dataset1 = CustomDataSet("/home/kilox/workspace/isaac_dev/_out_sdrec39")
        output_dir = '/data'

        train_set, val_set, test_set = random_split(
            dataset1, [0.5, 0.2, 0.3]
        )

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        with open(f"{output_dir}/train.txt", "w") as f:
            for data in train_set:
                f.write(f"{data[0]},{data[1]}\n")

        with open(f"{output_dir}/val.txt", "w") as f:
            for data in val_set:
                f.write(f"{data[0]},{data[1]}\n")

        with open(f"{output_dir}/test.txt", "w") as f:
            for data in test_set:
                f.write(f"{data[0]},{data[1]}\n")

    @classmethod
    def gen_full(cls):
        dataset1 = CustomDataSet("/home/kilox/data/custom_cloud/CC_2023_10_25")
        dataset2 = CustomDataSet("/home/kilox/data/custom_cloud/CC_2023_10_26")
        dataset = dataset1 + dataset2
        with open(
                os.path.join("/home/kilox/data/custom_cloud/train", "full.txt"), "w"
        ) as f:
            for data in dataset:
                f.write(f"{data[0]},{data[1]}\n")


if __name__ == "__main__":
    DataLoaderTools.gen_train_val_test_splite()
