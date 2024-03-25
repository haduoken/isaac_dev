import random
import time
import os, sys

import torch

from core import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 4

model = PerceptionNet(batch_size=batch_size, device=device)
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

loss = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 1, 0.5])).to(device)

converge = False

id = 1
while True:
    id += 1
    saved_logs_path = os.path.join(os.curdir, "work_dir", f"summary_{id}")
    if not os.path.exists(saved_logs_path):
        os.makedirs(saved_logs_path, exist_ok=True)
        break

writer = SummaryWriter(saved_logs_path)

train_loader = get_loader(
    "data/train.txt", batch_size=batch_size
)
# val_loader = get_loader(
#     "/home/kilox/data/custom_cloud/train/val.txt", batch_size=batch_size
# )

for epoch in range(36):
    cur_lr = optimizer.param_groups[0]["lr"]
    # 训练阶段
    model.train()
    total_loss = 0
    start = time.time()
    iter_cnt = 0

    for dir_name, train_idx in tqdm(train_loader):
        optimizer.zero_grad()

        frame_arr = load_batch(dir_name, train_idx)
        pillar_feat, non_empty_mask = Voxelization.vox(x, device)

        predict = model(pillar_feat)

        # 只统计有点云的地方
        label[~non_empty_mask] = 2
        label = torch.from_numpy(label)
        label = label.to(device)

        ce_loss = loss(predict, label)

        ce_loss += dice_loss(
            F.softmax(predict, dim=1),
            F.one_hot(label, 3).permute(0, 3, 1, 2),
            multiclass=True,
        )

        ce_loss.backward()

        optimizer.step()

        total_loss += ce_loss.item()
        iter_cnt += 1

    end = time.time()

    print(
        f"epoch {epoch} loss {total_loss / iter_cnt} learning rate {cur_lr} use time {end - start}"
    )
    writer.add_scalar("total loss", total_loss / iter_cnt, epoch)
    writer.add_scalar("learn rate ", cur_lr, epoch)

    # 每3 epoch 保存一次ckpts
    if epoch % 3 == 0:
        torch.save(model.state_dict(), f"checkpoints/model_{epoch}.pth")

    # 每隔3个epoch进行一次评估
    # if epoch % 3 == 0:
    #     # 评估阶段
    #     eval_loss = 0
    #     model.eval()
    #     with torch.no_grad():
    #         iter_cnt = 0
    #         for dir_name, val_idx in tqdm(val_loader):
    #             x, label = load_batch(dir_name, val_idx)
    #             pillar_feat, non_empty_mask = Voxelization.vox(x, device)
    #
    #             predict = model(pillar_feat)
    #
    #             # 类别 0 代表地面 1 代表障碍物 2 代表没有物体
    #             label[~non_empty_mask] = 2
    #             label = torch.from_numpy(label)
    #             label = label.to(device)
    #
    #             ce_loss = loss(predict, label)
    #             ce_loss += dice_loss(
    #                 F.softmax(predict, dim=1).float(),
    #                 F.one_hot(label, 3).permute(0, 3, 1, 2).float(),
    #                 multiclass=True,
    #             )
    #
    #             eval_loss += ce_loss.item()
    #             iter_cnt += 1
    #
    #         print(f"epoch {epoch} evaluate loss {eval_loss/iter_cnt}")
    #         writer.add_scalar("eval loss", eval_loss / iter_cnt, epoch)
