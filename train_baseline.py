import laspy
import numpy as np
import open3d as o3d
import torch
from datasets import Vegetation
from dataloader import get_dataloader
from config import Config
from utils import get_inputs, processbar
from models.network import GGNet
from evaluate import evaluate_baseline

cfg = Config()
# train dataloader
vegetation_train_dataset = Vegetation(root="D:/Desktop/TestData", mode="train")
vegetation_train_dataset.config = cfg
train_loader, _ = get_dataloader(vegetation_train_dataset, 1, num_workers=0, shuffle=True, neighborhood_limits=cfg.neighborhood_limits)
# test dataloader
vegetation_test_dataset = Vegetation(root="D:/Desktop/TestData", mode="test", data_augmentation=False, dirs=["sparse + steep"])
vegetation_test_dataset.config = cfg
test_loader, _ = get_dataloader(vegetation_test_dataset, 1, num_workers=0, shuffle=False, neighborhood_limits=cfg.neighborhood_limits)
# model
param_save_path = "params/ggnet-best-baseline.pth"
net = GGNet(config=cfg)
net.to(cfg.device)
net.load_state_dict(torch.load(param_save_path))

optimizer = torch.optim.AdamW(params=net.parameters(), lr=0.0001, weight_decay=0.0001)


def train():
    max_test_precision = 0
    for epoch_count in range(cfg.max_epoch):
        net.train()
        train_loss = 0
        num_iter = int(len(train_loader.dataset) // train_loader.batch_size)
        c_loader_iter = train_loader.__iter__()
        for c_iter in range(num_iter):
            ##################################
            # load inputs to device.
            inputs = c_loader_iter.next()
            inputs = get_inputs(inputs, cfg)
            ##################################
            # print(inputs["points"][0].shape[0])
            super_points_score_pred, _ = net(inputs)
            loss_bce, precision, recall = net.compute_loss(super_points_score_pred, inputs)

            train_loss += loss_bce.item()
            # backward pass
            optimizer.zero_grad()
            loss_bce.backward()
            optimizer.step()

            print("\rprocess: %s   bce loss: %.5f   precision: %.5f   recall: %.5f" % (
                processbar(c_iter + 1, num_iter), loss_bce.item(), precision, recall), end=""
            )
        train_loss /= len(train_loader.dataset)
        print("\nepoch: %d  train loss: %.5f" % (epoch_count + 1, train_loss))
        test_loss, test_precision, test_recall = evaluate_baseline(net, test_loader)
        if max_test_precision < test_precision:
            max_test_precision = test_precision
            print("Save ...")
            torch.save(net.state_dict(), param_save_path)
            print("Save finish !!!")


if __name__ == '__main__':
    # train()
    evaluate_baseline(net, test_loader)        #  precision: 0.90962   recall: 0.93989  F1-score: 0.92451