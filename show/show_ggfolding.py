import numpy as np
import open3d as o3d
import torch
from datasets import Vegetation
from dataloader import get_dataloader
from config import Config
from utils import get_inputs, processbar
from models.network import GGFoldingNet

cfg = Config()
# model
param_save_path = "../params/ggfolding.pth"
net = GGFoldingNet(config=cfg)
net.to(cfg.device)
net.load_state_dict(torch.load(param_save_path))


def get_point_cloud(pts, color=None, estimate_normal=False):
    pc = o3d.PointCloud()
    pc.points = o3d.Vector3dVector(pts)
    if color is not None:
        pc.colors = o3d.Vector3dVector(np.array([color]*pts.shape[0]))
    if estimate_normal:
        o3d.estimate_normals(pc)
    return pc


def show_ground_generation(net, test_loader):
    num_iter = int(len(test_loader.dataset) // test_loader.batch_size)
    c_loader_iter = test_loader.__iter__()
    test_bce_loss, test_precision, test_recall, test_cd_loss = 0, 0, 0, 0
    net.eval()
    with torch.no_grad():
        for c_iter in range(num_iter):
            ##################################
            # load inputs to device.
            inputs = c_loader_iter.next()
            inputs = get_inputs(inputs, test_loader.dataset.config)
            ##################################
            ground_generation, score_pred = net(inputs)
            loss_bce, precision, recall, loss_cd = net.compute_loss(ground_generation, score_pred, inputs)
            test_bce_loss, test_precision, test_recall, test_cd_loss = test_bce_loss + loss_bce.item(), test_precision + precision, test_recall + recall, test_cd_loss + loss_cd.item()*10000
            print("\rtest process: %s   bce loss: %.5f   precision: %.5f   recall: %.5f   cd: %.5f" % (
                processbar(c_iter+1, num_iter), test_bce_loss/(c_iter+1), test_precision/(c_iter+1),
                test_recall/(c_iter+1), test_cd_loss/(c_iter+1)), end=""
            )
            # ##########  show  ################
            pts = inputs["points"][0]
            ground_gt = pts[inputs["ground_mask"][0].int() > 0.5]
            ground_gt_pc = get_point_cloud(ground_gt.detach().cpu().numpy(), [0, 0.651, 0.929], True)
            # ground_gt_pc = get_point_cloud(ground_gt.detach().cpu().numpy(), [0.7, 0.7, 0.7], True)
            generation_pc = get_point_cloud(ground_generation.detach().cpu().numpy(), [1, 0.706, 0], True)
            objects_pc = get_point_cloud(pts[inputs["ground_mask"][0].int() < 0.5].detach().cpu().numpy(), [0.7, 0.7, 0.7], True)
            # 展示 物体 + 标签
            o3d.draw_geometries([ground_gt_pc, objects_pc], window_name="generation", width=1000, height=800)
            # 展示标签
            o3d.draw_geometries([ground_gt_pc], window_name="generation", width=1000, height=800)
            # 展示生成
            o3d.draw_geometries([generation_pc], window_name="generation", width=1000, height=800)
            # 展示 物体 + 生成
            o3d.draw_geometries([objects_pc, generation_pc], window_name="generation", width=1000, height=800)

        test_bce_loss, test_cd_loss = test_bce_loss / num_iter, test_cd_loss / num_iter
        test_precision, test_recall = test_precision / num_iter, test_recall / num_iter
    print()
    return test_bce_loss, test_precision, test_recall, test_cd_loss


if __name__ == '__main__':
    ########################### train dataloader ################################
    vegetation_train_dataset = Vegetation(root="D:/Desktop/TestData", mode="train")
    vegetation_train_dataset.config = cfg
    train_loader, _ = get_dataloader(vegetation_train_dataset, 1, num_workers=0, shuffle=True, neighborhood_limits=cfg.neighborhood_limits)
    ########################### test dataloader #################################
    # dense slow
    vegetation_test_dense_slow = Vegetation(root="D:/Desktop/TestData", dirs=["dense + slow"], mode="test", data_augmentation=False)
    vegetation_test_dense_slow.config = cfg
    dense_slow_loader, _ = get_dataloader(vegetation_test_dense_slow, 1, num_workers=0, shuffle=False, neighborhood_limits=cfg.neighborhood_limits)
    # dense steep
    vegetation_test_dense_steep = Vegetation(root="D:/Desktop/TestData", dirs=["dense + steep"], mode="test", data_augmentation=False)
    vegetation_test_dense_steep.config = cfg
    dense_steep_loader, _ = get_dataloader(vegetation_test_dense_steep, 1, num_workers=0, shuffle=False, neighborhood_limits=cfg.neighborhood_limits)
    # sparse steep
    vegetation_test_sparse_steep = Vegetation(root="D:/Desktop/TestData", dirs=["sparse + steep"], mode="test", data_augmentation=False)
    vegetation_test_sparse_steep.config = cfg
    sparse_steep_loader, _ = get_dataloader(vegetation_test_sparse_steep, 1, num_workers=0, shuffle=False, neighborhood_limits=cfg.neighborhood_limits)
    # all data
    vegetation_test = Vegetation(root="D:/Desktop/TestData", mode="test", data_augmentation=False)
    vegetation_test.config = cfg
    test_loader, _ = get_dataloader(vegetation_test, 1, num_workers=0, shuffle=False, neighborhood_limits=cfg.neighborhood_limits)
    print(len(vegetation_test_dense_slow), len(vegetation_test_dense_steep), len(vegetation_test_sparse_steep), len(vegetation_train_dataset))

    print("data load finish !!!")
    # show_ground_generation(net, dense_slow_loader)  # precision: 0.89778   recall: 0.98500   cd: 13.26942
    # show_ground_generation(net, dense_steep_loader)  # precision: 0.92359   recall: 0.92037   cd: 20.05995
    show_ground_generation(net, sparse_steep_loader)  # precision: 0.82270   recall: 0.95791   cd: 16.89672
    # show_ground_generation(net, test_loader)  # precision: 0.90526   recall: 0.94333   cd: 17.71301