import numpy as np
import open3d as o3d
import os
import laspy
from torch.utils import data
from scipy.spatial.transform import Rotation


class Vegetation(data.Dataset):
    def __init__(self, root, dirs=("dense + slow", "dense + steep", "sparse + steep"), mode="train", data_augmentation=True):
        train_names, train_labels = [], []
        test_names, test_labels = [], []
        for dir_name in dirs:
            dir_name = root + "/" + dir_name
            print(dir_name)
            data_names, labels = [], []
            for _, _, filelist in os.walk(dir_name):
                for i in range(0, len(filelist), 2):
                    data_names.append(dir_name + "/" + filelist[i])
                    labels.append(dir_name + "/" + filelist[i+1])
                    t_name, l_name = filelist[i][:filelist[i].find(".")], filelist[i+1][:filelist[i+1].find(".")]
                    if t_name != l_name:
                        print("dataset loading error, please check !!")
            # print(data_names[:10])
            # print(labels[:10])
            datanum = len(data_names)
            test_num = datanum // 10
            train_names += data_names[:datanum-test_num]
            train_labels += labels[:datanum-test_num]
            test_names += data_names[datanum-test_num:]
            test_labels += labels[datanum-test_num:]

        self.mode = mode
        if self.mode == "train":
            self.datas, self.labels = train_names, train_labels
        elif self.mode == "test":
            self.datas, self.labels = test_names, test_labels
        self.config = None
        self.max_point_num = 57000
        self.data_augmentation = data_augmentation
        self.rot_factor = 1.

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        inp = laspy.read(self.datas[index]).xyz
        ground = laspy.read(self.labels[index]).xyz

        if inp.shape[0] > self.max_point_num:
            rand_inds = np.random.permutation(inp.shape[0])[:self.max_point_num]
            inp = inp[rand_inds]

        inp_pc = o3d.PointCloud()
        inp_pc.points = o3d.Vector3dVector(inp)
        inp_pc.colors = o3d.Vector3dVector(np.array([[0.7, 0.7, 0.7]]*inp.shape[0]))

        # ground_pc = o3d.PointCloud()
        # ground_pc.points = o3d.Vector3dVector(ground)
        # ground_pc.colors = o3d.Vector3dVector(np.array([[0, 0.651, 0.929]]*ground.shape[0]))
        # 寻找地面点，存地面点下标，把地面改成蓝色
        inp_tree = o3d.KDTreeFlann(inp_pc)
        ground_in_inp_inds = set()
        for point in ground:
            _, inds, _ = inp_tree.search_knn_vector_3d(point, 20)
            for j in range(20):
                ground_in_inp_inds.add(inds[j])
        ground_in_inp_inds = list(ground_in_inp_inds)
        ground_in_inp_inds = np.array(ground_in_inp_inds).astype(np.int)
        np.asarray(inp_pc.colors)[ground_in_inp_inds] = np.array([[0, 0.651, 0.929]]*ground_in_inp_inds.shape[0])

        # print(inp.shape[0], ground.shape[0])
        # o3d.estimate_normals(inp_pc)

        # 数据标准化
        inp_mean = np.mean(inp, axis=0, keepdims=True)
        inp = inp - inp_mean
        inp_norm = np.linalg.norm(inp, axis=1)
        inp = inp / np.max(inp_norm).item() * 1.7
        # print(np.max(inp_norm).item())
        # print(inp.max(axis=0), inp.min(axis=0), inp.max(axis=0)-inp.min(axis=0))
        inp_pc.points = o3d.Vector3dVector(inp)

        # # 画图专用
        # inp_pc = o3d.voxel_down_sample(inp_pc, voxel_size=0.025)
        # o3d.draw_geometries([inp_pc], window_name="level1", width=1000, height=800)
        # inp_pc = o3d.voxel_down_sample(inp_pc, voxel_size=0.05)
        # o3d.draw_geometries([inp_pc], window_name="level2", width=1000, height=800)
        # inp_pc = o3d.voxel_down_sample(inp_pc, voxel_size=0.1)
        # o3d.draw_geometries([inp_pc], window_name="level3", width=1000, height=800)
        # inp_pc = o3d.voxel_down_sample(inp_pc, voxel_size=0.2)
        # o3d.draw_geometries([inp_pc], window_name="level4", width=1000, height=800)
        # # inp_pc.points = o3d.Vector3dVector(np.asarray(inp_pc.points)[np.asarray(inp_pc.colors)[:, 2] == 0.929])
        # # inp_pc.colors = o3d.Vector3dVector(np.array([[1, 0.706, 0]]*np.asarray(inp_pc.points).shape[0]))
        # # o3d.draw_geometries([inp_pc], window_name="base points", width=1000, height=800)

        # 训练集数据增强
        if self.data_augmentation:
            euler_ab = np.random.rand(3) * np.pi * 2 / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            inp = np.matmul(rot_ab, inp.T).T

        inp_feats = np.ones_like(inp[:, :1]).astype(np.float32)

        ground_mask = np.zeros(shape=(inp.shape[0], ))
        ground_mask[ground_in_inp_inds] = 1
        return inp, inp_feats, ground_mask


if __name__ == '__main__':
    vegetation = Vegetation(root="D:/Desktop/TestData", mode="train")
    vegetation_test = Vegetation(root="D:/Desktop/TestData", mode="test")
    print(len(vegetation), len(vegetation_test))
    # vegetation = Vegetation(root="D:/Desktop/TestData", dirs=["dense + steep"])
    for i in range(len(vegetation)):
        _ = vegetation[i]