import torch
from torch import nn


def processbar(current, totle):
    process_str = ""
    for i in range(int(20*current/totle)):
        process_str += "█"
    while len(process_str) < 20:
        process_str += " "
    return "%s|  %d / %d" % (process_str, current, totle)


def get_inputs(inputs, config):
    for k, v in inputs.items():
        if k == "sample":
            continue
        if type(v) == list:
            inputs[k] = [item.to(config.device) for item in v]
        else:
            inputs[k] = v.to(config.device)
    return inputs


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1)[:, :, None]
    dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
    return dist


# 补全
class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, f, f_):
        if f.shape[1] == 0:
            return 0
        # f : patch_num x point_num x 4
        # f_: patch_num x M x 4
        # dis: patch_num x point_num x M
        try:
            dis = square_distance(f, f_)
            # dis = torch.sqrt(dis)
            # f2f_: patch_num x point_num   f_2f: patch_num x M
            f2f_, f_2f = dis.min(dim=2)[0], dis.min(dim=1)[0]
            # d = torch.stack([f2f_.mean(dim=1), f_2f.mean(dim=1)], dim=0).max(dim=0)[0]
            d = f2f_.mean(dim=1) + f_2f.mean(dim=1)
        except:
            print(f.shape, f_.shape)
        return d.mean()