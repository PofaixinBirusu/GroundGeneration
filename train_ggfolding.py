import torch
from datasets import Vegetation
from dataloader import get_dataloader
from config import Config
from utils import get_inputs, processbar
from models.network import GGFoldingNet
from evaluate import evaluate_generator

cfg = Config()
# train dataloader
vegetation_train_dataset = Vegetation(root="D:/Desktop/TestData", mode="train")
vegetation_train_dataset.config = cfg
train_loader, _ = get_dataloader(vegetation_train_dataset, 1, num_workers=0, shuffle=True, neighborhood_limits=cfg.neighborhood_limits)
# test dataloader
vegetation_test_dataset = Vegetation(root="D:/Desktop/TestData", mode="test", data_augmentation=False)
vegetation_test_dataset.config = cfg
test_loader, _ = get_dataloader(vegetation_test_dataset, 1, num_workers=0, shuffle=False, neighborhood_limits=cfg.neighborhood_limits)
# model
param_save_path = "params/ggfolding.pth"
net = GGFoldingNet(config=cfg)
net.to(cfg.device)
# net.load_state_dict(torch.load(param_save_path))
net.backbone.load_state_dict(torch.load("./params/ggnet-best-baseline.pth"))

lr = 0.0001
min_lr = 0.00001
lr_update_step = 20
optimizer = torch.optim.AdamW(params=net.parameters(), lr=0.0001, weight_decay=0.0001)


def update_lr(optimizer, gamma=0.5):
    global lr
    lr = max(lr*gamma, min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("lr update finished  cur lr: %.5f" % lr)


def train():
    min_cd_loss = 1e8
    for epoch_count in range(1, cfg.max_epoch+1):
        net.train()
        train_bce_loss, train_cd_loss = 0, 0
        num_iter = int(len(train_loader.dataset) // train_loader.batch_size)
        c_loader_iter = train_loader.__iter__()
        for c_iter in range(num_iter):
            ##################################
            # load inputs to device.
            inputs = c_loader_iter.next()
            inputs = get_inputs(inputs, cfg)
            ##################################
            ground_generation, score_pred = net(inputs)
            loss_bce, precision, recall, loss_cd = net.compute_loss(ground_generation, score_pred, inputs)

            train_bce_loss, train_cd_loss = train_bce_loss + loss_bce.item(), train_cd_loss + loss_cd.item()
            # backward pass
            loss = loss_bce + 100*loss_cd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("\rprocess: %s   bce loss: %.5f   precision: %.5f   recall: %.5f   cd loss: %.5f" % (
                processbar(c_iter + 1, num_iter), loss_bce.item(), precision, recall, loss_cd.item()*10000), end=""
            )
        train_bce_loss, train_cd_loss = train_bce_loss/len(train_loader.dataset), train_cd_loss/len(train_loader.dataset)
        print("\nepoch: %d  train bce loss: %.5f   cd loss: %.5f" % (epoch_count, train_bce_loss, train_cd_loss))
        test_bce_loss, test_precision, test_recall, test_cd_loss = evaluate_generator(net, test_loader)
        if min_cd_loss > test_cd_loss:
            min_cd_loss = test_cd_loss
            print("Save ...")
            torch.save(net.state_dict(), param_save_path)
            print("Save finish !!!")
        if epoch_count % lr_update_step == 0:
            update_lr(optimizer, 0.5)


if __name__ == '__main__':
    train()