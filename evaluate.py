import torch
from config import Config
from utils import get_inputs, processbar


def evaluate_baseline(net, test_loader):
    num_iter = int(len(test_loader.dataset) // test_loader.batch_size)
    c_loader_iter = test_loader.__iter__()
    test_loss, test_precision, test_recall = 0, 0, 0
    net.eval()
    with torch.no_grad():
        for c_iter in range(num_iter):
            ##################################
            # load inputs to device.
            inputs = c_loader_iter.next()
            inputs = get_inputs(inputs, test_loader.dataset.config)
            ##################################
            super_points_score_pred, _ = net(inputs)
            loss_bce, precision, recall = net.compute_loss(super_points_score_pred, inputs)
            test_loss, test_precision, test_recall = test_loss + loss_bce.item(), test_precision + precision, test_recall + recall
            print("\rtest process: %s   bce loss: %.5f   precision: %.5f   recall: %.5f" % (
                processbar(c_iter+1, num_iter), test_loss/(c_iter+1), test_precision/(c_iter+1), test_recall/(c_iter+1)),
                end=""
            )
        test_loss = test_loss / num_iter
        test_precision, test_recall = test_precision / num_iter, test_recall / num_iter
    print()
    return test_loss, test_precision, test_recall


def evaluate_generator(net, test_loader):
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
        test_bce_loss, test_cd_loss = test_bce_loss / num_iter, test_cd_loss / num_iter
        test_precision, test_recall = test_precision / num_iter, test_recall / num_iter
    print()
    return test_bce_loss, test_precision, test_recall, test_cd_loss
