from tqdm import tqdm
import torch.nn.functional as F
import torch
import numpy as np
from utils.metrics import mask_classes


# code copied from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=RI1Y8bSImD7N
# test using a knn monitor
def knn_monitor(net, dataset, memory_data_loader, test_data_loader, device, cl_default, task_id, k=200, t=0.1,
                hide_progress=False):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    # classes = 100
    total_top1 = total_top1_mask = total_top5 = total_num = 0.0
    feature_bank = []
    with torch.no_grad():
        # generate feature bank
        for (data1, data2), target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=True):
            if cl_default:
                feature1 = net(data1.cuda(non_blocking=True), return_features=True)
                feature2 = net(data2.cuda(non_blocking=True), return_features=True)
                feature = torch.cat(feature1, feature2)
            else:
                feature1 = net(data1.cuda(non_blocking=True))
                feature2 = net(data2.cuda(non_blocking=True))
                feature = torch.cat((feature1, feature2), 1)

            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc='kNN', disable=True)
        for (data1, data2), target in test_bar:
            data1, data2, target = data1.cuda(non_blocking=True), data2.cuda(non_blocking=True), target.cuda(
                non_blocking=True)
            if cl_default:
                feature1 = net(data1, return_features=True)
                feature2 = net(data2, return_features=True)
                feature = torch.cat(feature1, feature2)
            else:
                feature1 = net(data1)
                feature2 = net(data2)
                feature = torch.cat((feature1, feature2), 1)  # [128,512->1024]
            feature = F.normalize(feature, dim=1)

            pred_scores = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data1.shape[0]
            _, preds = torch.max(pred_scores.data, 1)
            total_top1 += torch.sum(preds == target).item()

            pred_scores = mask_classes(pred_scores, dataset, task_id)
            _, preds = torch.max(pred_scores.data, 1)
            total_top1_mask += torch.sum(preds == target).item()

    return total_top1 / total_num * 100, total_top1_mask / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K] [256,200]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    return pred_scores
