import torch
from audtorch.metrics.functional import pearsonr


def ic_loss(y: torch.Tensor, y_pred: torch.Tensor):
    y = y.squeeze()  # 删除dim=1维度，从[n, 1] => [n]
    y_pred = y_pred.squeeze()  # 删除dim=1维度，从[n, 1] => [n]
    ic = torch.mean(pearsonr(y, y_pred))
    return 1 - ic


def weight_ic_loss(y: torch.Tensor, y_pred: torch.Tensor):
    y = y.squeeze()  # 删除dim=1维度，从[n, 1] => [n]
    y_pred = y_pred.squeeze()  # 删除dim=1维度，从[n, 1] => [n]
    n = y.shape[-1]  # 获得股票个数

    # 使用torch的argsort函数
    sort_num = torch.argsort(torch.argsort(-y)) + 1
    # 使用torch的数学函数
    weight = (0.9 ** ((sort_num.float() - 1) / (n - 1)))

    w_y = y * weight
    w_x = y_pred * weight
    w_corr = torch.mean(pearsonr(w_y, w_x))
    # 将损失函数转换为一个越小越好的损失值, 并映射到[0, 1]区间内
    return (1 - w_corr) / 2

