import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm


__all__ = ["fill_na",
           "convert_to_tensor",
           "split_train_test_val",
           "init_weights",
           "l1_regularization",
           "clip_grad_norm",
           "setup_seed",
           "show_cuda_info",
           "get_predictions",
           ]



def fill_na(X, y):
    filled_data_list = []
    for df in X:
        # 使用前一个有效值进行填充
        df_filled = df.fillna(method='ffill')
        # 如果第一行存在NaN，使用后一个有效值进行填充
        # df_filled = df_filled.fillna(method='bfill')
        df_filled = df_filled.fillna(0.)
        filled_data_list.append(df_filled)

    df_filled = y.fillna(method='ffill')
    # df_filled = df_filled.fillna(method='bfill')
    df_filled = df_filled.fillna(0.)
    return filled_data_list, df_filled



def convert_to_tensor(data_list):
    # 将每个DataFrame转换为numpy数组，并存储在一个列表中
    numpy_arrays = [df.values for df in data_list]

    # 将所有的numpy数组堆叠成一个3D的numpy数组
    stacked_array = np.stack(numpy_arrays, axis=-1)  # 最后一个维度是因子维度

    # 将3D的numpy数组转换为Tensor
    tensor = torch.from_numpy(stacked_array)

    return tensor


def split_train_test_val(X, y, train_prop, test_prop, val_prop: float = None):
    """
    按比例分割数据集
    """
    assert len(X) == len(y)
    if val_prop is None:
        val_prop = 1. - train_prop - test_prop if (train_prop + test_prop < 1.) else 0.
    cumsum_prop = np.cumsum([train_prop, test_prop, val_prop])
    cumsum_sca = np.round(cumsum_prop * len(X)).astype(int)  # 修改这里
    train_X = X[: cumsum_sca[0]]
    train_y = y[: cumsum_sca[0]]
    test_X = X[cumsum_sca[0]: cumsum_sca[1]]
    test_y = y[cumsum_sca[0]: cumsum_sca[1]]
    val_X = X[cumsum_sca[1]:]
    val_y = y[cumsum_sca[1]:]
    return train_X, train_y, test_X, test_y, val_X, val_y


def init_weights(m):
    """
    模型参数初始化
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.orthogonal_(m._parameters[param])


def l1_regularization(model: nn.Module):
    """
    L1正则化
    """
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(abs(param))
    return regularization_loss


def clip_grad_norm(model: nn.Module, max_norm: float):
    """
    模型梯度剪裁

    Returns:
        None
    """
    # Calculate the gradient norm for all model parameters
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += torch.norm(p.grad.data, 2)**2
    total_norm = torch.sqrt(total_norm).item()

    # Clip the gradients if the norm exceeds the specified threshold
    if total_norm > max_norm:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(max_norm / total_norm)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def show_cuda_info():
    print("Cuda is available : ", torch.cuda.is_available())
    print("Cuda num : ", torch.cuda.device_count())  # gpu数量
    print("Current cuda index : ", torch.cuda.current_device())
    print("Current cuda name : ", torch.cuda.get_device_name(0))


def get_predictions(model, dataloader, device):
    model.eval()  # 设置模型为评估模式
    predictions = []

    with torch.no_grad():  # 关闭梯度计算
        for X, _ in tqdm(dataloader):  # 我们只关心输入数据，不需要标签
            X = X.to(device).squeeze()
            y_pred = model(X)
            # print(y_pred.transpose(1, 0).shape)
            # raise
            predictions.append(y_pred.transpose(1, 0).cpu().numpy())  # 转换为numpy数组并添加到列表中

    predictions = np.concatenate(predictions, axis=0)  # 将所有预测结果拼接到一起
    return predictions
