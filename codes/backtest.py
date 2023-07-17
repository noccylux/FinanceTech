import warnings

import matplotlib.pyplot as plt
import numba as nb
import numpy as np

warnings.filterwarnings("ignore")


@nb.njit()
def _decay_linear(x, d):
    res = np.empty_like(x)
    w = np.arange(1, d + 1)
    w = w / np.sum(w)
    for i in range(d - 1, x.shape[0]):
        for j in range(x.shape[1]):
            res[i, j] = np.sum(x[i - d + 1:i + 1, j] * w)
    return res[d - 1:]


def backtest(y: np.ndarray,
             y_pred: np.ndarray,
             truncation: float = 1.,
             decay: int = 0,
             book_size: float = 20000000,
             commission_rate: float = 0.002  # 假设手续费率为0.2%
             ):
    """
    :param y: 实际收益率。
    :param y_pred: 因子预测收益率。
    :param truncation: 股票的最大仓位限制，使用因子值分配的仓位若大于这个仓位则截取到该仓位。
    :param decay: 将因子值进行线性衰减加权平均，可以降低换手率。
    :param book_size: 每日交易的本金金额。
    :param commission_rate: 手续费率
    :return:
    """
    # 注意每个函数都要兼容nan
    assert len(y) == len(y_pred)
    assert (truncation >= 0.) and (truncation <= 1.)
    assert decay >= 0

    y_pred = _decay_linear(y_pred, d=decay) if decay > 0 else y_pred  # 计算decay
    y = y[len(y) - len(y_pred):]
    y_pred = (y_pred - np.nanmean(y_pred, axis=1).reshape([-1, 1]))

    max_ = np.nansum(np.abs(y_pred), axis=1) * truncation  # 计算最大仓位
    max_ = np.repeat(max_[:, np.newaxis], y_pred.shape[-1], axis=1)
    mask = abs(y_pred) > max_  # 计算仓位大于最大仓位的股票
    y_pred[mask] = np.sign(y_pred[mask]) * max_[mask]  # 将仓位大于最大仓位的股票仓位乘上max_比例，减低仓位
    y_pred = (y_pred - np.nanmean(y_pred, axis=1).reshape([-1, 1]))  # 市场中性化
    y_pred = y_pred / (np.nansum(abs(y_pred), axis=1).reshape([-1, 1]))  # 仓位归一化

    position = y_pred * book_size  # 每只股票的每日仓位

    long_count = np.sum(position > 0, axis=1)  # 每日做多股票数量
    short_count = np.sum(position < 0, axis=1)  # 每日做空股票数量

    tnl_daily = np.nansum(y * position, axis=1)  # 每日收益
    trade_vol_daily = position[1:] - position[:-1]  # 计算每日交易量
    tnl_daily = tnl_daily[1:]  # 删去第0行(因为计算每日交易量时第0行无法计算)
    trade_cost_daily = np.nansum(np.abs(trade_vol_daily * commission_rate), axis=1)  # 计算每日交易成本
    tnl_daily -= trade_cost_daily  # 从每日收益中扣除交易成本

    returns = tnl_daily / book_size  # 每日收益率

    ir = np.mean(tnl_daily) / (np.std(tnl_daily) + 1e-6)
    sharpe = np.sqrt(252) * ir  # 计算年化sharp

    daily_turnover = abs(trade_vol_daily) / book_size  # 每日换手率
    draw_down = np.max(tnl_daily) - np.min(tnl_daily)  # 最大回撤

    results = {
        "long_count": long_count,
        "short_count": short_count,
        "tnl_daily": tnl_daily,
        "raw_tnl_daily": tnl_daily + trade_cost_daily,
        "returns": float(252 * np.mean(returns)),
        "returns_daily": returns,
        "ir": ir,
        "sharpe": sharpe,
        "trade_vol_daily": trade_vol_daily,
        "daily_turnover": daily_turnover,
        "trade_cost_daily": trade_cost_daily,
        "turnover": float(np.nanmean(np.nansum(daily_turnover, axis=1))),
        "draw_down": draw_down,
    }
    return results


def show_back_test(y: np.ndarray,
                   y_pred: np.ndarray,
                   truncation: float = 1.,
                   decay: int = 0,
                   book_size: float = 20000000,
                   commission_rate: float = 0.002):
    if len(y) != len(y_pred):
        y = y[len(y) - len(y_pred):]
    res = backtest(y=y,
                   y_pred=y_pred,
                   truncation=truncation,
                   decay=decay,
                   book_size=book_size,
                   commission_rate=commission_rate)

    # 绘制折线图
    plt.plot(np.cumsum(res["raw_tnl_daily"]), color="#797A79", alpha=1., label='Cumulative Returns')
    plt.plot(np.cumsum(res["trade_cost_daily"]), color="#F5CB1F", alpha=0.6, label='Cumulative Fees')
    plt.plot(np.cumsum(res["tnl_daily"]), color="#E94E42", alpha=0.8, label='Cumulative Returns (excluding fees)')

    # 添加图例和标签
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Cumulative Value')

    # 显示图形
    plt.show()

    for key in ["returns", "ir", "sharpe", "turnover", "draw_down"]:
        print("{:<10} : {:<15.4f}".format(key, res[key]))
