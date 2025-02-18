import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def calculate_metrics_by_period(df_data: pd.DataFrame, arr_positions: np.ndarray, arr_returns: np.ndarray, arr_nav: np.ndarray) -> dict:
    """计算给定时期的评价指标

    Args:
        df_data (pd.DataFrame): 原始数据
        arr_positions (np.ndarray): 持仓信息
        arr_returns (np.ndarray): 收益率
        arr_nav (np.ndarray): 净值
    Returns:
        dict: 评价指标计算结果
    """
    metrics = {}
    rf = 0.03
    # param_metirc = weights2calculate([0.5, 0.2, 0.3])

    # 基础指标计算
    annual_return = (arr_nav.iloc[-1] ** (252/len(arr_nav))) - 1
    annual_vol = arr_returns.std() * np.sqrt(252)
    sharpe = (annual_return - rf) / annual_vol if annual_vol != 0 else 0
    
    # 计算最大回撤及其起始日期
    drawdown = arr_nav.cummax() - arr_nav
    max_drawdown = drawdown.max() / arr_nav.cummax().max()
    max_drawdown_end_idx = drawdown.argmax()
    max_drawdown_start_idx = arr_nav[:max_drawdown_end_idx + 1].argmax()
    max_drawdown_start_date = df_data["tdate"].iloc[max_drawdown_start_idx]
    max_drawdown_end_date = df_data["tdate"].iloc[max_drawdown_end_idx]

    # 计算胜率和盈亏比
    daily_pnl = arr_returns * arr_positions[:-1]
    win_rate = (daily_pnl > 0).mean()
    # 衡量因子的择时有效性，或者因子的稳定性，当前bar对应的，过去100个bar，他的gain_loss_ratio，
    gain_loss_ratio = abs(daily_pnl[daily_pnl > 0].mean() / daily_pnl[daily_pnl < 0].mean()) \
        if len(daily_pnl[daily_pnl < 0]) > 0 else np.inf

    metrics.update({
       '总收益': arr_nav.iloc[-1] - 1,
       '年化收益': annual_return,
       '年化波动率': annual_vol,
       '夏普比率': sharpe,
       '最大回撤率': max_drawdown,
       '最大回撤起始日': max_drawdown_start_date,
       '最大回撤结束日': max_drawdown_end_date,
       '总交易次数': len(arr_returns),
       '胜率': win_rate,
       '盈亏比': gain_loss_ratio
    })
    return metrics


def calculate_returns_and_nav(df_data: pd.DataFrame, arr_positions: np.ndarray) -> tuple:
    """计算收益率和净值

    Args:
        df_data (pd.DataFrame): 数据
        arr_positions (np.ndarray): 持仓信号
    Returns:
        tuple: (list, numpy.ndarray) 每次预测的收益率(去掉了最后一个bar)，累积净值
    """
    price_change = df_data["close"].pct_change().fillna(0)
    strategy_returns = arr_positions[:-1] * price_change[1:] # 每一个position都是预测值，要与下一个bar的price_change相乘

    # position_diff = np.abs(positions[1:] - positions[:-1])
    # fees = 0.0002 # 手续费为万二
    # pnl = strategy_returns
    # strategy_returns = pnl - fees * position_diff # 收益率 = 收益率 - 手续费 - 滑点
    # cumu_const = np.cumsum(fees * position_diff) # 累计手续费

    cumulative_returns = (1 + strategy_returns).cumprod() # 累积收益率 复利
    # cumulative_returns = 1 + strategy_returns.cumsum() # 累积收益率 单利
    return strategy_returns, cumulative_returns

def calculate_positions(arr_prediction: np.ndarray, float_scale: float = 0.0005, float_position_size: float = 1) -> np.ndarray:
    """计算持仓信号 
       注：持仓信号可以用sigmoid  tanh, relu, zscore, 等函数先进行计算处理
    Args:
        arr_prediction (np.ndarray): 预测值
        float_scale (float, optional): 缩放因子. Defaults to 0.0005.
        float_position_size (int, optional): 持仓规模. Defaults to 1.
    Returns:
        np.ndarray: 持仓信号
    """
    position = (arr_prediction / float_scale) * float_position_size
    return np.clip(position, -1, 1)

def backtest(df_data: pd.DataFrame, str_index_code: str, int_frequency: int, int_n: int) -> pd.DataFrame:
    """回测核心函数

    Args:
        df_data (pd.DataFrame): 原始数据 对齐了因子
        str_index_code (str): 指数代码
        int_frequency (int): 频率
        int_n (int): 预测步长
    """
    # 1. 数据预处理
    df_temp = df_data[["etime", "tdate", "close", "fct"]].dropna(axis=0).reset_index(drop=True)

    # 2. 计算时间步长和收益率 这里的n_days其实是指预测几根bar
    t_delta = int(1 * int_n) if int_frequency == 15 else int(int(240 / int(int_frequency)) * int_n)
    # 这里的return，是我们的label，是未来t_delta天的收益率，他和我们计算绩效的时候，不一样。
    df_temp["ret"] = df_temp["close"].shift(-t_delta) / df_temp["close"] - 1
    df_temp = df_temp.dropna(axis=0).reset_index(drop=True)

    # 3. 训练集和测试集划分 !!!这里改成动态分配！！！！
    idx_train_set_end = df_temp[
        (df_temp["etime"].dt.year == 2019) &
        (df_temp["etime"].dt.month == 12) &
        (df_temp["etime"].dt.day == 31)
        ].index[0]

    X_train = df_temp.loc[:idx_train_set_end,"fct"].values.reshape(-1,1)
    Y_train = df_temp.loc[:idx_train_set_end,"ret"].values.reshape(-1,1)
    X_test = df_temp.loc[idx_train_set_end + 1:,"fct"].values.reshape(-1,1)

    # 4. 模型训练和预测 Y = AX + B
    model = LinearRegression(fit_intercept=True) # t.statistic, weight, bias
    model.fit(X_train,Y_train)
    y_train_pred = model.predict(X_train).flatten()
    y_test_pred = model.predict(X_test).flatten()

    # 5. 初始化绩效结果DataFrame
    indicators_frame = pd.DataFrame()

    # 6. 绩效计算
    # 6.1 结算每年的指标
    for year in df_temp["etime"].dt.year.unique():
        year_mask = df_temp["etime"].dt.year == year
        year_data = df_temp[year_mask]
        if len(year_data) > 0:
            year_positions = calculate_positions(
                model.predict(year_data["fct"].values.reshape(-1,1)).flatten()
            )
            year_returns, year_nav = calculate_returns_and_nav(year_data, year_positions)
            year_metrics = calculate_metrics_by_period(year_data, year_positions, year_returns, year_nav)
            indicators_frame = pd.concat([
                indicators_frame,
                pd.DataFrame(year_metrics, index=[year])
            ])
    # 6.2 计算样本内指标
    train_data = df_temp[:idx_train_set_end + 1]
    train_positions = calculate_positions(y_train_pred)
    train_returns, train_nav = calculate_returns_and_nav(train_data, train_positions)
    indicators_frame.loc["样本内"] = calculate_metrics_by_period(train_data, train_positions, train_returns, train_nav)
    # 6.3 计算样本外指标
    test_data = df_temp[idx_train_set_end + 1:]
    test_positions = calculate_positions(y_test_pred)
    test_returns, test_nav = calculate_returns_and_nav(test_data, test_positions)
    indicators_frame.loc["样本外"] = calculate_metrics_by_period(test_data, test_positions, test_returns, test_nav)
    # 6.4 计算总体指标
    total_positions = np.concatenate([train_positions, test_positions])
    total_returns, total_nav = calculate_returns_and_nav(df_temp, total_positions)
    indicators_frame.loc['总体'] = calculate_metrics_by_period(df_temp, total_positions, total_returns, total_nav)

    return indicators_frame

    




