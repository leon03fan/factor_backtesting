from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as py
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from pprint import pprint

from func import backtest


def generate_etime_close_data_divd_time(date_bgn:str, date_end:str, str_index_code:str, int_frequency:int) -> pd.DataFrame:
    """从总数据集中按照date_bgn和date_end筛选要进行后续因子测试的数据

    Args:
        date_bgn (str): 开始日期
        date_end (str): 结束日期
        str_index_code (str): 指数代码
        int_frequency (int): 需要预测的bar数量

    Returns:
        pd.DataFrame: 包含"etime", "tdate", "close"的筛选后的数据（除了close，也可以包含其他因子）
    """
    str_file_path = f"./in/{str_index_code}_{int_frequency}.csv"
    df_bar = pd.read_csv(str_file_path)

    df_bar["etime"] = pd.to_datetime(df_bar["etime"])
    df_bar["tdate"] = pd.to_datetime(df_bar["etime"]).dt.date
    df_bar["label"] = "-1"
    dt_bgn = pd.to_datetime(date_bgn)
    dt_end = pd.to_datetime(date_end)
    ser_mask = df_bar["etime"].between(dt_bgn, dt_end)

    return df_bar[ser_mask][["etime", "tdate", "close"]].reset_index(drop=True)


def iter_func(params:list) -> pd.DataFrame:
    """
    迭代函数，用于遍历因子数据并进行回测

    Args:
        params (list[tuple]): 包含频率、因子名称、因子数据和原始数据的数据元组
    Returns:
        pd.DataFrame: 回测结果的DataFrame
    """
    # 元祖传参
    int_freq, str_col_name, ser_fct_series, df_data = params # 频率，因子名称，因子数据，daframe(etime，tdate，close)
    # 表格构建 !!!!!这里应该用index进行对齐!!!!!
    df_data["fct"] = ser_fct_series.values
    # n_days非常重要,调节我们的预测步长，
    ind_frame = backtest(original_data=df_data, index_code="510050", frequency=int_freq, n_days=1)

    print('frequency: {}\nfct_name: {}\n'.format(int_freq, str_col_name))
    print(ind_frame)
    print('\n')
    print('夏普比率（样本外）：{}\n\n'.format(ind_frame.loc['样本外', '夏普比率']))  # 输出样本外夏普比率

    ind_frame['params'] = str_col_name

    return ind_frame


if __name__ == "__main__":
    # 初始化
    start_time = time.time()
    df_final = pd.DataFrame()
    # 因子数据和文件路径
    str_file_path = "./in/fct_compare_0702.csv"
    int_job_num = 8
    int_freq = 15

    # 引入行情数据，注意日期
    df_original =  generate_etime_close_data_divd_time(date_bgn='2005-02-23', date_end='2022-11-30', str_index_code='510050.SH', int_frequency=int_freq)
    df_fct = pd.read_csv(str_file_path, index_col=0)

    inputs = []
    for str_fct_name in df_fct.columns:
        inputs.append((int_freq, str_fct_name, df_fct[str_fct_name], df_original)) 
    # pprint(inputs)    

    # 周期频率，因子名称，对应的因子数据，etime，tdate，close
    with ProcessPoolExecutor(max_workers=int_job_num) as executor:
        results = {executor.submit(iter_func, i) : i for i in inputs}
        for i in as_completed(results):
            try:
                df_final = pd.concat([df_final, i.result()])
            except Exception as exception:
                print(exception)
                
    df_final.to_csv("./out/单因子测试结果.csv")
    end_time = time.time()
    print('Time cost:====', end_time - start_time)