import numpy as np
import pandas as pd


df = pd.DataFrame()
df["data"] = np.random.rand(20)
# 数据也可以是series格式

# 简单移动平均
#simp_moving_avg = df["data"].rolling(window=window, min_periods=1).mean()
# 加权移动平均
#weighted_moving_avg = df["data"].rolling(window=window, min_periods=1, win_type="cosine").mean()
# 指数加权移动平均
#ewma = df["data"].ewm(alpha=alpha, min_periods=1).mean()

def siimp_window_filter(windowsize,s):
    tmps = s.copy()
    i=0
    while(i<len(s)):
        tmp_window =np.array(s[i:min(i+windowsize,len(s)-1)])
        tmp_mean = np.mean(tmp_window)
        if sum(tmp_window>tmp_mean)>windowsize/2:
            half_mean = np.mean(tmp_window[tmp_window>tmp_mean])
            tmp_window[tmp_window<tmp_mean]=half_mean
        else:
            half_mean = np.mean(tmp_window[tmp_window<tmp_mean])
            tmp_window[tmp_window>tmp_mean]=half_mean
        tmps[i:min(i+windowsize,len(s)-1)]=tmp_window
        i=i+windowsize
    return tmps


def exponential_smoothing(alpha, s):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型参数， list
    '''
    s_temp = []
    s_temp.append(s[0])
    print(s_temp)
    for i in range(1, len(s), 1):
        s_temp.append(alpha * s[i-1] + (1 - alpha) * s_temp[i-1])
    return s_temp

