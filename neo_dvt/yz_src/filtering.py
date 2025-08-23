import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def filtering(recording_traces, filter_params,fs):
    filter_en = filter_params['filter_en']
    low_cutoff = filter_params['corner_freq'][0]
    high_cutoff = filter_params['corner_freq'][1]
    order = filter_params['order']  # 滤波器阶数
    axis = filter_params['axis']
    '''
    frequencies, power_spectrum = signal.welch(recording_traces, fs)
    # 绘制频谱图
    plt.figure(figsize=(8, 6))
    plt.semilogy(frequencies, power_spectrum)
    plt.title('信号频谱')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度')
    plt.grid(True)
    plt.show()
    '''
    if (filter_en == 1):
        # 计算归一化截止频率
        low_cutoff_normalized = low_cutoff / (0.5 * fs)  
        high_cutoff_normalized = high_cutoff / (0.5 * fs)
        b, a = signal.butter(order, [low_cutoff_normalized, high_cutoff_normalized], btype='band', analog=False, output='ba')
        recording_traces = signal.lfilter(b, a, recording_traces,axis = axis)
        #print('axis',axis)
    else:
        recording_traces = recording_traces
    '''
    # 使用 scipy.signal 中的 welch 函数获取频谱
    frequencies, power_spectrum = signal.welch(recording_traces, fs)
    
    # 绘制频谱图
    plt.figure(figsize=(8, 6))
    plt.semilogy(frequencies, power_spectrum)
    plt.title('信号频谱')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度')
    plt.grid(True)
    plt.show()
    '''
    return recording_traces