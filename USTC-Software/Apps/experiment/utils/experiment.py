#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import re


class OscillationAnalyzer:
    def __init__(self, txt_path):
        self.df = None
        self.oscillation_x = None
        self.oscillation_y = None
        self.png_path = None

        self.read_txt_to_df(txt_path)
        self.plot_oscillation('t/s', 'T/°C', marker='>', marker_size=0.1, linestyle='--', marker_color='lightblue',line_color='red', linewidth=1)

        # 计算震荡期间数据的平均值和标准差
        self.mean = np.mean(self.oscillation_y)
        self.std = np.std(self.oscillation_y)
        _, self.period = self.estimate_period()

    def read_txt_to_df(self, filename):
        self.png_path = os.path.splitext(filename)[0] + ".png"   

        with open(filename, 'r') as f:
            content = f.readlines()

        content = [re.split(r'\s+', line.strip()) for line in content]
        self.df = pd.DataFrame(content, columns=['t/s', 'T/°C'])

    def plot_oscillation(self, x_name, y_name, marker='>', marker_size=1, linestyle='-', marker_color='red',
                         line_color='blue', linewidth=1):
        x = self.df[x_name].astype(float)
        y = self.df[y_name].astype(float)

        threshold_index = 0
        for i in range(1, len(y)):
            if y[i] < y[i - 1]:
                threshold_index = i
                break

        self.oscillation_x = x[threshold_index:]
        self.oscillation_y = y[threshold_index:]

        plt.plot(self.oscillation_x, self.oscillation_y, linestyle=linestyle, color=line_color, linewidth=linewidth)
        plt.scatter(self.oscillation_x, self.oscillation_y, marker=marker, color=marker_color, linewidth=marker_size)

        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Plot of " + y_name + " vs " + x_name)
        plt.savefig(self.png_path)

        # plt.show()


    def estimate_period(self, window_size=20):
        smoothed_data = pd.Series(self.oscillation_y).rolling(window=window_size).mean().dropna().values

        peaks, _ = find_peaks(smoothed_data)
        periods = np.diff(peaks)
        mean_period = np.mean(periods)

        return len(peaks) - 1, mean_period


# In[ ]:


import os
from glob import glob
from pathlib import Path

def process_txt_files():
    # 获取当前目录下所有的txt文件路径
    current_path = Path(__file__).resolve().parent.parent
    target_path = current_path/"upload"
    directory = target_path
    txt_files = glob(str(directory) + "/*.txt")

    data = []

    # 遍历每个txt文件并处理
    for txt_file in txt_files:
        print(f"Processing file: {txt_file}")
        analyzer = OscillationAnalyzer(txt_file)

        data.append({
            'File': analyzer.png_path.split(".")[0],
            'Period': analyzer.period,
            'Mean': analyzer.mean,
            'Std': analyzer.std
        })
        # print("Average period:", analyzer.period)
        # print("Mean:", analyzer.mean)
        # print("Std:", analyzer.std)
        # print("---------------------------------------------")

    # 将结果存储在DataFrame中
    df_results = pd.DataFrame(data)
    csv_path = current_path/"upload/log.csv"
    df_results.to_csv(csv_path, index=True)  # 保存为CSV文件
    print(df_results)

