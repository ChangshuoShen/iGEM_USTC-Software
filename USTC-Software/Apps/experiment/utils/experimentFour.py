# %% [markdown]
# # 首先考察工作曲线的情况

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib import font_manager
from scipy.optimize import curve_fit
from pathlib import Path

def date_process():
# 设置中文字体
    plt.rcParams['font.family'] = 'SimHei'

#路径设置
    current_path = Path(__file__).resolve().parent.parent
    target_path = current_path/"upload/数据.xlsx"

# 读取Excel文件中的Sheet1工作表
    df = pd.read_excel(target_path, sheet_name='工作曲线')

# 打印前几行数据
    print(df)

# 对两列进行相除操作
    df['正丙醇的质量分数'] = df['正丙醇质量/g'] / df['总质量/g']

# 打印结果
    print(df)

# %%
# 提取 x 和 y 列的数据
    x = df['平均折光率/(nD)']
    y = df['正丙醇的质量分数']

# 绘制散点图
    plt.scatter(x, y, label='Data')

# 进行线性回归拟合
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * x + intercept

# 绘制拟合直线
    plt.plot(x, line, color='red', label='Fit')

# 添加图例和标签
    plt.legend()
    plt.xlabel('平均折光率/nD')
    plt.ylabel('正丙醇的质量分数')
    target_plt_path_first = current_path/"upload/线性拟合曲线.png"
    plt.savefig(target_plt_path_first)
# 显示图形
    # plt.show()


# %%
# 自定义指数函数模型
    def exponential_func(x, a, b):
        return a * np.exp(b * x)
# 拟合曲线
# popt, pcov = curve_fit(exponential_func, x, y)
    popt, pcov = curve_fit(exponential_func, x, y, p0=[1, 1], maxfev=100000)

# 绘制散点图
    plt.scatter(x, y, label='Data')

# 绘制拟合曲线
    x_fit = np.linspace(min(x), max(x), 100)  # 生成拟合曲线的 x 值范围
    y_fit = exponential_func(x_fit, *popt)
    plt.plot(x_fit, y_fit, color='red', label='Fit')

# 添加图例和标签
    plt.legend()
    plt.xlabel('平均折光率/nD')
    plt.ylabel('正丙醇的质量分数')
    target_plt_path_second = current_path/"upload/指数拟合曲线.png"
    plt.savefig(target_plt_path_second)
# 显示图形
    # plt.show()

# %% [markdown]
# # 然后考察原始数据的处理
# %%
# 读取Excel文件中的Sheet1工作表
    df_origin = pd.read_excel(target_path, sheet_name='原始数据记录')
# print(df_origin)
# 对两列进行相除操作
    df_origin['平均温度'] = (df_origin['沸点1/℃']+df_origin['沸点2/℃']+df_origin['沸点3/℃']) / 3
    df_origin['气相中正丙醇的质量分数'] = exponential_func(df_origin['气相折光率/nD'], *popt)
    df_origin['液相中正丙醇的质量分数'] = exponential_func(df_origin['液相折光率/nD'], *popt)
# 打印结果
    print(df_origin['平均温度'])

# %%
    df_origin['矫正后的平均温度'] = df_origin['平均温度'] + ((273.15 + df_origin['平均温度']) / 10 ) * ((101325-df_origin['气压/kPa'].mean()*1000) / 101325)
# plt.plot(df_origin['气相中正丙醇的质量分数'], df_origin['平均温度'], color='red', label='gas phase',marker="o")
# plt.plot(df_origin['液相中正丙醇的质量分数'], df_origin['平均温度'], color='blue', label='liquid phase',marker="*")
    plt.plot(df_origin['气相中正丙醇的质量分数'], df_origin['矫正后的平均温度'], color='red', label='re gas phase ',marker="o")
    plt.plot(df_origin['液相中正丙醇的质量分数'], df_origin['矫正后的平均温度'], color='blue', label='re liquid phase',marker="*")
# 添加图例和标签
    plt.legend()
    plt.xlabel('正丙醇的质量分数')
    plt.ylabel('平均沸点/℃')
    target_plt_path_third = current_path/"upload/质量分数-平均沸点.png"
    plt.savefig(target_plt_path_third)
# 显示图形
    # plt.show()

# %%



