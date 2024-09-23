#!/usr/bin/env python
# coding: utf-8

# In[90]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体，这里以微软雅黑为例
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# def plot_xlsx_data(file_path):
#     # 读取 Excel 文件
#     data = pd.read_excel(file_path)

#     # 提取第一行作为横坐标
#     x = data.iloc[:, 0].values

#     # 提取第二行作为纵坐标
#     y = data.iloc[:,1].values

#     # 绘制点线图
#     plt.figure(figsize=(8, 6))
#     plt.plot(x, y, marker='o', linestyle='-')
#     plt.title('苯甲酸燃烧的温度-时间曲线')
#     plt.xlabel(r'时间/s')
#     plt.ylabel(r'温度/$^\circ$C')
#     plt.grid(True)
#     plt.show()


def plot_xlsx_data(file_path, pre=150, last=200,intersection_x = 6 ):
    # 读取 Excel 文件
    data = pd.read_excel(file_path)

    # 提取第一行作为横坐标
    x = data.iloc[:, 0].values

    # 提取第二行作为纵坐标
    y = data.iloc[:, 1].values

    # 前pre个数据点进行线性拟合
    x_fit_pre = x[:pre]
    y_fit_pre = y[:pre]

    # 后last个数据点进行线性拟合
    x_fit_last = x[-last:]
    y_fit_last = y[-last:]

    # 进行线性拟合
    coeffs_pre = np.polyfit(x_fit_pre, y_fit_pre, 1)  # 前pre个数据点的拟合
    coeffs_last = np.polyfit(x_fit_last, y_fit_last, 1)  # 后last个数据点的拟合

    slope_pre, intercept_pre = coeffs_pre[0], coeffs_pre[1]  # 斜率和截距（前pre个数据点）
    slope_last, intercept_last = coeffs_last[0], coeffs_last[1]  # 斜率和截距（后last个数据点）

    # 生成拟合直线的数据
    line_x_pre = np.linspace(min(x_fit_pre), max(x) / 2, 100)
    line_y_pre = slope_pre * line_x_pre + intercept_pre

    line_x_last = np.linspace(max(x) / 4, max(x_fit_last), 100)
    line_y_last = slope_last * line_x_last + intercept_last

     # 绘制点线图和拟合直线
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', label='原始数据')
    plt.plot(line_x_pre, line_y_pre, linestyle='--', color='red', label='前{}个数据点拟合'.format(pre))
    plt.plot(line_x_last, line_y_last, linestyle='--', color='blue', label='后{}个数据点拟合'.format(last))

    # 添加垂直线
    # 7分钟对应的时间（以秒为单位）
    plt.axvline(x=intersection_x, color='green', linestyle=':', label='垂直线')

    # 计算曲线与垂直线的交点
    intersection_y = np.interp(intersection_x, x, y)
    plt.axhline(y=intersection_y, color='orange', linestyle='-.', label='水平线')
    plt.scatter(intersection_x, intersection_y, color='black', marker='x', s=100, label='交点')
    plt.annotate(f'({intersection_x}, {intersection_y:.2f})', (intersection_x, intersection_y),
                 textcoords="offset points", xytext=(10, -10), ha='center')

    # 计算两条拟合直线与垂直线的交点
    intersection_pre_x = intersection_x
    intersection_pre_y = slope_pre * intersection_pre_x + intercept_pre

    plt.scatter(intersection_pre_x, intersection_pre_y, color='purple', marker='x', s=100, label='拟合直线交点')
    plt.annotate(f'({intersection_pre_x:.2f}, {intersection_pre_y:.2f})', (intersection_pre_x, intersection_pre_y),
                 textcoords="offset points", xytext=(10, -10), ha='center')
    
    # 计算两条拟合直线与垂直线的交点
    intersection_last_x = intersection_x
    intersection_last_y = slope_last * intersection_last_x + intercept_last

    plt.scatter(intersection_last_x, intersection_last_y, color='yellow', marker='x', s=100, label='拟合直线交点')
    plt.annotate(f'({intersection_last_x:.2f}, {intersection_last_y:.2f})', (intersection_last_x, intersection_last_y),
                 textcoords="offset points", xytext=(10, -10), ha='center')

    plt.title('燃烧的温度-时间曲线')
    plt.xlabel(r'时间/s')
    plt.ylabel(r'温度/$^\circ$C')
    plt.legend()  # 添加图例
    plt.grid(True)
    # plt.show()

    return (slope_pre, intercept_pre), (slope_last, intercept_last), (intersection_x, intersection_y), (intersection_pre_x, intersection_pre_y), (intersection_last_x, intersection_last_y)



# In[111]:

def process_one():
    pre = int(360/30)
    last = int(3000/30)
    current_path = Path(__file__).resolve().parent.parent
    target_path = current_path/"upload/萘.xlsx"

    (slope_pre, intercept_pre), (slope_last, intercept_last), (intersection_x, intersection_y), (intersection_pre_x, intersection_pre_y), (intersection_last_x, intersection_last_y) = plot_xlsx_data(target_path, pre=pre, last=last)
    print("前{}个数据点的斜率和截距:".format(pre), slope_pre, intercept_pre)
    print("后{}个数据点的斜率和截距:".format(last), slope_last, intercept_last)
    print("与曲线相交的点坐标:", intersection_x, intersection_y)
    print("与前端拟合直线相交的点坐标:", intersection_pre_x, intersection_pre_y)
    print("与后端拟合直线相交的点坐标:", intersection_last_x, intersection_last_y)
    
    #创建文件并记录数值
    text_path = current_path/"upload/date萘.txt"
    with open(text_path, "w", encoding="utf-8") as file:
        file.write("前{}个数据点的斜率和截距： {}, {}".format(pre, slope_pre, intercept_pre))
        file.write("后{}个数据点的斜率和截距:  {}, {}".format(last, slope_last, intercept_last))
        file.write("与曲线相交的点坐标:  {}, {}".format(intersection_x, intersection_y))
        file.write("与前端拟合直线相交的点坐标:  {}, {}".format(intersection_pre_x, intersection_pre_y))
        file.write("与后端拟合直线相交的点坐标:  {}, {}".format(intersection_last_x, intersection_last_y))
    file.close()


# In[100]:

def process_two():
    pre = int(200/30)
    last = int(600/30)
    current_path = Path(__file__).resolve().parent.parent
    target_path = current_path/"upload/苯甲酸.xlsx"

    (slope_pre, intercept_pre), (slope_last, intercept_last), (intersection_x, intersection_y), (intersection_pre_x, intersection_pre_y), (intersection_last_x, intersection_last_y) = plot_xlsx_data(target_path, pre=pre, last=last, intersection_x=5)
    print("前{}个数据点的斜率和截距:".format(pre), slope_pre, intercept_pre)
    print("后{}个数据点的斜率和截距:".format(last), slope_last, intercept_last)
    print("与曲线相交的点坐标:", intersection_x, intersection_y)
    print("与前端拟合直线相交的点坐标:", intersection_pre_x, intersection_pre_y)
    print("与后端拟合直线相交的点坐标:", intersection_last_x, intersection_last_y)
    
    #创建文件并记录数值
    text_path = current_path/"upload/date苯甲酸.txt"
    with open(text_path, "w", encoding="utf-8") as file:
        file.write("前{}个数据点的斜率和截距： {}, {}".format(pre, slope_pre, intercept_pre))
        file.write("后{}个数据点的斜率和截距:  {}, {}".format(last, slope_last, intercept_last))
        file.write("与曲线相交的点坐标:  {}, {}".format(intersection_x, intersection_y))
        file.write("与前端拟合直线相交的点坐标:  {}, {}".format(intersection_pre_x, intersection_pre_y))
        file.write("与后端拟合直线相交的点坐标:  {}, {}".format(intersection_last_x, intersection_last_y))
    file.close()

def process_three():
    process_one()
    process_two()