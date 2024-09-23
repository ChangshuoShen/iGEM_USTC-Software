import os
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import math
import numpy as np
from sklearn.metrics import r2_score
from pathlib import Path

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'

# 文件夹路径
current_path = Path(__file__).resolve().parent.parent
folder_path = current_path/'upload'

# 获取文件夹中所有xlsx文件
files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]

def xlsx_process_one():
    # 创建一个空的图表
    plt.figure(figsize=(8, 6))

    # 读取每个文件并绘制曲线
    for file in files:
        file_path = os.path.join(folder_path, file)
        # print(file_path)
        # 读取xlsx文件
        data = pd.read_excel(file_path)
    
        # 提取横坐标和纵坐标列数据
        x = data.iloc[:, 0]
        y = data.iloc[:, 1]
    
        # 绘制曲线并添加图例
        plt.plot(x, y, label=file.split('_')[0])

    # 添加图例、坐标轴标签和标题
    plt.legend()
    plt.xlabel('λ(nm)')
    plt.ylabel('吸光度(Abs)')
    plt.title('吸光度与波长')

    # 显示图表
    plt.grid(True)
    png_path_first = folder_path/"Abs_λ.png"
    plt.savefig(png_path_first)
    # plt.show()
def xlsx_process_two():
    # 创建一个空的图表
    plt.figure(figsize=(8, 6))

    # 读取每个文件并绘制曲线
    for file in files:
        file_path = os.path.join(folder_path, file)
        # print(file_path)
        # 读取xlsx文件
        data = pd.read_excel(file_path)

        # 设置生成文件名称
        png_path = os.path.splitext(file_path)[0] + ".png"

        # 提取横坐标和纵坐标列数据
        x = data.iloc[:, 0]
        y = data.iloc[:, 1]
    
        # 绘制曲线并添加图例
        plt.plot(x, y, label=file.split('_')[0])

        # 添加图例、坐标轴标签和标题
        plt.legend()
        plt.xlabel('λ(nm)')
        plt.ylabel('吸光度(Abs)')
        plt.title('吸光度与波长')

        # 显示图表
        plt.grid(True)
        plt.savefig(png_path)
        # plt.show()
def xlsx_process_three():
    D_dic = {}

    for file in files:
        file_path = os.path.join(folder_path, file)
    
        # 读取xlsx文件
        data_D1 = pd.read_excel(file_path)
    
        value_to_find = 591
        second_column_data = data_D1[data_D1.iloc[:, 0] == value_to_find].iloc[:, 1]
        filename_without_extension = os.path.splitext(file)[0]
    
        # 存储数据到字典中
        D_dic[filename_without_extension] = list(second_column_data)[0]
        # D_dic[file.split("_")[0]] = list(second_column_data)[0]

    # 在循环结束后提取数据
    pprint(D_dic)
    data_dict = D_dic

    # 获取D1和D2的值
    print(data_dict)
    D1 = data_dict['酸']
    D2 = data_dict['碱']

    # 初始化存储计算结果的字典
    result_dict = {}
    result_list = []

    # 计算并存储结果
    for key, value in data_dict.items():
        if key in ['酸', '碱']:
            # result_dict[key] = value
            pass
        else:
            D = value
            if (D == D1) or (D == D2) or (D2 - D == 0) or (D - D1 <= 0):
                # result_dict[key] = 'Invalid Calculation'
                print('Invalid Calculation')
            else:
                print(key,D1, D2, D, "\n", "-"*12)
                result = math.log10((D - D1) / (D2 - D))
                # print(result)
                result_dict[key] = round(result, 4) if not math.isnan(result) else 'Invalid Calculation'
                result_list.append(round(result, 4)) if not math.isnan(result) else 'Invalid Calculation'

    print(result_dict)
    print(result_list)
    # 设置中文字体，确保负号正常显示
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False # 显示负号

    # 数据
    ph_list = [3.15, 3.42, 3.64, 3.82, 4.21, 4.42, 4.6]
    abs_list = result_list

    # 散点图
    plt.scatter(ph_list, abs_list, label='数据点')

    # 进行拟合直线
    coefficients = np.polyfit(ph_list, abs_list, 1)
    poly = np.poly1d(coefficients)
    plt.plot(ph_list, poly(ph_list), label='拟合直线', color='red')

    # 设置图例和标签
    plt.legend()
    plt.xlabel('pH')
    plt.ylabel('lgX')

    # 显示图表
    plt.grid(True)
    ph_lgx_path = folder_path/"ph_lgx.png"
    plt.savefig(ph_lgx_path)
    # plt.show()

    # 数据
    ph_list = [3.15, 3.42, 3.64, 3.82, 4.21, 4.42, 4.6]
    abs_list =result_list

    # 进行拟合直线
    coefficients = np.polyfit(ph_list, abs_list, 1)
    poly = np.poly1d(coefficients)

    # 获取截距和R²值
    intercept = poly[0]
    r_squared = r2_score(abs_list, poly(ph_list))
    slope = coefficients[0]

    #创建文件记录数据
    date_path = folder_path/"date.text"
    with open(date_path, "w", encoding="utf-8") as file:
        file.write("斜率：")
        file.write(str(slope))
        file.write("截距：")
        file.write(str(intercept))
        file.write("R² 值")
        file.write(str(r_squared))
    file.close()

    print(f"斜率: {slope}")
    print(f"截距: {intercept}")
    print(f"R² 值: {r_squared}")
def xlsx_process():
    xlsx_process_one()
    xlsx_process_two()
    xlsx_process_three()
