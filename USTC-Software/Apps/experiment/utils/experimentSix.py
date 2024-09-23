# %% [markdown]
# # 首先考察工作曲线的情况

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from scipy import stats
# from matplotlib import font_manager
# from scipy.optimize import curve_fit
from datetime import datetime
from pathlib import Path

def experimentSix_process():
# 设置中文字体
    plt.rcParams['font.family'] = 'SimHei'
#获取文件存储路径
    current_path = Path(__file__).resolve().parent.parent
    target_path = current_path/"upload"

# %%
    def calculate_relative_times(element_value, df):
        """
        计算相对时间（单位是秒）并返回正值的列表。

        Parameters:
            element_value (datetime.time): 指定的时间元素，将与其他时间进行比较。
            df (pandas.DataFrame): 包含电脑时间列的数据框。

        Returns:
            list of float: 包含正值相对时间的列表，单位是秒。
        """
    # 存储正值的列表
        positive_times = []

    # 遍历 DataFrame 中的 "电脑时间" 列
        for other_time in df['电脑时间']:
        # 计算相对时间（单位是秒）
            relative_time = (datetime.combine(datetime.min, other_time) - datetime.combine(datetime.min, element_value)).total_seconds()
        # 如果相对时间为正，则将其添加到列表中
            if relative_time > 0:
                positive_times.append(relative_time)

        return positive_times


# %%
    def get_k(excel_name=None, sheet_name=None, specific_value=None, data_scale=None, png_title=None, a=5.22*10**-3):
        """
        根据提供的 Excel 文件和参数计算斜率和截距，绘制图表，并返回结果。

        Parameters:
            excel_name (str): Excel 文件名，包含数据。
            sheet_name (str): Excel 表格名，包含数据。
            specific_value (str): 用于查找索引的特定值（时间值）。
            data_scale (int): 用于截取数据的规模。
            png_title (str): 生成的图表的标题。
            a (float): 可选参数，用于斜率计算的常数，默认为 5.22 * 10**-3。

        Returns:
            tuple: 包含三个值，依次为斜率 (k)，截距 (intercept)，和最后一个温度值。

        Example:
            get_k(excel_name="data.xlsx", sheet_name="Sheet1", specific_value="15:40:36", data_scale=10, png_title="Data Plot", a=0.005)
        """
    # 读取 Excel 数据
        read_path = target_path/str(excel_name)
        df = pd.read_excel(read_path, sheet_name=f"{sheet_name}")

    # 查找特定值对应的索引
        index_L0 = df[df['电脑时间'].astype(str) == specific_value].index

        if not index_L0.empty:
            print("特定值对应的索引是:", index_L0[0])
        else:
            print("未找到匹配的值")

    # 提取所需数据
        L_0 = df["电导率测量结果"][index_L0].values[0]
        L_t = np.array(df["电导率测量结果"][index_L0[0]+1:].values)
        time_L0 = df["电脑时间"][index_L0].values[0]
        last_temperature_value = df["温度"].iloc[-1]

    # 计算相对时间
        positive_times_list = np.array(calculate_relative_times(time_L0, df))

    # 计算因变量
        right = (L_0 - L_t) / positive_times_list

    # 使用线性拟合计算斜率和截距
        coefficients = np.polyfit(L_t[data_scale:], right[data_scale:], 1)
        slope = coefficients[0]
        intercept = coefficients[1]

    # 创建拟合线
        fit_line = slope * L_t + intercept

    # 绘制图表
        plt.scatter(L_t[1:], right[1:], label='原始数据', color='blue')
        plt.plot(L_t, fit_line, label='拟合线', color='red')

    # 添加标签和标题
        plt.xlabel(r'$L_t(\mu S\cdot cm^{-1})$')
        plt.ylabel(r"$\frac{L_0-L_t}{t}(\mu S\cdot cm^{-1}\cdot s^{-1})$")
        plt.title(f'{png_title}')

    # 添加图例
        plt.legend()
        plt.savefig(target_path/f"{png_title}.png")
    # 显示和保存图像
        # plt.show()
    

    # 返回斜率、截距和最后一个温度值
        return slope/a, intercept/a, last_temperature_value




# %%
    sheet_35 = "35"
    specific_value_35 = "16:49:50"
    sheet_30 = "30"
    specific_value_30 = "15:40:56"
    k_30, L_inf_30, T_30 = get_k(excel_name="数据.xlsx", sheet_name=sheet_30, specific_value = specific_value_30, png_title="30下的拟合图像", data_scale=15)
    k_35, L_inf_35, T_35 = get_k(excel_name="数据.xlsx", sheet_name=sheet_35, specific_value = specific_value_35, png_title="35下的拟合图像", data_scale=5)

# %%
    def get_E_a(k_1=None, k_2=None, T_1=None, T_2=None):
        R = 8.314
        T_1 += 273.15
        T_2 += 273.15
        E_a = -(R*np.log(k_2/k_1))/(1/T_2-1/T_1)
        return E_a

# %%
    Ea = get_E_a(k_30, k_35, T_30, T_35)
    print(Ea,"\n", k_30, T_30,L_inf_30,"\n",k_35, T_35 ,L_inf_35 )
    date_write_path = target_path/"Ea.text"
    with open(date_write_path, "w", encoding="utf-8") as file:
        file.write(str(Ea)+'/n')
        file.write(f"{k_30},{T_30},{L_inf_30}"+'/n')
        file.write(f"{k_35},{T_35},{L_inf_35}")
    file.close()


