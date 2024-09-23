#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
# from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

def experimentEight_process():

# 指定Excel文件路径
    current_path = Path(__file__).resolve().parent.parent
    target_path = current_path/"upload"
    excel_file_path = target_path/'数据.xlsx'

# 使用Pandas的read_excel函数加载Excel文件并转化为DataFrame
    df = pd.read_excel(excel_file_path)

# 选择要计算平均值的列
    columns_to_average = ["流出时间1", "流出时间2", "流出时间3"]

# 将时间转换为秒数并计算平均值
    df[columns_to_average] = df[columns_to_average].applymap(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    df["平均流出时间"] = df[columns_to_average].mean(axis=1)

# 将秒数转换回时间格式
# df["平均流出时间"] = df["平均流出时间"].apply(lambda x: timedelta(seconds=x))

# 打印带有新列的DataFrame
    print(df)


# In[66]:


# 打印"平均流出时间"列的最后一个元素
    relative_C = df["相对浓度"].iloc[:-1].copy()
    t0 = df["平均流出时间"].iloc[-1]
# print(t0)
    t_list = df["平均流出时间"].iloc[:-1]
    t0 =  107.6764738536693
    print(t0)
    relative_times =t_list / t0
    eta_r = relative_times.copy()
    eta_sp = (relative_times - 1).copy()
# print(t_list / t0)
# print(t_list)
# print(relative_times)
# print(eta_r, eta_sp)
    eta_sp_C = eta_sp / relative_C
    ln_eta_r_C = np.log(eta_r) / relative_C
# print(relative_C)
    print("eta_c",eta_sp_C, "eta_ln",ln_eta_r_C)


# In[74]:


    from typing import List, Union, Tuple
# import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    # plt.show()
# get_ipython().run_line_magic('matplotlib', 'inline')


    def linear_fit_and_plot(x_data: Union[np.ndarray, List[float]],
                         y_data: Union[np.ndarray, List[float]],
                         save_path: str,
                         png_title) -> Tuple[str, float, float]:
        """
        Perform linear fit on the given data and plot the scatter plot and the fitted line.

        Parameters:
        - x_data: Input x data (array or list)
        - y_data: Input y data (array or list)

        Returns:
        - Tuple containing:
            1. LaTeX code for the linear equation
            2. Slope of the fitted line
            3. Intercept of the fitted line
        """
    # Convert input data to numpy arrays
        plt.figure()
        x_data = np.array(x_data)
        y_data = np.array(y_data)

    # Perform linear fit
        slope, intercept = np.polyfit(x_data, y_data, 1)

    # Generate the fitted line
        x_fit = np.arange(0, max(x_data)+0.2, 0.1)
        fit_line = np.polyval([slope, intercept], x_fit)

    # Plot the scatter plot and the fitted line
        plt.scatter(x_data, y_data, label='Original Data')
        plt.plot(x_fit, fit_line, color='red', label='Linear Fit')
        plt.xlabel(r'$C$')
        plt.ylabel(r"$\frac{\eta_{sp}}{C}-C$")
        plt.title(png_title)
        plt.legend()
    # plt.savefig(save_path)
        # plt.show()
        save_path_path = target_path/save_path
        plt.savefig(save_path_path)
    

    # Format the linear equation in LaTeX
        equation_latex = f"${'%.2f' % slope}x + {'%.2f' % intercept}$"

        return equation_latex, slope, intercept

# Example usage:
    x_data_example = relative_C[1:5]
    y_data_example = eta_sp_C[1:5]

    print(x_data_example, y_data_example)

    equation, slope, intercept = linear_fit_and_plot(x_data_example, y_data_example,"eta_sp_C_t0.png", r"$\frac{\eta_{sp}}{C}-C$")

    print("Linear Equation:", equation)
    print("Slope:", slope)
    print("Intercept:", intercept)


# Linear Equation: $-0.09x + 0.77$
# Slope: -0.08817605858825003
# Intercept: 0.7723515368402318
# <Figure size 432x288 with 0 Axes>
# 
# Linear Equation: $0.21x + 0.77$
# Slope: 0.21122842523376334
# Intercept: 0.7724032373324471
# <Figure size 432x288 with 0 Axes>

# In[75]:


# from typing import List, Union, Tuple
# import numpy as np
# import matplotlib.pyplot as plt

    def linear_fit_and_plot(x_data: Union[np.ndarray, List[float]],
                         y_data: Union[np.ndarray, List[float]],
                         label: str) -> Tuple[str, float, float]:
        """
        Perform linear fit on the given data and plot the scatter plot and the fitted line.

        Parameters:
        - x_data: Input x data (array or list)
        - y_data: Input y data (array or list)
        - label: Label for the data in the plot

        Returns:
        - Tuple containing:
            1. LaTeX code for the linear equation
            2. Slope of the fitted line
            3. Intercept of the fitted line
        """
    # Convert input data to numpy arrays
        x_data = np.array(x_data)
        y_data = np.array(y_data)

    # Perform linear fit
        slope, intercept = np.polyfit(x_data, y_data, 1)

        x_fit = np.arange(0, max(x_data)+0.2, 0.1)
        fit_line = np.polyval([slope, intercept], x_fit)

    # Plot the scatter plot and the fitted line
        plt.scatter(x_data, y_data, label=f'{label} - Original Data')
        plt.plot(x_fit, fit_line, label=f'{label} - Linear Fit')

    # Format the linear equation in LaTeX
        equation_latex = f"${'%.2f' % slope}x + {'%.2f' % intercept}$"

        return equation_latex, slope, intercept

# Example usage:
    x_data_example_1 = relative_C[1:5]
    y_data_example_1 = eta_sp_C[1:5]

    x_data_example_2 = relative_C[1:5]
    y_data_example_2 = ln_eta_r_C[1:5]

    plt.figure()
    equation_1, slope_1, intercept_1 = linear_fit_and_plot(x_data_example_1, y_data_example_1, label=r"$\frac{\eta_{sp}}{C}$")
    equation_2, slope_2, intercept_2 = linear_fit_and_plot(x_data_example_2, y_data_example_2, label=r"$\ln\frac{\eta_{r}}{C}$")

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    png_save_path = target_path/"all.png"
    plt.savefig(png_save_path)
    # plt.show()

    print("Linear Equation 1:", equation_1)
    print("Slope 1:", slope_1)
    print("Intercept 1:", intercept_1)

    print("Linear Equation 2:", equation_2)
    print("Slope 2:", slope_2)
    print("Intercept 2:", intercept_2)


# In[69]:


# 外推得到t0
    def linear_fit_and_plot(x_data: Union[np.ndarray, List[float]],
                         y_data: Union[np.ndarray, List[float]],
                         save_path: str,
                         png_title) -> Tuple[str, float, float]:
        """
        Perform linear fit on the given data and plot the scatter plot and the fitted line.

        Parameters:
        - x_data: Input x data (array or list)
        - y_data: Input y data (array or list)

        Returns:
            - Tuple containing:
            1. LaTeX code for the linear equation
            2. Slope of the fitted line
            3. Intercept of the fitted line
        """
    # Convert input data to numpy arrays
        plt.figure()
        x_data = np.array(x_data)
        y_data = np.array(y_data)

    # Perform linear fit
        slope, intercept = np.polyfit(x_data, y_data, 1)

    # Generate the fitted line
        x_fit = np.arange(0, max(x_data)+0.2, 0.1)
        fit_line = np.polyval([slope, intercept], x_fit)

    # Plot the scatter plot and the fitted line
        plt.scatter(x_data, y_data, label='Original Data')
        plt.plot(x_fit, fit_line, color='red', label='Linear Fit')
        plt.xlabel(r'$C$')
        plt.ylabel(r"$t(s)$")
        plt.title(png_title)
        plt.legend()
        save_path_end = target_path/save_path
        plt.savefig(save_path_end)
        # plt.show()
    

    # Format the linear equation in LaTeX
        equation_latex = f"${'%.2f' % slope}x + {'%.2f' % intercept}$"

        return equation_latex, slope, intercept


# Example usage:
    x_data_example = relative_C
    y_data_example = t_list

    equation, slope, intercept = linear_fit_and_plot(x_data_example, y_data_example,"get_t0.png", r"$t-C$")

    print("Linear Equation:", equation)
    print("Slope:", slope)
    print("Intercept:", intercept)

