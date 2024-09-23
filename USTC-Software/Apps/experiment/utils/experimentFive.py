# %% [markdown]
# # 首先考察工作曲线的情况

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from scipy import stats
# from matplotlib import font_manager
from scipy.optimize import curve_fit



# %% [markdown]
# ## 读取数据
# 
# 这里还有一些麻烦,因为这个`xls`文件是旧版本,现在的`df`很难处理,所以我是先使用wps转化`xls`为`xlsx`,然后再读取.并且,为了简化变量,也是减轻内存的负担,我删除了不重要的行,只是保留了关心的**温度和旋光**

# %%
import os
# import pandas as pd
from pathlib import Path

def experimentFive_process():
# 设置中文字体
    plt.rcParams['font.family'] = 'SimHei'


# 设置文件夹路径
    current_path = Path(__file__).resolve().parent.parent
    folder_path = current_path/"upload"

# 获取文件夹中的所有xls文件
    files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]
    df_dic = {}
# 循环处理每个文件
    for file in files:
    # 读取Excel文件
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path)

    # 删除指定行
        df = df.drop([0, 1, 2, 3, 65])
        df = df.iloc[:, [8, 9]] # 这样会删除原来的df,要小心
    # df = df.rename(columns={8: "旋光",9:"温度"})  # 因为本来就没有名字,所以就谈不上"rename"
        title = file.split("_")[0]
        df.columns = [f"{title}", f"{title}tem"]   # 这种方法不好,但是因为我们这里的数量少,所以还可以

        df_dic[f'{file.split("_")[0]}'] = df

# %%
    def get_tem(df=None, tem_list=[]):
        np_array = df.values
        tem = np.mean(np_array[:,1])
    # round(num, 2)
        tem_list.append(round(tem,1)+273.15)
        return tem_list

    tem_list = []
    for df in df_dic:
        tem_list = get_tem(df_dic[df], tem_list)
    print(tem_list)

# %% [markdown]
# ## 画图
# 
# 这里是为了得到$\alpha_{∞}$和指数函数的参数

# %%
# import numpy as np
# from scipy.optimize import curve_fit

    def plot_exp(df=None,exp_param = {}):

    # 自定义指数函数模型
        def exponential_func(t, a, b, c):
            return a * np.exp(b * t) - c

    # 提取x和y数据
        np_array = df.values
        title = df.keys()
        x =np.array(list(range(len(np_array[:,0]))))
        y = np_array[:, 0]

    # 拟合曲线
        popt, pcov = curve_fit(exponential_func, x, y, p0=[1, -1, 0])

    # 输出拟合参数
        exp_param[f"{title[0]}-param"] = popt
        exp_param[f"{title[0]}-data"] = y

    # 绘制原始数据散点图
        plt.scatter(x, y, label='Original Data')

    # 绘制拟合曲线
        x_fit = np.linspace(x.min(), x.max()*3, 100)  # 生成拟合曲线的 x 值范围
        y_fit = exponential_func(x_fit, *popt)
        plt.plot(x_fit, y_fit, color='red', label='Exponential Fit')
        exp_param[f"{title[0]}-alpha_inf"] = y_fit[-1]

    # 添加图例和标签
        plt.legend()
        plt.xlabel('时间(s)')
        plt.ylabel('旋光(°)')
        plt.title(f'{title[0]}')
        plt.savefig(folder_path/f'{title[0]}.png')
    # # 显示图形
    # plt.show()
        plt.close()
    
        return exp_param

    for df in df_dic:
        param = plot_exp(df_dic[df])

# %% [markdown]
# # 进入数据处理
# 
# 

# %% [markdown]
# ## 进入实际的数据处理
# 
# 首先得到了差和对数

# %%
    def transform_df(param=None):
    # 创建一个空的DataFrame
        df = pd.DataFrame()
        xg = []
        inf = []
        data = []
    
    # 遍历字典中的每个键值对
        for key, value in param.items():
            if key.endswith('param'):
                xg.append(value)
            elif key.endswith('inf'):
                inf.append(value)
            elif key.endswith('data'):
                data.append(value)

        df["param"] = xg
        df["inf"] = inf
        df["data"] = data
    # 将DataFrame的索引重置为从0开始的递增整数
        df.reset_index(drop=True, inplace=True)

    # # 使用rename()函数对df的列进行重命名
    # df.rename(columns=dict(zip(df.rows, name)), inplace=True)
    # 使用rename()函数对df的每一行进行重命名
        name = [n.split("-")[0] for n in list(param.keys())[::3]]
        df.rename(index=dict(zip(df.index, name)), inplace=True)

        df['差'] = df['data'] - df['inf']

        log_list = []
        for key in df["差"].keys():
            log_list.append(np.log(np.array(list(df.loc[f"{str(key)}", "差"]))))
        df["log"] = log_list

        return df

    df = transform_df(param)
    print(df.shape)

# %% [markdown]
# ## 进行线性拟合
# 
# $$\ln(\alpha_t-\alpha_{\inf})=-kt+\ln(\alpha_0-\alpha_{\inf})$$
# 
# 这里的$\alpha_t-\alpha_{\inf}$之前已经得到了,这里只要拟合$k,\alpha_0$
# 
# 并且关注时间

# %%
    data_points = len(df["data"][0])
    step_size = 30
    end_value = step_size * data_points
    time_array = np.arange(0, end_value, step_size)

# %% [markdown]
# ## 速率常数,半衰期
# 
# 得到了需要的**k**,作为速率常数.根据速率常数**k**又得到了半衰期$t_{\frac{1}{2}}$

# %%
# plt.rcParams['font.family'] = 'Arial Unicode MS'  # 将字体更改为具备广泛字符支持的字体
    def linear_fit(x, y, title):
    # 使用numpy的polyfit函数进行线性拟合
        coeffs = np.polyfit(x, y, 1)
    
    # 计算拟合直线上的点
        x_fit = np.linspace(x.min(), x.max(), 2)
        y_fit = np.polyval(coeffs, x_fit)
    
    # 绘制原始数据和拟合直线
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.plot(x_fit, y_fit, 'r--')
        ax.set_xlabel('时间(t/s)')
        ax.set_ylabel('ln(aplpha_t_alpha_inf)')
        ax.set_title(f'{title}')
        plt.savefig(folder_path/f'{title}_log.png')
    # plt.show()
        plt.close()
    
    # 返回拟合参数
        return coeffs

# 假设x和y分别代表自变量和因变量的数据
# coeffs = linear_fit(time_array, df["log"]['dma1'], title="dma1")
# print(coeffs)

    k_list = []
    alpha_0_list = []
    t_12 = []
    for key, value in df["log"].items():
    # print(key, value)
        key = str(key)
        coeffs = linear_fit(time_array, df["log"][f'{key}'], title=f"{key}")
    # print(coeffs)
        k_list.append(np.abs(coeffs[0]))
        alpha_0_list.append(coeffs[1])
        t_12.append(np.log(2)/np.abs(coeffs[0]))

    df["k"] = k_list
    df['alpha_0'] = alpha_0_list
    df['t_12'] = t_12
    df["temperature"] = tem_list

# %% [markdown]
# ## 考察活化能

# %%
    R = 8.314
# 每组大小
    group_size = 2

# 生成每一组的起始行和结束行的索引
    group_indices = [(i, i+group_size) for i in range(0, len(df), group_size)]

    E_a_list = []
# 对每一组进行操作
    for start, end in group_indices:
        group_df = df.iloc[start:end]
    # 进行操作
    # ...
    # print(group_df.iloc[0]["k"])
        k1 = group_df.iloc[0]["k"]
    # print(k1)
        k2 = group_df.iloc[1]["k"]
    # print(k2)
        T1 = group_df.iloc[0]["temperature"]
    # print(T1)
        T2 = group_df.iloc[1]["temperature"]
    # print(T2)
        E_a = np.log(k2/k1)*R*-1/(1/T2 - 1/T1)
    # print(E_a)
        E_a_list.append(E_a)
        E_a_list.append(E_a)
    # print("\n$$$$$$$$$$$$$\n")
# print(E_a_list)
    df["E_a"] = E_a_list


# %%
# 指定要写入的列
    columns_to_write = ['k', 'alpha_0',"inf","t_12","E_a"]

#指定保存目录
    save_path = folder_path/"output.xlsx"

# 创建 ExcelWriter 对象
    writer = pd.ExcelWriter(save_path, engine='xlsxwriter')

# 将特定列的 DataFrame 写入工作表
    df[columns_to_write].to_excel(writer, sheet_name='Sheet1', index=False)

# 保存并关闭 ExcelWriter 对象
    
    writer.close()


