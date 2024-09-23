# %%
import os
import codecs
from pathlib import Path
import numpy as np
def experimentSeven_process():
    def convert_encoding(folder_path, old_encoding, new_encoding):
    # 获取指定文件夹中的所有txt文件
        txt_files = [file for file in os.listdir(folder_path) if file.endswith(".txt")]

    # 遍历每个txt文件
        for file in txt_files:
        # 构建文件的完整路径
            file_path = os.path.join(folder_path, file)

        # 读取txt文件并进行编码格式的转换
            try:
                with codecs.open(file_path, "r", old_encoding) as f:
                    content = f.read()

                with codecs.open(file_path, "w", new_encoding) as f:
                    f.write(content)
            except UnicodeDecodeError:
                print(f"无法读取文件 {file}，请检查文件编码格式")
                continue

            print(f"文件 {file} 已转换为 {new_encoding} 编码")

# %%
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False		# 显示负号

    def save_plot_list(file_path,df_dic,save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print(f"{save_folder}已创建")
        else:
            print(f"{save_folder}已存在")
        def save_plot(df, save_folder,img_name):
        # 解析df中的时间和微压差列
            time = df['时间']
            pressure_diff = df['微压差']

        # 创建图形并绘制折线图
            plt.figure(figsize=(10, 6))   # 可根据需要调整图形大小
            plt.plot(time, pressure_diff)
            plt.xlabel('时间')
            plt.ylabel('微压差')
            plt.title(f"{img_name.split('png')[0]}"+'微压差随时间变化')
            plt.grid(True)

        # 保存图像
            save_path = os.path.join(save_folder, img_name)
            plt.savefig(save_path)

        # 显示图像
            # plt.show()

        img_files = [file.split(".txt")[0]+".png" for file in os.listdir(file_path) if file.endswith(".txt")]
        txt_files = [file for file in os.listdir(file_path) if file.endswith(".txt")]
        for txt_name, img_name in zip(txt_files,img_files):
            save_plot(df_dic[txt_name],save_folder,img_name)
    

# %%
    # import os
    import pandas as pd

    def read_txt_files(folder_path):
    # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    # 获取文件夹中的所有txt文件
        txt_files = [file for file in os.listdir(folder_path) if file.endswith(".txt")]
    
    # 创建一个空字典用于存储DataFrame
        dataframes = {}
    
    # 遍历每个txt文件
        for file in txt_files:
        # 构建文件的完整路径
            file_path = os.path.join(folder_path, file)
        
        # 读取txt文件并转换为DataFrame
            df = pd.read_csv(file_path, sep='\s+')
        
        # 将DataFrame存储到字典中
            dataframes[file] = df
    
        return dataframes

# %%
# 读取数据，绘制原始数据曲线
    # import os
    # import pandas as pd

# 指定文件夹路径
    current_path = Path(__file__).resolve().parent.parent
    folder_path = current_path/"upload"

    dataframes = read_txt_files(folder_path)

# %%
    save_plot_list(file_path = current_path/"upload",save_folder = current_path/"upload",df_dic=dataframes)

# %% [markdown]
# # 数据后处理
# 
# 首先得到每一个实验下的最大压强差

# %%
    def get_max_pa_list(file_path = current_path/"upload",dataframes=None):
        def get_max_pa(df):
        # 解析df中的时间和微压差列
            time = df['时间']
            pressure_diff = df['微压差']

            max_pa = -min(pressure_diff)
            return max_pa
    
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            print(f"{file_path}已创建")
        else:
            print(f"{file_path}已存在")

        name_files = [file.split('.txt')[0] for file in os.listdir(file_path) if file.endswith(".txt")]
        file_files = [file for file in os.listdir(file_path) if file.endswith(".txt")]
    # print(file_files)
        value_list = []
        for name in file_files:
        # print(name)
            value_list.append(get_max_pa(dataframes[name]))
    # df_data = {'experiment_name': name_files, 'max_detla_P': value_list}
    # 使用zip()函数将两个列表合并为一个元组列表，然后将其转换为字典
        data_dict = dict(zip(name_files, value_list))

    # 将字典转换为DataFrame
        df_data = pd.DataFrame(data_dict.items(), columns=['experiment_name', 'max_detla_P'])
        df_data = df_data.set_index('experiment_name')
        return np.array(value_list[:-1]), value_list[-1]
    

# print(get_max_pa(dataframes["1-0.4正丁醇.txt"]))


# %%
    max_p, water_P = get_max_pa_list(dataframes=dataframes)
    print(max_p)
    print(water_P)

# %%
    def get_K(max_p_W = None):
    # 计算K值
        rho_w = 1000    # 水的密度，以kg/m^3作为单位
        g = 10  # 重力加速度，以N/kg作为单位
        sigmma = 71.97 * 10 ** 3    # 水的表面张力，单位是N*m^{-1}
        delat_h_w = max_p_W/(rho_w * g)    # 高度差，以m作为单位
        K = sigmma / delat_h_w
    # print(delat_h_w)
    # print(K)
        return K
    K = get_K(max_p_W=water_P)
    print(K)

# %%
    def get_rho_butanol(V0 = 100,V_list=[0.4,0.8,1.2,1.6,2.0,2.4]):
        V_list = np.array(V_list)
    # print(V_list)
        rho_v = 92.6 # 丁醇的摩尔体积 cm^3/mol
        n_list = (V_list / rho_v) / (V_list + V0)   # 得到了溶液浓度，单位是mol*cm^{-3}
        n_list = np.array(n_list)

        return n_list*1000


    rho_list = get_rho_butanol()
    print(rho_list)
# print(sigmma_list)

# %%
    def get_sigmmas(delta_P_list=None, K=None,rho_list=None):
    # 计算不同浓度的溶液的的表面张力
        g = 10
        sigma = K * delta_P_list /(rho_list*g)    # 得到了表面张力
        return sigma / (10**7)
    sigma_list = get_sigmmas(delta_P_list=max_p, K=K,rho_list=rho_list)
    print(sigma_list)

# %%
    def plot_polyfit(x, y, degree, save_folder=current_path/"upload",img_name='polyfit.png'):
        '''
        输入：
        x: 作为横坐标的Numpy数组
        y: 作为纵坐标的Numpy数组
        degree: 多项式拟合的次数
    
        输出：
        绘制原始数据以及多项式拟合曲线的图形
        '''
        save_path = os.path.join(save_folder, img_name)
    # 多项式拟合
        coeffs = np.polyfit(x, y, degree)
    
        poly = np.poly1d(coeffs)
        print(poly)
        x_plot = np.linspace(x.min(), x.max(), 100)

    # 获取拟合后的函数的表达式
        fit_expr = ""
        for i,e in zip(coeffs,list(range(0,degree+1,1))[::-1]):
            fit_expr += "({}*x**{}) +".format(round(i, degree), e)
        fit_expr = fit_expr[:-1]
    # print(fit_expr)

    # 绘制图形
        plt.scatter(x, y, label='data')
        plt.ylabel(r'$\sigma(10^{-3}N/m)$')
        plt.xlabel(r'$c(mol\cdot L^{-1})$')
        plt.title("正丁醇水溶液的表面张力与浓度的关系图")
        plt.grid(True)
        plt.plot(x_plot, poly(x_plot), color='red', label='polyfit')
        plt.legend(loc='best')
    
        plt.savefig(save_path)
        # plt.show()

        return fit_expr
    
    

# %%
    fit_exp = plot_polyfit(x = np.array([0.0437,0.0874,0.1311,0.1748,0.2185,0.2622]), y = sigma_list, degree = 6)
    print(fit_exp)

# %%
    import sympy as sp
    # import numpy as np

    def calculate_derivative(fit_exp, x_vals):
    # 将 fit_exp 字符串转换为 SymPy 表达式
        x = sp.Symbol('x')
        expr = sp.sympify(fit_exp)

    # 对表达式进行微分
        dfdx = sp.diff(expr, x)
        print(dfdx)

    # 将表达式转换为可计算的函数
        dfdx_func = sp.lambdify(x, dfdx, 'numpy')

    # 使用 dfdx_func 对 x_vals 进行计算
        result = dfdx_func(x_vals)

        return result/(10**11)


# 调用函数进行计算
    d_sigma_dc = calculate_derivative(fit_exp, np.array([0.0437,0.0874,0.1311,0.1748,0.2185,0.2622]))

# 输出结果
    print(d_sigma_dc)


# %%
    def get_varGamma(d_sigma_dc=None, rho_list=None):
        R = 8.314
        T = 298
        varGamma = -(d_sigma_dc * rho_list) / (R * T) * 10**6
        return varGamma
    varGamma = get_varGamma(d_sigma_dc, rho_list=np.array([0.0437,0.0874,0.1311,0.1748,0.2185,0.2622]))
    print(varGamma)


# %%
    # import numpy as np
    # import matplotlib.pyplot as plt

    def get_c_varGamma(rho_list=None, varGamma=None, plot=True):
    # 计算 c/T
        c_T = rho_list / varGamma / 1000
        if plot:
        # 绘制数据散点图
            plt.scatter(rho_list, c_T, label='data')

        # 进行线性拟合
            fit = np.polyfit(rho_list, c_T, 1)
            fit_fn = np.poly1d(fit)
            print(fit)

        # 绘制拟合线
            plt.plot(rho_list, fit_fn(rho_list), 'r', label='linear fit')

            plt.ylabel(r'$c/T(\times 10^4m)$')
            plt.xlabel(r'$c(mol\cdot L^{-1})$')
            plt.title(r"c/T ~ c")
            plt.grid(True)

            plt.legend(loc='best')
            img_save_path = current_path/"upload\\线性拟合"
            plt.savefig(img_save_path)
            # plt.show()

        return fit

# 调用函数并绘制图形
    rho_list = np.array([0.0437, 0.0874, 0.1311, 0.1748, 0.2185, 0.2622])
# varGamma = 72.75
    c_varGamma = get_c_varGamma(rho_list, varGamma)
    print(c_varGamma)

# %%
    def get_S0(k=None):
        NA = 6.022140857 * 10**23
        return 1 / (k*NA)

    k = c_varGamma[0]
    print(k)
    print(get_S0(k))

# %%



