# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 08:58:11 2025

@author: win10
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# ========== 1. 原模型及函数 ==========

class SMPOscillator:
    def __init__(self, params):
        self.m = params['m']
        self.c = params['c']
        self.k0 = params['k0']
        self.alpha = params['alpha']
        self.gamma = params['gamma']
        self.mu = params['mu']
        self.tau = params['tau']
        self.T0 = params['T0']
        self.beta = params.get('beta', 0.1)
        self.A0 = 0.0

    def critical_temp(self, max_iter=50, tol=1e-4):
        """迭代法计算临界温度"""
        T_guess = self.T0 - self.k0/self.alpha + self.mu*self.gamma/self.alpha
        self.A0 = 0.01
        for _ in range(max_iter):
            k_eff = self.k0 + self.alpha*(T_guess - self.T0)
            new_A0 = self.gamma*(T_guess - self.T0) / (k_eff + self.gamma*self.mu)
            if abs(new_A0 - self.A0) < tol:
                break
            self.A0 = new_A0
            numerator = self.mu*(self.gamma - self.alpha*self.A0) - self.c/self.tau
            denominator = self.alpha*(1 + self.c/(self.m*self.tau))
            T_guess = self.T0 + numerator/denominator
        return T_guess

    def period(self, Tc):
        """计算线性化周期估计值"""
        k_eff = self.k0 + self.alpha*(Tc - self.T0)
        omega = np.sqrt(k_eff/self.m + self.c/(self.m*self.tau))
        return 2*np.pi/omega if omega != 0 else np.inf

    def model(self, t, y):
        A, v, T = y
        if A < 0:
            A = 0
            v = max(v, 0)
        dT = (self.Tc - T - self.mu*A)/self.tau
        k_eff = self.k0 + self.alpha*(T - self.T0)
        dv = (self.gamma*(T - self.T0) - k_eff*A - self.c*v - self.beta*v**3)/self.m
        return [v, dv, dT]

    def simulate(self, Tc, t_span, y0, dt=0.1):
        self.Tc = Tc
        sol = solve_ivp(self.model, t_span, y0, method='Radau',
                        t_eval=np.arange(t_span[0], t_span[1], dt),
                        rtol=1e-6, atol=1e-8)
        return sol

def create_3d_animation(t, A, filename='oscillation.gif'):
    """生成受限振动的3D动画（略，与本题无关，可保持原样）"""
    pass

# ========== 2. 扫描逻辑及归一化处理 ==========

def measure_amplitude(smp, Tdrive, sim_time=50.0, dt=0.1):
    """
    在给定驱动温度 Tdrive 条件下，模拟系统一段时间，
    并返回稳态后的振荡幅度（取后20%时间内的 max(A)-min(A)）。
    """
    sol = smp.simulate(Tdrive, [0, sim_time], y0=[0.1, 0.5, Tdrive], dt=dt)
    A_vals = sol.y[0]
    cutoff = int(0.8 * len(A_vals))
    A_steady = A_vals[cutoff:]
    amp = np.max(A_steady) - np.min(A_steady)
    return amp

def scan_parameter(param_name, base_params, scales):
    """
    扫描某个参数 param_name 在给定 scales 范围（相对于 base_params[param_name] 的倍数），
    返回 (scales, T_C, T_osc, amplitude) 四个数组。
    """
    T_C_list = []
    T_osc_list = []
    amp_list = []
    base_val = base_params[param_name]

    for s in scales:
        new_params = base_params.copy()
        new_params[param_name] = base_val * s
        smp = SMPOscillator(new_params)
        Tc_val = smp.critical_temp()
        Tosc_val = smp.period(Tc_val)
        # 超临界驱动温度设定：Tc + 10K
        drive_temp = Tc_val + 10
        amp_val = measure_amplitude(smp, drive_temp, sim_time=50.0, dt=0.1)
        
        T_C_list.append(Tc_val)
        T_osc_list.append(Tosc_val)
        amp_list.append(amp_val)

    return scales, np.array(T_C_list), np.array(T_osc_list), np.array(amp_list)

def save_normalized_data(filename, metric_data_dict):
    """
    将指标数据保存到txt文件中。
    metric_data_dict：字典，键为参数名称，值为 (scales, data_array)
    保存格式：第一列为扫描因子，其后各列为归一化后的数据（归一化为各自列除以该列的最大值）。
    """
    # 假设所有参数扫描的 scales 数组相同
    params = list(metric_data_dict.keys())
    scales = metric_data_dict[params[0]][0]
    header = "ScalingFactor\t" + "\t".join(params) + "\n"
    lines = [header]
    # 对每个参数的数据做归一化处理
    norm_data = {}
    for p in params:
        s, data_arr = metric_data_dict[p]
        max_val = np.max(data_arr)
        norm_data[p] = data_arr / max_val if max_val != 0 else data_arr

    for i in range(len(scales)):
        line = f"{scales[i]:.4f}"
        for p in params:
            value = norm_data[p][i]
            line += f"\t{value:.4f}"
        line += "\n"
        lines.append(line)
    with open(filename, "w") as f:
        f.writelines(lines)

def get_top_parameters(metric_data_dict, top_n=3):
    """
    根据指标数据，计算每个参数的影响力（最大值-最小值），
    返回影响最大的 top_n 个参数列表。
    """
    influences = {}
    for p, (scales, data_arr) in metric_data_dict.items():
        influences[p] = np.max(data_arr) - np.min(data_arr)
    sorted_params = sorted(influences, key=lambda p: influences[p], reverse=True)
    return sorted_params[:top_n]

# ========== 3. 主程序：扫描、归一化输出数据及绘图 ==========

if __name__ == "__main__":

    # 3.1 定义基准参数（除 T0 外所有参数均待扫描）
    base_params = {
        'm': 1.0,
        'c': 1.0,
        'k0': 10.0,
        'alpha': -0.1,
        'gamma': 1,
        'mu': 1,
        'tau': 1,
        'T0': 300,
        'beta': 1
    }

    # 要扫描的参数列表（除 T0 之外），注意不要重复
    params_list = ['m', 'c', 'beta', 'k0', 'alpha', 'gamma', 'mu', 'tau']
    # 扫描范围：例如从0.2到5倍，共40个点
    scan_scales = np.linspace(0.5, 3, 30)

    # 用于保存各参数扫描的三个指标数据
    Tc_data_dict = {}
    Tosc_data_dict = {}
    Amp_data_dict = {}

    for p in params_list:
        s, Tc_arr, Tosc_arr, amp_arr = scan_parameter(p, base_params, scan_scales)
        Tc_data_dict[p] = (s, Tc_arr)
        Tosc_data_dict[p] = (s, Tosc_arr)
        Amp_data_dict[p] = (s, amp_arr)

    # 3.2 选取对各指标影响最大的前三个参数
    top3_Tc = get_top_parameters(Tc_data_dict, top_n=3)
    top3_Tosc = get_top_parameters(Tosc_data_dict, top_n=3)
    top3_Amp = get_top_parameters(Amp_data_dict, top_n=3)

    # 保存数据到txt文件（分别针对 T_C, T_osc, 振幅）
    Tc_top_dict = {p: Tc_data_dict[p] for p in top3_Tc}
    Tosc_top_dict = {p: Tosc_data_dict[p] for p in top3_Tosc}
    Amp_top_dict = {p: Amp_data_dict[p] for p in top3_Amp}

    save_normalized_data("Tc_data.txt", Tc_top_dict)
    save_normalized_data("Tosc_data.txt", Tosc_top_dict)
    save_normalized_data("Amp_data.txt", Amp_top_dict)
    print("归一化数据已输出到 Tc_data.txt, Tosc_data.txt, Amp_data.txt。")

    # 3.3 根据txt文件绘制pdf矢量图
    # 这里直接使用内存中的数据绘图
    # 为避免重合，定义不同的颜色和标记形状
    markers = ['o', 's', '^']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    # 绘制归一化的 T_C（使用 top3_Tc 的参数）
    for i, p in enumerate(top3_Tc):
        scales, data = Tc_data_dict[p]
        max_val = np.max(data)
        norm_data = data / max_val if max_val != 0 else data
        axs[0].plot(scales, norm_data, marker=markers[i], color=colors[i], linestyle='-', label=p)
    axs[0].set_ylabel('Normalized $T_C$')
    axs[0].set_title('Normalized Critical Temperature vs Scaling Factor')
    axs[0].legend()

    # 绘制归一化的 T_osc（使用 top3_Tosc 的参数）
    for i, p in enumerate(top3_Tosc):
        scales, data = Tosc_data_dict[p]
        max_val = np.max(data)
        norm_data = data / max_val if max_val != 0 else data
        axs[1].plot(scales, norm_data, marker=markers[i], color=colors[i], linestyle='-', label=p)
    axs[1].set_ylabel('Normalized $T_{osc}$')
    axs[1].set_title('Normalized Oscillation Period vs Scaling Factor')
    axs[1].legend()

    # 绘制归一化的 振幅（使用 top3_Amp 的参数）
    for i, p in enumerate(top3_Amp):
        scales, data = Amp_data_dict[p]
        max_val = np.max(data)
        norm_data = data / max_val if max_val != 0 else data
        axs[2].plot(scales, norm_data, marker=markers[i], color=colors[i], linestyle='-', label=p)
    axs[2].set_xlabel('Scaling Factor')
    axs[2].set_ylabel('Normalized Amplitude')
    axs[2].set_title('Normalized Amplitude vs Scaling Factor')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig("normalized_metrics.pdf")
    print("归一化指标变化趋势图已保存为 normalized_metrics.pdf。")
    plt.show()
