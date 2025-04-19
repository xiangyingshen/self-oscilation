import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# -------------------------------
# 模型代码定义
# -------------------------------
class SMPOscillator:
    def __init__(self, params):
        # 物理参数（SI单位制）
        self.m = params['m']          # 质量 [kg]
        self.c = params['c']          # 阻尼系数 [N·s/m]
        self.k0 = params['k0']        # 基础刚度 [N/m]
        self.alpha = params['alpha']  # 刚度温度系数 [N/(m·K)]
        self.gamma = params['gamma']  # 热膨胀系数 [N/K]
        self.mu = params['mu']        # 散热系数 [K/m]
        self.tau = params['tau']      # 热弛豫时间 [s]
        self.T0 = params['T0']        # 参考温度 [K]
        self.beta = params.get('beta', 0.1)  # 非线性阻尼系数 [N·s³/m³]
        self.A0 = 0.0  # 初始化振幅基准值

    def critical_temp(self, max_iter=50, tol=1e-4):
        """迭代法计算临界温度"""
        T_guess = self.T0 - self.k0/self.alpha + self.mu*self.gamma/self.alpha
        self.A0 = 0.01  # 初始振幅猜测
        
        for _ in range(max_iter):
            k_eff = self.k0 + self.alpha*(T_guess - self.T0)
            new_A0 = self.gamma*(T_guess - self.T0) / (k_eff + self.gamma*self.mu)
            
            # 收敛判断
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
        """非线性动力学模型"""
        A, v, T = y
        
        # 接触约束处理
        if A < 0:
            A = 0
            v = max(v, 0)  # 禁止向下运动
            
        # 热传导方程
        dT = (self.Tc - T - self.mu*A)/self.tau
        
        # 有效刚度计算
        k_eff = self.k0 + self.alpha*(T - self.T0)
        
        # 非线性阻尼项（立方阻尼）
        dv = (self.gamma*(T - self.T0) - k_eff*A - self.c*v - self.beta*v**3)/self.m
        
        return [v, dv, dT]

    def simulate(self, Tc, t_span, y0, dt=0.1):
        """运行动力学模拟"""
        self.Tc = Tc
        sol = solve_ivp(self.model, t_span, y0, method='Radau',
                        t_eval=np.arange(t_span[0], t_span[1], dt),
                        rtol=1e-6, atol=1e-8)
        return sol

# -------------------------------
# 参数扫描与数据输出
# -------------------------------

# 基准参数（除 T0 外，其余都扫描）
params_base = {
    'm': 1.0,        
    'c': 1,       
    'k0': 10.0,     
    'alpha': -1,  
    'gamma': 1,    
    'mu': 1,       
    'tau': 1,        
    'T0': 300,       
    'beta': 1      
}

# 待扫描的参数名称，共8个
param_names = ['m', 'c', 'k0', 'alpha', 'gamma', 'mu', 'tau', 'beta']

# 定义扫描倍数（示例：0.8倍、1.0倍、1.2倍）
scan_factors = [0.2, 1.0, 5.0]

# 时间范围：0~50秒，步长0.1秒
t_span = [0, 50]
dt = 0.1

# 确保输出文件保存目录存在
output_dir = "output_txt"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历每个参数及其扫描倍数，保存归一化后的时间-振幅数据到 txt 文件
for p_name in param_names:
    for factor in scan_factors:
        # 拷贝基准参数并修改当前参数
        params = params_base.copy()
        params[p_name] = params_base[p_name] * factor
        
        # 初始化模型
        smp = SMPOscillator(params)
        
        # 计算临界温度，设置超临界温度为 Tc_crit + 15
        Tc_crit = smp.critical_temp()
        Tc = Tc_crit + 15
        
        # 运行模拟
        sol = smp.simulate(Tc, t_span, y0=[0.1, 0.5, Tc], dt=dt)
        
        # 提取时间和振幅数据
        time_array = sol.t
        amplitude_array = sol.y[0]
        
        # 归一化振幅（除以该组数据的最大振幅值，若最大值为0则保持0）
        max_amp = np.max(amplitude_array)
        if max_amp != 0:
            amplitude_norm = amplitude_array / max_amp
        else:
            amplitude_norm = amplitude_array
        
        # 输出文件名，例如：output_txt/k0_factor1.2.txt
        out_filename = os.path.join(output_dir, f"{p_name}_factor{factor}.txt")
        data_to_save = np.column_stack((time_array, amplitude_norm))
        np.savetxt(out_filename, data_to_save, 
                   header=f"Time(s)  Normalized Amplitude\nParam: {p_name}, factor: {factor:.2f}, Tc: {Tc:.2f}",
                   fmt="%.5f")
        print(f"已保存文件: {out_filename}")

# -------------------------------
# 根据所有输出的txt文件绘图
# -------------------------------

# 为每个参数绘制一个子图，共8个子图（2行4列）
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
axs = axs.flatten()

# 遍历每个参数
for idx, p_name in enumerate(param_names):
    ax = axs[idx]
    # 对于当前参数，依次加载不同扫描因子的文件并绘制曲线
    for factor in scan_factors:
        file_path = os.path.join(output_dir, f"{p_name}_factor{factor}.txt")
        try:
            data = np.loadtxt(file_path)
            t_data = data[:, 0]
            A_data = data[:, 1]
            ax.plot(t_data, A_data, label=f"factor {factor}")
        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
    ax.set_title(f"{p_name}", fontsize=16)   # 设置标题字体大小
    ax.set_xlabel("Time (s)", fontsize=16)               # 设置x轴标签字体大小
    ax.set_ylabel("Normalized Amplitude", fontsize=16)   # 设置y轴标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=16)  # 调整刻度标签字体大小
    ax.legend(loc="lower left", fontsize=12)             # 设置图例字体大小
    
plt.tight_layout()
plt.savefig("all_data_plots.pdf", dpi=120)
plt.show()
