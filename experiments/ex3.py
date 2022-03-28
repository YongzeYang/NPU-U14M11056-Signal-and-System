import numpy as np
import matplotlib.pyplot as plt


def h(w):
    """
    原函数的傅里叶变换
    :param w: omega
    :return: omega变换后得傅里叶变换
    """
    return np.sin(np.pi * w) / (w * (1 - w ** 2))


def f0(t):
    """
    返回参数为t的原函数
    :param t: 参数取值
    :return: 参数为t的原函数
    """
    return 0.5 * (1 + np.cos(t)) * (np.heaviside(t + np.pi, 0) - np.heaviside(t - np.pi, 0))


def ex1_1():
    """
    原函数的时域、幅度谱分析
    """
    t = np.arange(-np.pi, np.pi, 0.01)  # 自变量取值范围，间隔0.01
    f = 0.5 * (1 + np.cos(t)) * (np.heaviside(t + np.pi, 0) - np.heaviside(t - np.pi, 0))
    # 作图，时域图
    plt.plot(t, f)
    plt.grid()
    plt.xlabel("time[s]")
    plt.ylabel("f(t)")
    plt.title("Figure 1.1.1: Origin Function Time Domain Diagram")
    plt.show()

    # 原函数幅度谱
    w = np.linspace(-20, 20, 1000)
    h = np.sin(np.pi * w) / (w * (1 - w ** 2))
    plt.plot(w, h)
    plt.grid()
    plt.xlabel("w")
    plt.ylabel("F(jw)")
    plt.title("Figure 1.1.2: Origin Function Amplitude Spectrum")
    plt.show()


def ex1_2():
    """
    0.5s抽样的时域、幅度谱分析
    """
    t = np.arange(-np.pi, np.pi, 0.5)  # 自变量取值范围，取样间隔0.5s
    f = 0.5 * (1 + np.cos(t)) * (np.heaviside(t + np.pi, 0) - np.heaviside(t - np.pi, 0))
    # 作图，时域图
    plt.stem(t, f)
    plt.grid()
    plt.xlabel("time[s]")
    plt.ylabel("f")
    plt.title("Figure 1.2.1: Sampling Function(0.5s) Time Domain Diagram")
    plt.show()

    # 0.5s抽样函数幅度谱
    w = np.linspace(-20, 20, 1000)
    F = h(w)  # F为0.5s抽样函数的频谱，初始赋值为原函数的傅里叶变换
    for i in range(1, 3):  # omega从1递增到3叠加
        F += h(w + i * 4 * np.pi)
        F += h(w - i * 4 * np.pi)
    plt.plot(w, F * 2)
    plt.grid()
    plt.xlabel("w")
    plt.ylabel("F(jw)")
    plt.title("Figure 1.2.2: Sampling Function(0.5s) Amplitude Spectrum")
    plt.show()


def ex1_3():
    """
    1s抽样的时域、幅度谱分析
    """
    t = np.arange(-np.pi, np.pi, 1.0)  # 自变量取值范围，取样间隔1s
    f = 0.5 * (1 + np.cos(t)) * (np.heaviside(t + np.pi, 0) - np.heaviside(t - np.pi, 0))
    # 作图，时域图
    plt.stem(t, f)
    plt.grid()
    plt.xlabel("time[s]")
    plt.ylabel("f")
    plt.title("Figure 1.3.1: Sampling Function(1s) Time Domain Diagram")
    plt.show()

    # 1s抽样函数幅度谱
    w = np.linspace(-20, 20, 1000)
    F = h(w)  # F为1s抽样函数的频谱，初始赋值为原函数的傅里叶变换
    for i in range(1, 5):  # omega从1递增到3叠加
        F += h(w + i * 2 * np.pi)
        F += h(w - i * 2 * np.pi)
    plt.plot(w, F)
    plt.grid()
    plt.xlabel("w")
    plt.ylabel("F(jw)")
    plt.title("Figure 1.3.2: Sampling Function(1s) Amplitude Spectrum")
    plt.show()

    # 截取前三个周期
    w = np.linspace(-10, 10, 1000)
    F = h(w)  # F为1s抽样函数的频谱，初始赋值为原函数的傅里叶变换
    for i in range(1, 5):  # omega从1递增到3叠加
        F += h(w + i * 2 * np.pi)
        F += h(w - i * 2 * np.pi)
    plt.plot(w, F)
    plt.grid()
    plt.xlabel("w")
    plt.ylabel("F(jw)")
    plt.title("Figure 1.3.3: Sampling Function(1s) Amplitude Spectrum")
    plt.show()


def ex1_4():
    """
    2s抽样的时域、幅度谱分析
    """
    t = np.arange(-np.pi, np.pi, 2.0)  # 自变量取值范围，取样间隔2s
    f = 0.5 * (1 + np.cos(t)) * (np.heaviside(t + np.pi, 0) - np.heaviside(t - np.pi, 0))
    # 作图，时域图
    plt.stem(t, f)
    plt.grid()
    plt.xlabel("time[s]")
    plt.ylabel("f")
    plt.title("Figure 1.4.1: Sampling Function(2s) Time Domain Diagram")
    plt.show()

    # 2s抽样函数幅度谱
    w = np.linspace(-20, 20, 1000)
    F = h(w)  # F为2s抽样函数的频谱，初始赋值为原函数的傅里叶变换
    for i in range(1, 10):  # omega从1递增到3叠加
        F += h(w + i * np.pi)
        F += h(w - i * np.pi)
    plt.plot(w, F / 2)
    plt.grid()
    plt.xlabel("w")
    plt.ylabel("F(jw)")
    plt.title("Figure 1.4.2: Sampling Function(2s) Amplitude Spectrum")
    plt.show()

    # 截取前三个周期
    w = np.linspace(-5, 5, 1000)
    F = h(w)  # F为2s抽样函数的频谱，初始赋值为原函数的傅里叶变换
    for i in range(1, 10):  # omega从1递增到3叠加
        F += h(w + i * np.pi)
        F += h(w - i * np.pi)
    plt.plot(w, F / 2)
    plt.grid()
    plt.xlabel("w")
    plt.ylabel("F(jw)")
    plt.title("Figure 1.4.2: Sampling Function(2s) Amplitude Spectrum")
    plt.show()


def ex2_1():
    """
    T=1s时的信号重建与误差分析
    """
    t1 = np.arange(-np.pi, np.pi, 1)  # 自变量取值范围，间隔1
    F1 = f0(0) * np.sinc(2.4 * t1)  # 重建信号初始值
    for i in range(1, 10):  # K从1递增到10叠加
        F1 += f0(i * 1) * np.sinc(2.4 * (t1 - i * 1))
        F1 += f0(i * 1) * np.sinc(2.4 * (t1 + i * 1))

    plt.stem(t1, F1 * 2.4 * 1 / np.pi, label='Reconstruction', markerfmt='rD')  # 重建函数
    plt.stem(t1, f0(t1), label='Origin', markerfmt='gD')  # 原函数
    plt.plot(t1, abs(F1 * 2.4 * 1 / np.pi - f0(t1)), label='Difference', color='orange')  # 差值的绝对值
    plt.grid()
    plt.xlabel("time[s]")
    plt.ylabel("f")
    plt.title("Figure 2.1: Sampling Function(1s) Reconstruction")
    plt.legend()
    plt.show()


def ex2_2():
    t2 = np.arange(-np.pi, np.pi, 2)  # 自变量取值范围，间隔2
    F2 = f0(0) * np.sinc(2.4 * t2)  # 重建信号初始值

    for i in range(1, 10):  # K从1递增到10叠加
        F2 += f0(i * 2) * np.sinc(2.4 * (t2 - i * 2))
        F2 += f0(i * 2) * np.sinc(2.4 * (t2 + i * 2))

    plt.stem(t2, F2 * 2.4 * 2 / np.pi, label='Reconstruction', markerfmt='rD')
    plt.stem(t2, f0(t2), label='Origin', markerfmt='gD')
    plt.plot(t2, abs(F2 * 2.4 * 2 / np.pi - f0(t2)), label='Difference', color='orange')
    plt.grid()
    plt.xlabel("time[s]")
    plt.ylabel("f")
    plt.title("Figure 2.2: Sampling Function(2s) Reconstruction")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ex1_1()
    ex1_2()
    ex1_3()
    ex1_4()
    ex2_1()
    ex2_2()
