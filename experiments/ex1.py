import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn


def ex1_1():
    t = np.arange(-10, 10, 0.01)
    x = np.cos(2 * np.pi * t) * (np.heaviside(t, 1) - np.heaviside(t - 3, 1))
    plt.plot(t, x)
    plt.xlabel("t[s]")
    plt.ylabel("x(t)")
    plt.title("Figure 1.1: x(t)")
    plt.show()


def ex1_2():
    t = np.arange(-10, 10, 0.01)
    h = np.exp(-t) * np.heaviside(t, 1) - np.exp(-3 * t) * np.heaviside(t, 1)
    plt.plot(t, h)
    plt.xlabel("t[s]")
    plt.ylabel("h(t)")
    plt.title("Figure 1.2: h(t)")
    plt.show()


def ex1_3():
    t = np.arange(-2, 2, 0.01)
    h = np.heaviside(t + 0.5, 1) - np.heaviside(t - 0.5, 1)
    plt.plot(t, h)
    plt.xlabel("t[s]")
    plt.ylabel("y(t)")
    plt.title("Figure 1.3: y(t)=G_1(t)")
    plt.show()


def ex2_1():
    t = np.arange(-2, 2, 0.01)
    g = np.heaviside(t + 0.5, 1) - np.heaviside(t - 0.5, 1)
    # y(2t)
    y1 = np.heaviside(2 * t + 0.5, 1) - np.heaviside(2 * t - 0.5, 1)
    # y(t/2)
    y2 = np.heaviside(t / 2 + 0.5, 1) - np.heaviside(t / 2 - 0.5, 1)
    # y(1-2t)
    y3 = np.heaviside((1 - 2 * t) + 0.5, 1) - np.heaviside((1 - 2 * t) - 0.5, 1)
    # 第一幅图像
    plt.subplot(3, 1, 1)
    plt.plot(t, g, t, y1)
    plt.xlabel("t[s]")
    plt.ylabel("y(t)")
    plt.legend(['y(t)', 'y(2t)'])
    plt.title("Figure 2.1: y(t), y(2t), y(t/2) and y(1-2t)")
    # 第二幅图像
    plt.subplot(3, 1, 2)
    plt.plot(t, g, t, y2)
    plt.xlabel("t[s]")
    plt.ylabel("y(t)")
    plt.legend(['y(t)', 'y(t/2)'])
    # 第三幅图像
    plt.subplot(3, 1, 3)
    plt.plot(t, g, t, y3)
    plt.xlabel("t[s]")
    plt.ylabel("y(t)")
    plt.legend(['y(t)', 'y(1-2t)'])
    # 绘图
    plt.show()


def ex2_2():
    t = np.arange(-20, 20, 0.01)
    y = np.sin(t) + np.sin(np.pi * t / 4)
    plt.plot(t, y)
    plt.xlabel("t[s]")
    plt.ylabel("y(t)")
    plt.title("Figure 2.2: y(t)=sin(t)+sin(pi*t/4)")
    plt.grid()
    plt.show()


def ex2_3():
    t = np.arange(-10, 10, 0.01)
    y = np.sin(np.pi * t) + np.sin(2 * np.pi * t)
    plt.plot(t, y)
    plt.xlabel("t[s]")
    plt.ylabel("y(t)")
    plt.title("Figure 2.3: y(t)=sin(pi*t)+sin(2*pi*t)")
    plt.grid()
    plt.show()


def ex3():
    t = np.arange(0, 20, 0.01)
    y1 = np.exp(-3 * t) * np.heaviside(t, 0)
    y2 = np.exp(-t) * np.heaviside(t, 0)
    # 将y1和y2进行卷积。由于计算是离散的点，卷积后需要乘以步长。
    y = sgn.convolve(y1, y2) * 0.01
    # 理论值函数
    t_theory = np.arange(0, 40, 0.1)
    y_theory = -0.5 * np.exp(-3 * t_theory) + 0.5 * np.exp(-t_theory)
    # 绘制仿真函数图像
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, 39.99, 0.01), y, label='simulate')
    plt.ylabel("y(t)")
    plt.title("Figure 3: Convolution")
    plt.legend()
    # 绘制理论值对比图像
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, 39.99, 0.01), y, label='simulate')
    plt.plot(t_theory, y_theory, label='theory')
    plt.xlabel("t[s]")
    plt.ylabel("y(t)")
    plt.legend()
    plt.show()


def ex4():
    system = sgn.lti([1], [1, 3, 2])
    t = np.arange(0, 10, 0.01)
    f = np.exp(-2 * t) * np.heaviside(t, 0)
    y= np.exp(-t)-(1+t) * np.exp(-2*t)
    plt.subplot(2, 1, 1)
    plt.plot(t,y)
    plt.ylabel("r_ZS")
    plt.title('Figure 4: Zero-State Response')
    tout, yout, xout = sgn.lsim(system, f, t)
    plt.subplot(2, 1, 2)
    plt.plot(t,y,label='theory')
    plt.plot(tout, yout, label='simulate')
    plt.xlabel("t[s]")
    plt.ylabel("r_ZS")
    plt.legend()
    plt.show()


def ex5():
    x = np.mgrid[-10:10:0.02]
    y1 = 0
    for i in range(0, 3, 1):
        b = (-1) ** (i) * np.cos((2 * i + 1) * x) / (2 * i + 1)
        y1 = b + y1  # 这种求和的方法是从C语言移植过来的
    plt.plot(x, y1, 'orange', linewidth=0.6)
    plt.title('cos_square')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def ex5_2():
    t = np.arange(-10, 10, 0.01)
    y = 0.5 + sgn.square(np.pi * (t + 0.5)) / 2
    y1 = 0.5 + 2 * np.cos(np.pi * t) / np.pi - 2 * np.cos(3 * np.pi * t) / 3 / np.pi
    y2 = y1 + 2 * np.cos(5 * np.pi * t) / 5 / np.pi
    plt.subplot(2, 1, 1)
    plt.plot(t, y, t, y1)
    plt.ylabel("f(t)")
    plt.legend(['origin', 'first 3 sums'])
    plt.title("Figure 5.2: First 3 and 5 Sums Simulate")
    plt.subplot(2, 1, 2)
    plt.plot(t, y, t, y2)
    plt.xlabel("t[s]")
    plt.ylabel("f(t)")
    plt.legend(['origin', 'first 5 sums'])
    plt.show()


def ex5_3():
    t = np.arange(-10, 10, 0.01)
    y = 0.5 + sgn.square(np.pi * (t + 0.5)) / 2
    y1 = 0.5 + np.exp(1j * np.pi * t) / np.pi + np.exp(-1j * np.pi * t) / np.pi - np.exp(
        3j * np.pi * t) / np.pi / 3 - np.exp(-3j * np.pi * t) / np.pi / 3
    y2 = y1 + np.exp(5j * np.pi * t) / np.pi / 5 + np.exp(-5j * np.pi * t) / np.pi / 5
    plt.subplot(2, 1, 1)
    plt.plot(t, y, t, y1.real)
    plt.ylabel("f(t)")
    plt.legend(['origin', 'first 3 sums'])
    plt.title("Figure 5.3: First 3 and 5 Sums Simulate")
    plt.subplot(2, 1, 2)
    plt.plot(t, y, t, y2.real)
    plt.xlabel("t[s]")
    plt.ylabel("f(t)")
    plt.legend(['origin', 'first 5 sums'])
    plt.show()


def ex5_origin():
    t = np.linspace(-5, 5, 500, endpoint=False)
    plt.plot(t, 0.5 + sgn.square(np.pi * (t + 0.5)) / 2)
    plt.grid()
    plt.xlabel("t[s]")
    plt.ylabel("y(t)")
    plt.title('Figure 5: Rectangular Pulse Sequence')
    plt.show()


if __name__ == "__main__":
    ex1_1()
    ex1_2()
    ex1_3()
    ex2_1()
    ex2_2()
    ex2_3()
    ex3()
    ex4()
    ex5()
    ex5_2()
    ex5_3()
    ex5_origin()
