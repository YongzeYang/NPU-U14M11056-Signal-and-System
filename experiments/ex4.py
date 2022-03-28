import sympy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqs, tf2zpk, lti, impulse2

sympy.init_printing()
t, s = sympy.symbols('t, s')


def L(f):
    '''
    对函数求拉普拉斯变换
    :param f: 函数
    :return: 函数的拉普拉斯变换
    '''
    return sympy.laplace_transform(f, t, s, noconds=True)


def invL(F):
    '''
    对函数求拉普拉斯逆变换
    :param F: 函数
    :return: 函数的拉普拉斯逆变换
    '''
    return sympy.inverse_laplace_transform(F, s, t)


def ex1_1():
    fs = [sympy.exp(-2 * t) * sympy.Heaviside(t, 1),  # f1
          sympy.sin(2 * t) * sympy.Heaviside(t, 1),  # f2
          sympy.DiracDelta(t) + sympy.exp(2 * t) * sympy.Heaviside(t, 1) - 4 * sympy.exp(-t) * sympy.Heaviside(t, 1) / 3
          # f3
          ]
    for f in fs:  # 对于函数集中的每一个函数
        print("result: ", L(f))  # 输出拉普拉斯变换
        print()


def ex1_2():
    fs = [(4 * s + 5) / (s ** 2 + 5 * s + 6),  # F1
          (3 * s) / ((s + 4) * (s + 2)),  # F2
          (s + 5) / (s * (s ** 2 + 2 * s + 5))  # F3
          ]
    for f in fs:
        print("apart:  ", f.apart(s))  # 将函数化简
        print("result: ", invL(f.apart(s)))  # 反拉普拉斯变换
        print()


def ex1_3():
    f = (s ** 2 + 4 * s + 5) / (s ** 2 + 5 * s + 6)  # F(s)
    print(f.apart(s))  # 展开并输出到控制台


def ex1_4():
    z, p, _ = tf2zpk([1, -1], [1, 3, 2])
    # 绘制零点与极值点
    plt.scatter(z.real, np.imag(z), label = 'Zero Point')
    plt.scatter(p.real, np.imag(p), marker='x', label = 'Pole')
    # 作图
    plt.title("Figure 1.1: Pole-Zero Plot")
    plt.xlabel('σ')
    plt.ylabel('jw')
    plt.legend()
    plt.grid()
    plt.show()

    # 单位冲击响应
    t1, y1 = impulse2(lti([1, -1], [1, 3, 2]))
    plt.plot(t1, y1)
    plt.title("Figure 1.2: Impulse Response")
    plt.xlabel('t')
    plt.ylabel('h(t)')
    plt.grid()
    plt.show()

    # 频响
    w, H = freqs([1, -1], [1, 3, 2], worN=np.linspace(-15, 15, 500))

    # 相位谱
    plt.plot(w, np.angle(H))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("phase")
    plt.title("Figure 1.3: Phase Spectrum")
    plt.show()

    # 幅度谱
    plt.plot(w, abs(H))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.title("Figure 1.4: Amplitude Spectrum")
    plt.show()


if __name__ == "__main__":
    # ex1_1()
    # ex1_2()
    ex1_4()
