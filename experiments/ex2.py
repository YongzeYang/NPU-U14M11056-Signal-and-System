import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn


def ex1_1():
    t = np.arange(-10, 10, 0.01)
    f = np.exp(-t) * np.heaviside(t, 1)
    # 作图，相位图
    plt.plot(t, f)
    plt.grid()
    plt.xlabel("time[s]")
    plt.ylabel("value")
    plt.title("Figure 1.1.1: Waveform")
    plt.show()

    w = np.linspace(-10, 10, 1000)
    H = 1 / (1j * w + 1)

    # 相位谱
    plt.plot(w, np.angle(H))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("phase")
    plt.title("Figure 1.1.2: Phase Spectrum")
    plt.show()

    # 幅度谱
    plt.plot(w, abs(H))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.title("Figure 1.1.3: Amplitude Spectrum")
    plt.show()


def ex1_2():
    # 时间取值范围
    t = np.arange(-10, 10, 0.01)
    # 函数表达式
    f = np.exp(-2 * t) * np.heaviside(2 * t, 0)
    # 作图，相位图
    plt.plot(t, f)
    plt.grid()
    plt.xlabel("time[s]")
    plt.ylabel("value")
    plt.title("Figure 1.2.1: Waveform")
    plt.show()

    w = np.linspace(-10, 10, 1000)
    H = 1 / (1j * w + 2)

    # 相位谱
    plt.plot(w, np.angle(H))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("phase")
    plt.title("Figure 1.2.2: Phase Spectrum")
    plt.show()

    # 幅度谱
    plt.plot(w, abs(H))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.title("Figure 1.2.3: Amplitude Spectrum")
    plt.show()


def ex1_3():
    t = np.arange(-10, 10, 0.01)
    f = np.exp(-(t - 2)) * np.heaviside(t - 2, 0)
    # 作图，相位图
    plt.plot(t, f)
    plt.grid()
    plt.xlabel("time[s]")
    plt.ylabel("value")
    plt.title("Figure 1.3.1: Waveform")
    plt.show()

    w = np.linspace(-10, 10, 1000)
    H = np.exp(-2j * w) / (1j * w + 1)

    # 相位谱
    plt.plot(w, np.angle(H))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("phase")
    plt.title("Figure 1.3.2: Phase Spectrum")
    plt.show()

    # 幅度谱
    plt.plot(w, abs(H))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.title("Figure 1.3.3: Amplitude Spectrum")
    plt.show()


def ex1_4():
    t = np.arange(-10, 10, 0.01)
    f = np.exp(-t) * np.cos(2 * t) * np.heaviside(t, 0)
    # 作图，相位图
    plt.plot(t, f)
    plt.grid()
    plt.xlabel("time[s]")
    plt.ylabel("value")
    plt.title("Figure 1.4.1: Waveform")
    plt.show()

    w = np.linspace(-10, 10, 1000)
    H = (1 / (1 + 1j * (w - 2)) + 1 / (1 + 1j * (w + 2))) / 2

    # 相位谱
    plt.plot(w, np.angle(H))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("phase")
    plt.title("Figure 1.4.2: Phase Spectrum")
    plt.show()

    # 幅度谱
    plt.plot(w, abs(H))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.title("Figure 1.4.3: Amplitude Spectrum")
    plt.show()


def ex1_5():
    t = np.arange(-10, 10, 0.01)
    f = (np.exp(-t) - np.exp(-2 * t)) * np.heaviside(t, 0)
    # 作图，相位图
    plt.plot(t, f)
    plt.grid()
    plt.xlabel("time[s]")
    plt.ylabel("value")
    plt.title("Figure 1.5.1: Waveform")
    plt.show()

    w = np.linspace(-10, 10, 1000)
    H1 = 1 / (1j * w + 1) - 1 / (1j * w + 2)
    H2 = 1 / ((1j * w + 1) * (1j * w + 2))

    # 相位谱
    plt.plot(w, np.angle(H1))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("phase")
    plt.title("Figure 1.5.2: Phase Spectrum")
    plt.show()

    # 幅度谱
    plt.plot(w, abs(H1))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.title("Figure 1.5.3: Amplitude Spectrum")
    plt.show()

    # F1*F2相位谱
    plt.plot(w, np.angle(H2))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("phase")
    plt.title("Figure 1.5.4: Phase Spectrum (F1·F2)")
    plt.show()

    # F1*F2幅度谱
    plt.plot(w, abs(H2))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.title("Figure 1.5.5: Amplitude Spectrum (F1·F2)")
    plt.show()


def ex2_1():
    # 使用freqs函数
    w, H = sgn.freqs([1], [1, 2, 2, 1], worN=np.linspace(-10, 10, 200))

    # 相位谱
    plt.plot(w, np.angle(H))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("phase")
    plt.title("Figure 2.1.1: Phase Spectrum")
    plt.show()

    # 幅度谱
    plt.plot(w, abs(H))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.title("Figure 2.1.2: Amplitude Spectrum")
    plt.show()


def ex2_2():
    w, H = sgn.freqs([40], [1, 10 * np.sqrt(3)], worN=np.linspace(-10, 10, 200))
    # 作图，相位图
    plt.plot(w, H)
    plt.grid()
    plt.xlabel("time[s]")
    plt.ylabel("value")
    plt.title("Figure 2.2.1: Waveform")
    plt.show()

    # 相位谱
    plt.plot(w, np.angle(H))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("phase")
    plt.title("Figure 2.2.2: Phase Spectrum")
    plt.show()

    # 幅度谱
    plt.plot(w, abs(H))
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.title("Figure 2.2.3: Amplitude Spectrum")
    plt.show()


if __name__ == "__main__":
    ex1_1()
    ex1_2()
    ex1_3()
    ex1_4()
    ex1_5()
    ex2_1()
    ex2_2()
