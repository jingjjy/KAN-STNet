import numpy as np
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


def z_score_normalize(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = (data - mean) / std_dev
    return normalized_data


def get_L96_func(N, F):
    def L96(t, x):
        d = np.zeros(N)

        for i in range(N):
            d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
        return d
    return L96

def gen_L96_data(N, F, time_range=(0, 20), dt=0.02, x_init_way='norm', init_param={'mu': 0, 'sigma': 0.1}):


    t_eval = np.arange(time_range[0], time_range[1], dt)
    if x_init_way == 'ones':
        x0 = F * np.ones(N)  # Initial state (equilibrium)
        x0[0] += 0.01  # Add small perturbation to the first variable
    elif x_init_way == 'norm':
        x0 = np.random.randn(N) * init_param['sigma'] + init_param['mu']
    else:
        raise NotImplementedError()

    x = integrate.solve_ivp(get_L96_func(N, F), time_range, x0, t_eval=t_eval).y

    return x

if __name__ == '__main__':
    N = 60  # Number of variables
    F = 5  # Forcing
    time_range=(0, 30)
    dt=0.02

    data_x = gen_L96_data(N, F, time_range, dt)

    X = np.arange(*time_range, dt)
    Y = np.arange(N)
    print(data_x.shape)
    XX, YY = np.meshgrid(Y, X)
    fig = plt.figure(figsize=(4, 6))
    plt.contourf(XX, YY, data_x.T, 100, cmap=mpl.colormaps['seismic'])
    plt.title(f'F={F}')
    plt.xlabel('X')
    plt.ylabel('t')
    plt.tight_layout()
    plt.show()