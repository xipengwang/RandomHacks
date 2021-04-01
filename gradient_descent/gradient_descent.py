from matplotlib import pyplot as plt
import numpy as np

X, Y = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))

def plot_gradient_descent(ax, lr, func, dfunc, hessian):
    Z = func(X, Y)
    f = Z[:-1, :-1]
    z_min, z_max = 0, np.abs(f).max()

    c = ax.pcolormesh(X, Y, f, cmap='hot', vmin=z_min, vmax=z_max)
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    fig.colorbar(c, ax=ax)

    x = 8
    y = 8

    xs = [x]
    ys = [y]
    for epoch in range(0, 1000):
        df_x, df_y = dfunc(x, y)
        x -= lr * df_x
        y -= lr * df_y
        xs.append(x)
        ys.append(y)
        if func(x, y) < 1E-3:
            break
    cond = np.linalg.cond(hessian(x, y))
    ax.set_title(f"cond #: {cond}, lr: {lr}, iters: {epoch}")
    ax.plot(xs, ys, 'g-*')
    ax.set_xticks([])

lrs = [0.05, 0.06, 0.07, 0.08, 0.095, 0.1]

for lr in lrs:
    fig, (ax1, ax2) = plt.subplots(2)
    def func1(x, y):
        return x**2 + y**2
    def dfunc1(x, y):
        return [2*x, 2*y]
    # Hessian
    def hessian1(x, y):
        return np.array([[2, 0], [0, 2]])
    plot_gradient_descent(ax1, lr, func1, dfunc1, hessian1)

    def func2(x, y):
        return x**2 + 10*y**2
    def dfunc2(x, y):
        return [2*x, 20*y]
    def hessian2(x, y):
        return np.array([[2, 0], [0, 20]])

    plot_gradient_descent(ax2, lr, func2, dfunc2, hessian2)

plt.show()
