import torch
import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, cm
from matplotlib.patches import Circle, Ellipse
from mpl_toolkits.mplot3d import Axes3D, art3d
import os
import pickle
import sys
import pandas

sys.path.append('../Aging_Model')

with open('../Aging_Model/exp_aging_model.p', 'rb') as f:
    age_generator = pickle.load(f)

np.random.seed(0)
torch.manual_seed(0)

x1 = np.linspace(0, 1, 1001)
x2 = np.linspace(0, 1, 1001)
dx = x1[1] - x1[0]


def Loss(x1, x2):
    L = np.sin(10 * x1 + x2) + 2 * np.sin(2.5 * x2 + x1) + np.sin(6 * x1 * x2 + x1) + 1.5 * np.sin(
        12 * (x1 - x2)) + 2 * np.sin(12 * (1 - x1 + x2)) + 3 * np.exp(
        (-np.power(x1 - 0.4, 2.) - np.power(x2 - 0.4, 2.)) / 0.02)
    return -L + 4.760697618531012


LossSurface = np.zeros([1001, 1001])
for i in range(1001):
    for j in range(1001):
        LossSurface[i, j] = Loss(x1[i], x2[j])

np.min(LossSurface)

idx = np.where(LossSurface == np.min(LossSurface))

np.min(LossSurface), np.max(LossSurface)

cm.coolwarm.set_gamma(0.5)

fig = plt.figure(figsize=(12, 10))
ax = plt.axes(projection='3d')

X, Y = np.meshgrid(x1, x2)

surf = ax.plot_surface(X, Y, LossSurface, cmap=cm.coolwarm, alpha=0.5)

ax.scatter3D(x2[idx[1]], x1[idx[0]], LossSurface[idx[0], idx[1]], cmap=cm.seismic)

# # Set axes label
# ax.set_xlabel('x', labelpad=20)
# ax.set_ylabel('y', labelpad=20)
# ax.set_zlabel('z', labelpad=20)

# ax.set_xlim(0, 0.6)
# ax.set_ylim(0.2, 0.8)


plt.show()

from matplotlib.ticker import MaxNLocator

levels = MaxNLocator(nbins=10).tick_values(np.min(LossSurface), np.max(LossSurface))
plt.contourf(X, Y, LossSurface, levels=levels, cmap=cm.coolwarm, alpha=0.4)
plt.scatter(x2[idx[1]], x1[idx[0]], cmap=cm.seismic, c=LossSurface[idx[0], idx[1]])
# plt.colorbar()
plt.show()


def aging(g, models, t):
    decays = np.array([model([t]) for model in models]).flatten()
    return g * decays


def getweights(R, models, t):
    g = 1 / R
    if models is not None:
        g = aging(g, models, t)
    g = g / np.sum(g)
    W = g[:-2]
    return W


np.random.seed(3)
R = np.random.rand(4)


def showtrace(R, age_generator):
    models = age_generator.get_models(4)

    T = np.linspace(0, 1, 300)
    Ws = []
    for t in T:
        W = getweights(R, models, t)
        Ws.append(W)
    Ws = np.array(Ws)
    w1 = Ws[:, 0]
    w2 = Ws[:, 1]

    return w1, w2


w1, w2 = showtrace(R, age_generator)

width = np.max(w1) - np.min(w1)
height = np.max(w2) - np.min(w2)

w1 = w1 / width * 0.25
w2 = w2 / height * 0.15
w1 = w1 - w1[0]
w2 = w2 - w2[0]

plt.figure(figsize=[12, 8]);
plt.plot(w1, w2, '-', linewidth=3, label='$w(t)$', zorder=0);
plt.scatter(w1[0], w2[0], s=100, c='red', zorder=1);
plt.gca().set_aspect('equal');
plt.show()

starts = []
Ls = []
for s1 in np.linspace(0, 1, 101):
    for s2 in np.linspace(0, 1, 101):
        start = np.array([s1, s2])

        w1_move = np.round(w1 + start[0], 3)
        w2_move = w2 + start[1]

        if (np.min(w1_move) < 0 or np.min(w2_move) < 0 or np.max(w1_move) > 1 or np.max(w2_move) > 1):
            break
        else:
            L = 0
            for i1, i2 in zip(w1_move, w2_move):
                L += Loss(i1, i2)
            L /= 300
            Ls.append(L)
            starts.append(start)
    print(start)

arg = np.where(Ls == np.min(Ls))[0][0]


best_start = starts[arg]

w1_stupid = w1 + x1[idx[0]]
w2_stupid = w2 + x2[idx[1]]

L_stupid = []
for i1, i2 in zip(w1_stupid, w2_stupid):
    L_stupid.append(Loss(i1, i2))

w1_clever = w1 + best_start[0]
w2_clever = w2 + best_start[0]

L_clever = []
for i1, i2 in zip(w1_clever, w2_clever):
    L_clever.append(Loss(i1, i2))

from matplotlib.ticker import MaxNLocator

levels = MaxNLocator(nbins=10).tick_values(np.min(LossSurface), np.max(LossSurface))
plt.contourf(X, Y, LossSurface, levels=levels, cmap=cm.coolwarm, alpha=0.4)
plt.scatter(w2_clever, w1_clever, cmap=cm.coolwarm, c=L_clever, s=2)
plt.scatter(w2_stupid, w1_stupid, cmap=cm.coolwarm, c=L_stupid, s=2)
plt.gca().set_aspect('equal');
plt.show()

w1 = np.hstack((w1_clever, w1_stupid))
w2 = np.hstack((w2_clever, w2_stupid))
L = np.hstack((L_clever, L_stupid))


fig = plt.figure(figsize=(12, 10))
ax = plt.axes(projection='3d')

X, Y = np.meshgrid(x1, x2)

norm = plt.Normalize(0, 8)

surf = ax.plot_surface(X, Y, LossSurface, cmap=cm.coolwarm, alpha=0.4, norm=norm)

ax.scatter3D(w2, w1, L, cmap=cm.coolwarm)
# ax.scatter3D(w2, w1, L_stupid, cmap=cm.coolwarm, norm=norm)
# # Set axes label
# ax.set_xlabel('x', labelpad=20)
# ax.set_ylabel('y', labelpad=20)
# ax.set_zlabel('z', labelpad=20)

# ax.set_xlim(-, 1.01)
# ax.set_xlim(-1.01, 1.01)
ax.set_zlim(-8, 10)
# ax.set_axis_off()
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# cset = ax.contourf(X, Y, LossSurface,
#                    zdir ='z',
#                    offset = -8,
#                    cmap = cm.coolwarm, alpha=0.3, norm=norm)
# ax.scatter3D(w2_clever, w1_clever, -8, cmap=cm.coolwarm, c=L_clever, norm=norm)
# ax.scatter3D(w2_stupid, w1_stupid, -8, cmap=cm.coolwarm, c=L_stupid, norm=norm)


# ax.azim = 45
# ax.dist = 10
# ax.elev = 90
plt.show()

plt.plot(L_stupid)
plt.plot(L_clever)
