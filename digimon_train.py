# train_digimon.py
import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt

from ANN import GeneralNetwork  # your NN class with pack/unpack & helpers
from optimizers import LineSearchSGD, MiniBatchBFGS_Wolfe

# ---------------------------
# 1. Load dataset
# ---------------------------
df = pd.read_csv("DigiDB_digimonlist.csv")

# Binary target: Rookie vs non-Rookie
df['target'] = (df['Stage'] == 'Rookie').astype(int)

# Features: Attack & Memory
features = ['Lv50 Atk', 'Memory']
X = df[features].fillna(0).values.T  # shape (2, n_samples)
Y = df['target'].values.reshape(1, -1)  # shape (1, n_samples)

# One-hot encode target (2 classes: non-Rookie, Rookie)
Y_onehot = np.vstack([1 - Y, Y])


# ---------------------------
# 2. Wrap in Data-like object
# ---------------------------
class DigimonData:
    def __init__(self, X, Y):
        self.xtrain = X.astype(float)
        self.ytrain = Y.astype(float)
        self.highamdata = False

data = DigimonData(X, Y_onehot)


# ---------------------------
# 3. Train networks
# ---------------------------

# (a) Original SGD
net_sgd = GeneralNetwork(number_of_layers=3, neurons_per_layer=[2, 10, 2])
hist_sgd = net_sgd.train(data, eta=1, epochs=200)

# (b) LineSearchSGD
net_ls = GeneralNetwork(number_of_layers=3, neurons_per_layer=[2, 10, 2])
hist_ls = net_ls.train_with_linesearch_sgd(data, LineSearchSGD(), epochs=200)

# (c) Stochastic BFGS
net_sbfgs = GeneralNetwork(number_of_layers=3, neurons_per_layer=[2, 10, 2])
hist_sbfgs = net_sbfgs.train_with_minibatch_bfgs(data, MiniBatchBFGS_Wolfe(), epochs=200)


# ---------------------------
# 4. Plot learning curves
# ---------------------------
plt.figure(figsize=(8, 5))
plt.plot(hist_sgd, label="Original SGD")
plt.plot(hist_ls, label="LineSearch SGD")
plt.plot(hist_sbfgs, label="Stochastic BFGS")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.legend()
plt.title("Training Loss on Digimon Dataset (Attack vs Memory)")
plt.show()


# ---------------------------
# 5. Decision boundaries
# --------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
net_sgd.plot_decision_boundary(data, title="Original SGD", ax=axes[0])
net_ls.plot_decision_boundary(data, title="LineSearch SGD", ax=axes[1])
net_sbfgs.plot_decision_boundary(data, title="Stochastic BFGS", ax=axes[2])
plt.tight_layout()
plt.show()


