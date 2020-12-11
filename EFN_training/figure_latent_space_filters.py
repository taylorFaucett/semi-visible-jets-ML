import pickle
import numpy as np
import matplotlib.pyplot as plt
from energyflow.archs import EFN
from energyflow.utils import data_split, to_categorical
import pathlib

path = pathlib.Path.cwd()
plt.rcParams.update({"font.size": 6})

epochs = 10
event_size = 32 * 32
batch_size = 50

# Load data
data_folder = path.parent / "data"
with open(f"{data_folder}/pixelated_data_efps_normed.pickle", "rb") as handle:
    pixelated_data = pickle.load(handle)

energies = np.zeros((len(pixelated_data["ET"]), event_size))
coords = np.zeros((len(pixelated_data["ET"]), event_size, 2))
for i, (energy, eta, phi) in enumerate(
    zip(pixelated_data["ET"], pixelated_data["eta"], pixelated_data["phi"])
):
    for j in range(len(energy)):
        energies[i][j] = energy[j]
        coords[i][j][0] = eta[j]
        coords[i][j][1] = phi[j]

x = [energies, coords]
y = to_categorical(pixelated_data["label"], num_classes=2)

activations = ["relu", "LeakyReLU"]
num_filters = [4, 8]
fig, axes = plt.subplots(2, 2)
for i, act in enumerate(activations):
    for j, l in enumerate(num_filters):
        Phi_sizes, F_sizes = (100, 100, l), (100, 100, 100)

        efn = EFN(
            input_dim=2, Phi_sizes=Phi_sizes, F_sizes=F_sizes, Phi_acts=act, F_acts=act
        )

        efn.fit(x, y, epochs=epochs, batch_size=batch_size)

        # plot settings
        R, n = 0.4, 100
        colors = ["Reds", "Oranges", "Greens", "Blues", "Purples", "Greys"]
        grads = np.linspace(0.45, 0.55, 4)

        # evaluate filters
        X, Y, Z = efn.eval_filters(R, n=n)

        # plot filters
        print(np.shape(Z))
        for k, z in enumerate(Z):
            axes[i][j].contourf(
                X, Y, z / np.max(z), grads, cmap=colors[k % len(colors)]
            )

        axes[i][j].set_xticks(np.linspace(-R, R, 5))
        axes[i][j].set_yticks(np.linspace(-R, R, 5))
        axes[i][j].set_xticklabels([f"-{R}", f"-{R/2}", "0", f"{R/2}", f"{R}"])
        axes[i][j].set_yticklabels([f"-{R}", f"-{R/2}", "0", f"{R/2}", f"{R}"])
        axes[i][j].set_xlabel("Translated Rapidity y")
        axes[i][j].set_ylabel("Translated Azimuthal Angle phi")
        axes[i][j].set_title(f"{l} Observable Energy Flow Network Latent Space")

plt.tight_layout()
folder = "./figures"
plt.savefig(f"{folder}/latent_space_filters.png")
