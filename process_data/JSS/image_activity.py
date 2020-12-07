import numpy as np
import tqdm
import energyflow as ef


def calc(X):
    output = []
    for entry in tqdm.tqdm(X):
        # Separate the pT, eta, phi from the entry
        pT = entry[:, 0]
        eta = entry[:, 1]
        phi = entry[:, 2]

        # Calculate the observable value at each index ix
        obs_value = ef.image_activity(
            entry, f=0.95, R=1.0, npix=32, center=None, axis=None
        )

        # Add the observable value to the output list
        output.append(obs_value)

    # Return full set of values as a 1-D array
    return np.hstack(output)
