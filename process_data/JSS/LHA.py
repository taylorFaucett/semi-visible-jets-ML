import numpy as np
import tqdm
    
def calc(X):
    output = []
    kappa = 1
    beta = 0.5
    obs_value = []
    for entry in tqdm.tqdm(X):
        # Separate the pT, eta, phi from the entry
        pT = entry[:,0]
        eta = entry[:,1]
        phi = entry[:,2]
        obs_value = 0
        for pT_ix, eta_ix, phi_ix in zip(pT, eta, phi):
            theta_ix = np.sqrt((eta_ix**2) + (phi_ix**2))
            obs_value += (pT_ix ** kappa) * (theta_ix ** beta)
        output.append(obs_value)

    # Return full set of values as a 1-D array
    return np.hstack(output)

