import numpy as np
import tqdm

def calc(X):
    output = []
    for entry in tqdm.tqdm(X):
        # Separate the pT, eta, phi from the entry
        pT = entry[:,0]
        
        # Calculate the observable value at each index ix 
        obs_value = np.sqrt(np.sum(pT ** 2)) / np.sum(pT)

        # Add the observable value to the output list
        output.append(obs_value)

    # Return full set of pTD values
    return np.hstack(output)
