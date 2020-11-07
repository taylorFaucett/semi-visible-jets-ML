import numpy as np

def calc(X):
    output = []
    for entry in X:
        # Separate the pT, eta, phi from the entry
        pT = entry[:,0]
        eta = entry[:,1]
        phi = entry[:,2]
        
        # Calculate the observable value at each index ix 
        obs_value = np.sqrt(np.sum(pT_ix ** 2)) / np.sum(pT_ix)

        # Add the observable value to the output list
        output.append(obs_value)

    # Return full set of pTD values
    return np.hstack(output)
