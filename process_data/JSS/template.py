import numpy as np

def calc(X):
    output = []
    for entry in X:
        # Separate the pT, eta, phi from the entry
        pT = entry[:,0]
        eta = entry[:,1]
        phi = entry[:,2]
        
        # Calculate the observable value at each index ix 
        # example below is pTD
        obs_value = np.sqrt(np.sum(pT ** 2)) / np.sum(pT)

        # Add the observable value to the output list
        output.append(obs_value)

    # Return full set of values as a 1-D array
    return np.hstack(output)

