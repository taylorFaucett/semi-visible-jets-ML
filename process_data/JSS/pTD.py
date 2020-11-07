import numpy as np

def calc(X):
    pTD_output = []
    for pT_ix in X:
        # Separate just the pT values from the eta, phi columns
        pT_ix = pT_ix[:,0]
        
        #Calculate the pTD value at each index ix
        pTD_value = np.sqrt(np.sum(pT_ix ** 2)) / np.sum(pT_ix)

        # Add the pTD value to the output list
        pTD_output.append(pTD_value)

    # Return full set of pTD values
    return np.hstack(pTD_output)
