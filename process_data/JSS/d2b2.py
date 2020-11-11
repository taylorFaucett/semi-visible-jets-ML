import numpy as np
import tqdm
import energyflow as ef
    
def calc(X):
    output = []
    hl_graph = ef.D2(measure='hadr', 
                     beta=2, 
                     strassen=False, 
                     reg=0., 
                     kappa=1, 
                     normed=False, 
                     coords=None, 
                     check_input=True
                    )
    

    output = hl_graph.batch_compute(X)

    return np.hstack(output)

