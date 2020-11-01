import h5py
import numpy as np
import pandas as pd
import energyflow as ef
import glob
import os
from tqdm import tqdm, trange
import pathlib
path = pathlib.Path.cwd()

def efp(data, graph, kappa, beta):
    EFP_graph = ef.EFP(graph, measure="hadr", kappa=kappa, beta=beta, normed=False)
    X = EFP_graph.batch_compute(data)
    return X


def generate_efp():
    rinvs = ["0p0", "0p3", "1p0"]
    for rinv in rinvs:
        # Choose values for graphs, kappa, beta
        kappas = [-1, 0, 0.5, 1, 2]
        betas = [0.5, 1, 2]

        hdf_file = path.parent / "data" / "processed" / f"{rinv}-prep_data.h5"
        X = pd.read_hdf(hdf_file, "features").features.to_numpy()
        y = pd.read_hdf(hdf_file, "targets").targets.values
        # All prime graphs with dimension d<=7
        prime_d7 = ef.EFPSet("d<=7", "p==1")
        
        # All prime graphs with dimension d<=8 and chromatic number 4
        chrom_4 = ef.EFPSet("d<=8", "p==1", "c==4")
        efpsets = [prime_d7, chrom_4]
        for efpset in efpsets:
            graphs = efpset.graphs()
            print(f" Computing N={len(kappas)*len(betas)*len(graphs):,} graphs")
            for efp_ix, graph in enumerate(tqdm(graphs)):
                for kappa in kappas:
                    for beta in betas:
                        n, e, d, v, k, c, p, h = efpset.specs[efp_ix]
                        
                        # Each prime graph is uniquely identified by an (n,d,k) number
                        data_file = path.parent / "data" / "efp" / rinv / f"{n}_{d}_{k}_k_{kappa}_b_{beta}.feather"
                        if not os.path.exists(data_file):
                            print(data_file)
                            efp_val = efp(data=X, graph=graph, kappa=kappa, beta=beta)
                            efp_df = pd.DataFrame({f"features": efp_val, f"targets": y})
                            efp_df.to_feather(data_file)


if __name__ == "__main__":
    generate_efp()
