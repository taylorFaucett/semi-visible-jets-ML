from os import path, getcwd, mkdir
import pandas as pd
import numpy as np
import h5py
import glob
import energyflow as ef
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from average_decision_ordering import calc_ado
from nn import nn
import pathlib

path = pathlib.Path.cwd()

home = path.dirname(getcwd())
bf_dir = path.join(home, "brute-force")
pass_dir = path.join(bf_dir, "pass_data")
data_dir = path.join(home, "data")


def data_grabber(selected_efps):
    hl_file = path.join(data_dir, "raw", "HL", "test_no_pile_5000000.h5")
    hl_data = h5py.File(hl_file, "r")
    mass = np.hstack(hl_data["features"][:, 0])
    pT = np.hstack(pd.read_hdf(f"{data_dir}/raw/pT.h5", "pT").values)
    df = pd.DataFrame({"mass": mass, "pT": pT})
    for efp in selected_efps:
        efp_file = f"{data_dir}/efp/{efp}.feather"
        dfi = pd.read_feather(efp_file).features
        dfi = pd.DataFrame({efp: dfi.values})
        df = pd.concat([df, dfi], axis=1)
    scaler = preprocessing.StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df))
    y = np.hstack(hl_data["targets"][:])
    return df, y


def get_all_efps():
    efpset = ef.EFPSet("d<=5", "p==1")
    kappas = [0.5, 1, 2]
    betas = [0.5, 1, 2]
    graphs = efpset.graphs()
    graph_list, final_graphs = [], []
    for ix, graph in enumerate(graphs):
        for kappa in kappas:
            for beta in betas:
                n, e, d, v, k, c, p, h = efpset.specs[ix]
                graph_ix = f"EFP_{n}_{d}_{k}_k_{kappa}_b_{beta}"
                graph_list.append(graph_ix)
    glob_list = glob.glob(path.join(data_dir, "efp", "*.feather"))
    for graph in graph_list:
        if path.join(data_dir, "efp", graph + ".feather") in glob_list:
            final_graphs.append(graph)
    return final_graphs


def train_nn(efp_ix, past_efps, pass_ix):
    selected_efps = past_efps.copy()
    selected_efps.append(efp_ix)
    it_dir = f"{pass_dir}/p{pass_ix}"
    X, y = data_grabber(selected_efps)
    ll = h5py.File(path.join(data_dir, "raw", "LL.h5"), "r")["yhat"][:]

    X_train, X_val, y_train, y_val, ll_train, ll_val = train_test_split(
        X, y, ll, test_size=0.2, random_state=pass_ix
    )

    X_val, X_test, y_val, y_test, ll_val, ll_test = train_test_split(
        X_val, y_val, ll_val, test_size=0.5, random_state=pass_ix
    )

    # Try different network designs according to number of hidden layers and units
    model_file = path.join(it_dir, "models", efp_ix + ".h5")

    model = nn(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=20,  # 20,
        batch_size=2048,
        layers=3,
        nodes=50,
        model_file=model_file,
        verbose=0,
    )

    test_pred = np.hstack(model.predict(X_test))
    auc_val = roc_auc_score(y_test, test_pred)
    ado_val = calc_ado(ll_test, test_pred, y_test, 1000000)

    if path.isfile(f"{it_dir}/stats.csv"):
        stats_file = pd.read_csv(f"{it_dir}/stats.csv", index_col=0)
    else:
        stats_file = pd.DataFrame(columns=["efp", "auc", "ado"])
    df_ix = pd.DataFrame({"efp": efp_ix, "auc": auc_val, "ado": ado_val}, index=[0])
    stats_file = pd.concat([stats_file, df_ix], axis=0)
    stats_file.to_csv(f"{it_dir}/stats.csv")
    return auc_val, ado_val


def brute_force_iteration(pass_ix, past_efps):
    it_dir = f"{pass_dir}/p{pass_ix}"
    # First, check to see if we've already run this pass
    if not path.exists(it_dir):
        mkdir(it_dir)
        mkdir(f"{it_dir}/models")

    t = tqdm(all_efps)
    if path.isfile(f"{it_dir}/stats.csv"):
        existing_efps = pd.read_csv(f"{it_dir}/stats.csv").efp.values
    else:
        existing_efps = []
    for efp_ix in t:
        if not efp_ix in existing_efps:
            auc_val, ado_val = train_nn(efp_ix, past_efps, pass_ix)
            t.set_description(
                f"Last NN: {efp_ix} - AUC={auc_val:.3f} - ADO={ado_val:.3f}"
            )
            t.refresh()


def get_max_efp(pass_ix):
    it_dir = f"{pass_dir}/p{pass_ix}"
    stats_file = f"{it_dir}/stats.csv"
    stats_df = pd.read_csv(stats_file, index_col=0)
    sorted_stats = stats_df.sort_values(by=["ado"], ascending=False)
    maxs = sorted_stats.iloc[0]
    efp = maxs.efp
    auc = maxs.auc
    ado = maxs.ado
    return efp, auc, ado


if __name__ == "__main__":
    auc_val = 0
    pass_ix = 0
    ll_benchmark = 0.9550  # 0.9530
    all_efps = get_all_efps()
    bf_stats = pd.DataFrame(columns=["efp", "auc", "ado"])
    past_efps = []
    while auc_val < ll_benchmark:
        brute_force_iteration(pass_ix, past_efps)
        max_efp, max_auc, max_ado = get_max_efp(pass_ix)
        df_ix = pd.DataFrame(
            {"efp": max_efp, "auc": max_auc, "ado": max_ado}, index=[pass_ix]
        )
        bf_stats = pd.concat([bf_stats, df_ix], axis=0)
        bf_stats.to_csv(path.join(bf_dir, "selected_efps.csv"))
        past_efps.append(max_efp)
        auc_val = max_auc
        pass_ix += 1
        all_efps.remove(max_efp)
