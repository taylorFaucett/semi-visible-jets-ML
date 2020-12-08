import numpy as np
import pandas as pd
import glob
import h5py
import random
import shutil
import itertools

from sklearn.metrics import roc_auc_score
from os import getcwd, path, chdir, mkdir
from nn import nn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from visualization import visualization
from do_tools import do_calc, ado_calc

import pathlib

path = pathlib.Path.cwd()


def random_pairs(x, y, l):
    min_size = min(len(x), len(y))
    x = x[:min_size]
    y = y[:min_size]
    random.shuffle(x)
    random.shuffle(y)
    rp = np.vstack((x, y)).T
    while len(rp) < l:
        random.shuffle(x)
        random.shuffle(y)
        app_rp = np.vstack((x, y)).T
        rp = np.concatenate((rp, app_rp), axis=0)
    df = pd.DataFrame({"x": rp[:, 0], "y": rp[:, 1]})
    df.drop_duplicates(inplace=True, keep="first")
    return df.to_numpy()


def scale_data(x, mean=True):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        std = 1
    return (x - mean) / std


def get_data(selected_efps, include_hl, include_jet_pT, include_muon_pT, lessCones):
    # Data path
    data_path = path.parent / "data"

    # Load HL, pT and labels
    df = pd.DataFrame()

    # Combine data into a dataframe
    if include_hl:
        hl = pd.DataFrame(np.load(data_path / "raw" / "cones.npy"))
        if lessCones:
            hl = hl[[5, 6, 7, 8, 9, 11, 13, 15]]
        df = pd.concat([df, hl], axis=1)
    if include_muon_pT:
        muon_pT = pd.DataFrame(np.load(data_path / "raw" / "muon_pt.npy"))
        df = pd.concat([df, muon_pT], axis=1)
    if include_jet_pT:
        jet_pT = pd.DataFrame(np.load(data_path / "raw" / "jet_pt.npy"))
        df = pd.concat([df, jet_pT], axis=1)

    # Include any chosen EFPs
    efps = h5py.File(data_path / "efps.h5", "r")
    for efp in selected_efps:
        dfi = pd.DataFrame({efp: efps["efps"][efp]})
        df = pd.concat([df, dfi], axis=1)

    # Scale/Normalize the data
    scaler = preprocessing.StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df))
    y = np.load(data_path / "raw" / "labels.npy")
    return df, y


def isolate_order(ix, N_pairs):
    dif_order_path = pass_path / "dif_order.feather"
    if dif_order_path.exists():
        print(f"Skipping isolate_order for pass {ix}")
        dif_data = pd.read_feather(dif_order_path)
        idxp0 = dif_data["idx0"].values
        idxp1 = dif_data["idx1"].values
        return idxp0, idxp1
    else:
        # Get the predictions from the previous iteration
        hl_file = pass_path / "test_pred.feather"
        dfy = pd.read_feather(hl_file)

        # Separate data into signal and background
        dfy_sb = dfy.groupby("y")

        # Set signal/background
        df0 = dfy_sb.get_group(0)
        df1 = dfy_sb.get_group(1)

        # get the separate sig/bkg indices
        idx0 = df0.index.values.tolist()
        idx1 = df1.index.values.tolist()

        # generate a random set of sig/bkg pairs
        print(f"Generating (N={int(N_pairs):,}) sig/bkg pairs")
        idx_pairs = random_pairs(idx0, idx1, N_pairs)
        print(f"After duplicates remove, (N={int(len(idx_pairs)):,}) remaining")

        idxp0 = idx_pairs[:, 0]
        idxp1 = idx_pairs[:, 1]

        # grab the ll and hl values for those sig/bkg pairs
        dfy0 = dfy.iloc[idxp0]
        dfy1 = dfy.iloc[idxp1]
        ll0 = dfy0["ll"].values
        ll1 = dfy1["ll"].values
        hl0 = dfy0["hl"].values
        hl1 = dfy1["hl"].values

        # find differently ordered pairs
        dos = do_calc(fx0=ll0, fx1=ll1, gx0=hl0, gx1=hl1)

        # let's put all of the data and decision-ordering in 1 data frame
        do_df = pd.DataFrame(
            {
                "idx0": idxp0,
                "idx1": idxp1,
                "ll0": ll0,
                "ll1": ll1,
                "hl0": hl0,
                "hl1": hl1,
                "dos": dos,
            }
        )

        # split the similar and differently ordered sets
        do_df_grp = do_df.groupby("dos")
        dif_df = do_df_grp.get_group(0)
        sim_df = do_df_grp.get_group(1)
        dif_df.reset_index().to_feather(pass_path / "dif_order.feather")

        return idxp0, idxp1


def check_efps(ix):
    comp_file = pass_path / "dif_order_ado_comparison.csv"
    if comp_file.exists():
        print(f"Skipping check_efps for pass {ix}")
        return
    else:
        # Load the diff-ordering results
        dif_df = pd.read_feather(pass_path / "dif_order.feather")

        # Grab the dif-order indices and ll features corresponding to those
        idx0 = dif_df["idx0"].values
        idx1 = dif_df["idx1"].values
        ll0 = dif_df["ll0"].values
        ll1 = dif_df["ll1"].values

        print(f"Checking ADO on diff-order subset of size N = {len(dif_df):,}")

        # get the efps to check against the dif_order results
        efp_file = h5py.File(path.parent / "data" / "efps.h5", "r")
        efps = list(efp_file["efps"].keys())

        # If we only want IRC Safe observables, remove any EFP label from the list without "k=1"
        if irc_safe:
            efps = list(filter(lambda x: "k_1" in x, efps))

        if small_set:
            efps_new = []
            for nix in range(5):
                efps_ix = list(filter(lambda x: f"efp_{nix+1}" in x, efps))
                efps_new.extend(efps_ix)
            efps = efps_new.copy()

        if super_small_set:
            allowed_efp = [
                "efp_1_0_0",
                "efp_2_1_0",
                "efp_2_2_0",
                "efp_2_3_0",
                "efp_3_2_0",
                "efp_3_3_0",
                "efp_3_3_1",
                "efp_3_4_0",
                "efp_3_4_1",
                "efp_3_4_2",
                "efp_3_5_1",
                "efp_3_5_2",
                "efp_3_5_3",
                "efp_3_6_2",
                "efp_3_6_4",
                "efp_3_6_5",
                "efp_3_7_5",
                "efp_3_7_6",
            ]
            efps_new = []
            for efpi in efps:
                if efpi.split("_k_")[0] in allowed_efp:
                    efps_new.append(efpi)
            efps = efps_new.copy()

        if small_kb:
            efps = list(filter(lambda x: "k_1" in x, efps))
            efps_new = []
            kappas = [0, 1, 2]
            betas = [1, 2]
            kb_values = list(itertools.product(kappas, betas))
            kb_values = [f"k_{x[0]}_b_{x[1]}" for x in kb_values]
            for efpix in efps:
                for kb_value in kb_values:
                    if kb_value in efpix:
                        efps_new.append(efpix)
                        break
            efps = efps_new.copy()

        # Remove previously selected efps
        for selected_efp in selected_efps:
            print(f"removing efp: {selected_efp}")
            efps.remove(selected_efp)

        ado_df = pd.DataFrame()
        ado_max = 0
        for iy, efp in enumerate(tqdm(efps)):
            # select the dif-order subset from dif_df for the efp
            efp_df = efp_file["efps"][efp][:]

            # Use the same diff-order sig/bkg pairs to compare with ll predictions
            # efp0 = efp_df.iloc[idx0][efp_type].values
            # efp1 = efp_df.iloc[idx1][efp_type].values

            efp0 = np.take(efp_df, idx0)
            efp1 = np.take(efp_df, idx1)

            # Calculate the ado
            ado_val = ado_calc(fx0=ll0, fx1=ll1, gx0=efp0, gx1=efp1)

            dfi = pd.DataFrame({"efp": efp, "ado": ado_val}, index=[iy])
            ado_df = pd.concat([ado_df, dfi], axis=0)
        ado_df = ado_df.sort_values(by=["ado"], ascending=False)
        ado_df.to_csv(pass_path / "dif_order_ado_comparison.csv")


def get_max_efp(ix):
    df = pd.read_csv(pass_path / "dif_order_ado_comparison.csv", index_col=0)

    # sort by max ado
    dfs = df.sort_values(by=["ado"], ascending=False)
    efp_max = dfs.iloc[0]["efp"]
    ado_max = dfs.iloc[0]["ado"]
    print(f"Maximum dif-order graph selected: {efp_max}")
    return efp_max, ado_max


def train_nn(ix):
    pred_file = pass_path / "test_pred.feather"
    layers = 5
    nodes = 50
    batch_size = 64

    # Find the "first" EFP that is most similar to the NN(LL) predictions
    # Train a simple NN with this first choice
    X, y = get_data(
        selected_efps=selected_efps,
        include_hl=include_hl,
        include_jet_pT=include_jet_pT,
        include_muon_pT=include_muon_pT,
        lessCones=lessCones,
    )

    np.save(pass_path / "X.npy", X)
    np.save(pass_path / "y.npy", y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.5, random_state=42
    )

    # Try different network designs according to number of hidden layers and units
    model_file = model_path / f"model_l_{layers}_n_{nodes}_bs_{batch_size}.h5"

    model = nn(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=3000,
        batch_size=batch_size,
        layers=layers,
        nodes=nodes,
        model_file=model_file,
        verbose=2,
    )

    predictions = np.hstack(model.predict(X))
    auc_val = roc_auc_score(y_test, np.hstack(model.predict(X_test)))
    print(f"test-set AUC={auc_val:.4}")

    if truth_guided:
        ll = y.copy()
    else:
        ll = np.concatenate(
            np.load(path.parent / "data" / "raw" / "ll_predictions.npy")
        )
    test_df = pd.DataFrame({"hl": predictions, "y": y, "ll": ll})
    test_df.to_feather(pred_file)
    return auc_val


if __name__ == "__main__":
    run_name = "run"

    # Define run type (i.e. include HL, include pT)
    include_hl = True
    include_jet_pT = True
    include_muon_pT = False
    irc_safe = True
    truth_guided = False
    small_set = False
    small_kb = True
    super_small_set = False
    lessCones = True

    # Include data details in run_name
    if include_hl:
        run_name += "-hl"
    if include_jet_pT:
        run_name += "-jetPT"
    if include_muon_pT:
        run_name += "-muonPT"
    if irc_safe:
        run_name += "-ircSafe"
    if truth_guided:
        run_name += "-truthGuided"
    if small_set:
        run_name += "-smallSet"
    if super_small_set:
        run_name += "-superSmallSet"
    if small_kb:
        run_name += "-smallKB"
    if lessCones:
        run_name += "-8Cones"

    # Define run path
    run_path = path / "runs" / run_name

    # Define a stopping point based on LL AUC
    ll_benchmark = 0.8475

    if not run_path.exists():
        run_path.mkdir()

    # Copy guided_iteration and nn files
    shutil.copyfile(path / "guided_iteration.py", run_path / "guided_iteration_copy.py")
    shutil.copyfile(path / "nn.py", run_path / "nn_copy.py")

    selected_efps, aucs, ados = [], [], []
    ix, ado_max, auc_val = 0, 0, 0
    while ado_max < 1 and auc_val < ll_benchmark:
        print(
            f"Iteration auc (AUC={auc_val:.4f}) is less than the LL benchmark (AUC={ll_benchmark:.4f})"
        )
        print(f"Iteration will continue to run: Pass {ix}")
        # Define data sub-directories
        pass_path = run_path / f"p{ix}"
        model_path = pass_path / "models"

        # Setting the random seed to a predictable value (in this case iteration index)
        # Will make it easier to reproduce results in the future if necessary (despite shuffling diff-order pairs)
        random.seed(ix)

        # Create a sub-directory for the pass to store all relevant data
        if not pass_path.exists():
            pass_path.mkdir()
            model_path.mkdir()

        # Train a NN using current EFP selections (or just HL when ix=0)
        auc_val = train_nn(ix)
        print(f"Iteration {ix} -> AUC: {auc_val:.4}")

        # Store the auc results
        aucs.append(auc_val)
        pass_list = ["hl6"] + selected_efps
        ados_list = [np.nan] + ados
        efp_df = pd.DataFrame({"efp": pass_list, "auc": aucs, "ado": ados_list})
        efp_df.to_csv(run_path / "selected_efps.csv")

        # Isolate random dif-order pairs
        isolate_order(ix=ix, N_pairs=5e7)

        # Check ado with each EFP for most similar DO on dif-order pairs
        check_efps(ix)

        # Get the max EFP and save it
        efp_max, ado_max = get_max_efp(ix)
        selected_efps.append(efp_max)
        print(f"Selected EFPs in Pass {ix}")
        print(selected_efps)
        ados.append(ado_max)

        # Make plots
        viz = visualization(run_path, ix)
        viz.dif_order_hist_plots()
        viz.performance_plot()
        viz.clear_viz()
        ix += 1
