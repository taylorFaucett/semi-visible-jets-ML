import pandas as pd
import numpy as np
import h5py
import pathlib
from sklearn.metrics import roc_auc_score
from nn import nn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm

path = pathlib.Path.cwd()
    
def data_grabber(efp):
    scaler = preprocessing.StandardScaler()
    hl_file = path.parent / "data" / "raw" / "HL" / "test_no_pile_5000000.h5"
    hl_data = h5py.File(hl_file, "r")
    hl = hl_data["features"][:]
    y = hl_data["targets"][:]
    df = pd.DataFrame(hl)
    
    efp_file = path.parent / "data" / "efp" / f"{efp}.feather"
    dfp = pd.read_feather(efp_file)["features"]
    dfp = pd.DataFrame({efp: dfp.values})
    
    df = pd.concat([df, dfp], axis=1)
    df = pd.DataFrame(scaler.fit_transform(df))
    return df, np.hstack(y)

def train_nn(efp):
    layers = 3
    nodes = 100
    batch_size = 512

    # Find the "first" EFP that is most similar to the NN(LL) predictions
    # Train a simple NN with this first choice
    X, y = data_grabber(efp)

    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_val,
                                                    y_val,
                                                    test_size=0.5,
                                                    random_state=42)

    # Try different network designs according to number of hidden layers and units
    model_file = path / "6HL_1EFP_models" / f"{efp}.h5"

    model = nn(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=200,
        batch_size=batch_size,
        layers=layers,
        nodes=nodes,
        model_file=model_file,
        verbose=0,
    )

    predictions = np.hstack(model.predict(X))
    auc_val = roc_auc_score(y_test, np.hstack(model.predict(X_test)))
    print(f"test-set AUC={auc_val:.4}")
    return auc_val

if __name__ == "__main__":
    main_efp_file = path / "results" / "supplemental" / "p0" / "dif_order_ado_comparison.csv"
    main_efps = list(pd.read_csv(main_efp_file, index_col=0).iloc[:10].efp.values)
    other_efps = ["EFP_1_0_0_k_0.5_b_0.5", "EFP_1_0_0_k_0_b_0.5", 
            "EFP_1_0_0_k_2_b_0.5", "EFP_5_8_180_k_1_b_2", 
            "EFP_6_8_335_k_2_b_0.5", "EFP_2_1_0_k_2_b_0.5", 
            "EFP_8_7_1_k_0_b_1", "EFP_5_7_81_k_-1_b_2"
           ]
    
    best_kb = ['EFP_4_8_60_k_-1_b_0.5', 'EFP_6_7_12_k_-1_b_1', 'EFP_5_7_81_k_-1_b_2', 
               'EFP_2_1_0_k_0_b_0.5', 'EFP_8_7_1_k_0_b_1', 'EFP_2_1_0_k_0_b_2', 
               'EFP_6_7_1_k_0.5_b_0.5', 'EFP_6_7_105_k_0.5_b_1', 'EFP_2_1_0_k_0.5_b_2', 
               'EFP_6_5_0_k_1_b_0.5', 'EFP_6_7_12_k_-1_b_1', 'EFP_5_7_81_k_-1_b_2', 'EFP_4_4_3_k_2_b_0.5', 
               'EFP_2_1_0_k_2_b_1', 'EFP_2_1_0_k_2_b_2']
    best_c = ['EFP_4_4_3_k_2_b_0.5', 'EFP_6_8_335_k_2_b_0.5', 'EFP_8_7_16_k_2_b_0.5', 'EFP_1_0_0_k_0.5_b_0.5']
    efps = set(main_efps + other_efps + best_kb + best_c)
    df_out = pd.DataFrame()
    for ix, efp in enumerate(tqdm(efps)):
        auc_val = train_nn(efp)
        dfi = pd.DataFrame({"efp":efp, "auc":auc_val}, index=[ix])
        df_out = pd.concat([df_out, dfi])
        df_out.to_csv(path / "6HL_1EFP_results.csv")

