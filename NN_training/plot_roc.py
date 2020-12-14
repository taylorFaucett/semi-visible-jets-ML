import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd


def plot_roc(X_test, y_test, rinv):
    auc_save_file = path / "roc_df" / f"auc-{rinv}.txt"
    test_predictions = model.predict(X_test).ravel()
    auc = roc_auc_score(y_test, test_predictions)
    with open(auc_save_file, "w") as f:
        f.write(str(auc))

    fpr, tpr, thresholds = roc_curve(y_test, test_predictions)
    background_efficiency = fpr
    signal_efficiency = tpr
    background_rejection = 1.0 - background_efficiency

    roc_df = pd.DataFrame(
        {
            "sig_eff": signal_efficiency,
            "bkg_eff": background_efficiency,
            "bkg_rej": background_rejection,
        }
    )
    roc_df.to_csv(f"roc_df/{rinv}.csv")
    # background_rejection = 1./fpr
    rinv_str = rinv.replace("p", ".")
    plt.plot(
        signal_efficiency,
        background_rejection,
        lw=2,
        label="$r_{inv} = %s$ ($AUC = %0.3f$)" % (rinv_str, auc),
    )
    plt.xlabel("Signal efficiency $(\epsilon_S)$")
    plt.ylabel("Background rejection $(1 - \epsilon_B)$")
    plt.title("ROC - CNN on Jet Images")
    plt.legend(loc="lower left")
    plt.savefig(path / "figures" / "cnn_roc.png")
    plt.savefig(path / "figures" / "cnn_roc.pdf")
    return auc
