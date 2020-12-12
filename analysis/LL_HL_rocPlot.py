import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import warnings
import scipy.stats

path = pathlib.Path.cwd()
warnings.filterwarnings("ignore")

colors = {"HL": "r", "LL": "b"}


def mean_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


def average_arr(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[: len(l), idx] = l
    n = len(arr)
    m, se = np.mean(arr, axis=-1), scipy.stats.sem(arr, axis=-1)
    h = se * scipy.stats.t.ppf((1 + 0.95) / 2.0, n - 1)
    return m, h


rinvs = ["0p0", "0p3", "1p0"]
HL_LL = ["HL", "LL"]

fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for rix, rinv in enumerate(rinvs):
    for data_type in HL_LL:
        # Get AUC values and calculate mean/95% confidence interval
        auc_file = (
            path.parent
            / f"{data_type}_training"
            / "bootstrap_results"
            / rinv
            / "aucs.csv"
        )
        aucs = pd.read_csv(auc_file).auc
        auc_avg, auc_ci = mean_ci(aucs)

        # Get roc curve data
        roc_path = (
            path.parent / f"{data_type}_training" / "bootstrap_results" / rinv / "roc"
        )
        sig_effs, bkg_effs, bkg_rejs = [], [], []
        for roc_file in roc_path.glob("*"):
            roc_df = pd.read_csv(roc_file, index_col=0)
            sig_effs.append(roc_df.sig_eff.values)
            bkg_effs.append(roc_df.bkg_eff.values)
            bkg_rejs.append(1.0 / roc_df.bkg_eff.values)

        # Find the mean of the roc curve data arrays
        mean_sig_eff, ci_sig_eff = average_arr(sig_effs)
        mean_bkg_eff, ci_bkg_eff = average_arr(bkg_effs)
        mean_bkg_rej, ci_bkg_rej = average_arr(bkg_rejs)

        # Reduce the array size (repeatedly split in half by taking every other element)
        while len(x) > 1000:
            mean_sig_eff, ci_sig_eff = mean_sig_eff[::2], ci_sig_eff[::2]
            mean_bkg_eff, ci_bkg_eff = mean_bkg_eff[::2], ci_bkg_eff[::2]
            mean_bkg_rej, ci_bkg_rej = mean_bkg_rej[::2], ci_bkg_rej[::2]

        # Define plot labels
        label = "%s (N=%s) $\\left(\mathrm{AUC} = %0.4f \pm %0.4f \\right)$" % (
            data_type,
            len(aucs),
            auc_avg,
            auc_ci,
        )

        # Plot Title
        ax[rix].set_title("$r_{inv} = %s$" % rinv.replace("p", "."))

        # Curve and CI fill_between
        ax[rix].fill_between(
            mean_sig_eff,
            mean_bkg_rej - ci_bkg_rej,
            mean_bkg_rej + ci_bkg_rej,
            color=colors[data_type],
            alpha=0.25,
        )
        ax[rix].plot(
            mean_sig_eff,
            mean_bkg_rej,
            lw=2,
            color=colors[data_type],
            alpha=1,
            label=label,
        )

        # X and Y labels
        ax[rix].set_xlabel("Signal efficiency $(\epsilon_S)$")
        ax[0].set_ylabel("Background rejection $(1 / \epsilon_B)$")

        # Plot details
        ax[rix].set_yscale("log")
        ax[rix].set_xlim(left=0, right=1)
        ax[rix].set_ylim(bottom=1, top=1e5)
        ax[rix].legend(loc="lower left", fontsize="12")

# Finalize and save plots
plt.subplots_adjust(wspace=0.1)
plt.tight_layout()
plt.savefig(path.parent / "figures" / "bootstrap_roc.png")
plt.savefig(path.parent / "figures" / "bootstrap_roc.pdf")
