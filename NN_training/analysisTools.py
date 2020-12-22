import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import warnings
import scipy.stats

path = pathlib.Path.cwd()
warnings.filterwarnings("ignore")

class stats:
    def mean_ci(self, data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
        return m, h
    
class smoothROC:
    def average_arr(self, arrs, confidence=0.95):
        lens = [len(i) for i in arrs]
        arr = np.ma.empty((np.max(lens), len(arrs)))
        arr.mask = True
        for idx, l in enumerate(arrs):
            arr[: len(l), idx] = l
        n = len(arr)
        m, se = np.mean(arr, axis=-1), scipy.stats.sem(arr, axis=-1)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
        return m, h

    def plot(self):
        plt.rcParams["font.family"] = "serif"
        matplotlib.rcParams.update({"font.size": 12})

        colors = {"HL": "tab:orange", "LL": "tab:blue"}
        fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        for run_type in ("LL", "HL"):
            for rix, rinv in enumerate(["0p0", "0p3", "1p0"]):
                # Get AUC values and calculate mean/95% confidence interval
                auc_file = path / "bootstrap_results" / run_type / rinv / "aucs.csv"
                try:
                    aucs = pd.read_csv(auc_file).auc
                except:
                    break
                auc_avg, auc_ci = stats.mean_ci(aucs)

                # Get roc curve data
                roc_path = path / "bootstrap_results" / run_type / rinv / "roc"
                sig_effs, bkg_effs, bkg_rejs = [], [], []
                x_vals = np.linspace(0, 1, 400)
                for roc_file in roc_path.glob("*"):
                    # get signal/background efficiency and rejection
                    roc_df = pd.read_csv(roc_file, index_col=0)
                    sig_eff = roc_df.sig_eff.values
                    bkg_eff = roc_df.bkg_eff.values
                    bkg_rej = 1.0 / bkg_eff

                    # Interp to standardize data size
                    bkg_eff = np.interp(x_vals, sig_eff, bkg_eff)
                    bkg_rej = np.interp(x_vals, sig_eff, bkg_rej)

                    sig_effs.append(x_vals)
                    bkg_effs.append(bkg_eff)
                    bkg_rejs.append(bkg_rej)

                # Find the mean of the roc curve data arrays
                mean_sig_eff, ci_sig_eff = self.average_arr(sig_effs)
                mean_bkg_eff, ci_bkg_eff = self.average_arr(bkg_effs)
                mean_bkg_rej, ci_bkg_rej = self.average_arr(bkg_rejs)

                # Define plot labels
                label = "%s (N=%s) $\\left(\mathrm{AUC} = %0.4f \pm %0.4f \\right)$" % (
                    run_type,
                    len(aucs),
                    auc_avg,
                    auc_ci,
                )
                # label = '%s $\\left(\mathrm{AUC} = %0.4f \pm %0.4f \\right)$' %(run_type, auc_avg, auc_ci)

                # Plot Title
                ax[rix].set_title("$r_{inv} = %s$" % rinv.replace("p", "."))

                # Curve and CI fill_between
                ax[rix].fill_between(
                    mean_sig_eff,
                    mean_bkg_rej - ci_bkg_rej,
                    mean_bkg_rej + ci_bkg_rej,
                    color=colors[run_type],
                    alpha=0.25,
                )
                ax[rix].plot(
                    mean_sig_eff,
                    mean_bkg_rej,
                    lw=2,
                    color=colors[run_type],
                    alpha=1,
                    label=label,
                )

                # X and Y labels
                ax[rix].set_xlabel("Signal efficiency $(\epsilon_S)$")
                ax[0].set_ylabel("Background rejection $(1 / \epsilon_B)$")

                # Plot details
                ax[rix].set_yscale("log")
                ax[rix].set_xlim(left=0, right=1)
                ax[rix].set_ylim(bottom=1, top=6e5)
                ax[rix].legend(
                    loc="upper right",
                    fontsize="12",
                    framealpha=1,
                    shadow=True,
                    borderpad=0.5,
                    frameon=True,
                )

        # Finalize and save plots
        plt.subplots_adjust(wspace=0.05)
        plt.tight_layout()
        plt.savefig(path / "figures" / "mean_roc.png")
        plt.savefig(path / "figures" / "mean_roc.pdf")


class get_sherpa_results:
    def __init__(self, run_type, rinv, objective="Objective"):
        self.run_type = run_type
        self.rinv = rinv
        self.objective = objective

    def best(self, objective="Objective"):
        # Use sherpa parameters (same as used in bootstrapping)
        sherpa_file = (
            path / "sherpa_results" / self.run_type / self.rinv / "results.csv"
        )
        sherpa_results = pd.read_csv(sherpa_file, index_col="Trial-ID").groupby(
            "Status"
        )
        sorted_results = sherpa_results.get_group("COMPLETED").sort_values(
            by=objective, ascending=False
        )
        best_result = sorted_results.iloc[0].to_dict()
        return best_result

    def all(self, objective="Objective"):
        # Use sherpa parameters (same as used in bootstrapping)
        sherpa_file = (
            path / "sherpa_results" / self.run_type / self.rinv / "results.csv"
        )
        sherpa_results = pd.read_csv(sherpa_file, index_col="Trial-ID").groupby(
            "Status"
        )
        sorted_results = sherpa_results.get_group("COMPLETED").sort_values(
            by=objective, ascending=False
        )
        return sorted_results

