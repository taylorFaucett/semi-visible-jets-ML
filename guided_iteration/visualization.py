#! /home/tfaucett/miniconda3/envs/tf/bin/python
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from matplotlib.ticker import MaxNLocator
import pandas as pd
import glob
import seaborn as sns
import numpy as np
import matplotlib.image as image
import pathlib
import h5py

path = pathlib.Path.cwd()
plt.rcParams["font.family"] = "serif"

np.seterr(divide = 'ignore') 

def norm(x):
    normed = (x - min(x)) / (max(x) - min(x))
    return normed


class visualization:
    def __init__(self, it_dir, ix):
        self.it_dir = it_dir
        self.ix = ix

    def nn_design_heatmap(self):
        plt.figure(figsize=(6, 6))
        nn_stats = pd.read_csv(f"{self.it_dir}/p{self.ix}/nn_stats.csv")
        nn_stats = nn_stats[nn_stats.auc >= 0.75]
        max_idx = nn_stats.auc.idxmax()
        df = nn_stats.pivot_table(index="nodes", columns="layers", values="auc")
        ax = sns.heatmap(df, annot=True, fmt=".3g")
        maxs = nn_stats.iloc[max_idx]
        plt.title(
            f"Max AUC = {maxs.auc:.3f} ({maxs.layers:.0f} layers | {maxs.nodes:.0f} hidden units)"
        )
        plt.tight_layout()
        plt.savefig(f"{self.it_dir}/p{self.ix}/nn_stats_plot.png")
        plt.clf()

    def performance_table(self):
        df = pd.read_csv(f"{self.it_dir}/selected_efps.csv")

        # split the efp info for performance table
        graphs, kappas, betas = [], [], []
        efps = df.efp.values
        for ix, efp in enumerate(efps):
            if ix == 0:
                graphs.append(np.nan)
                kappas.append(np.nan)
                betas.append(np.nan)
            else:
                efp_parts = efp.split("_")
                graphs.append(efp_parts[2])
                kappas.append(efp_parts[4])
                betas.append(efp_parts[-1])
        efp_parts_df = pd.DataFrame({"graph": graphs, "kappa": kappas, "beta": betas})
        df = pd.concat([df, efp_parts_df], axis=1)
        df = df[{"graph", "kappa", "beta", "auc", "ado"}]

        column_order = ["graph", "kappa", "beta", "auc", "ado"]

        df = df.reindex(columns=column_order)

        fig, ax = plt.subplots(1)
        # hide axes
        fig.patch.set_visible(False)
        ax.axis("off")
        ax.axis("tight")

        ax.table(cellText=df.values, colLabels=df.columns, loc="center")
        plt.savefig(f"{self.it_dir}/performance_table.png")
        plt.clf()

    def performance_plot(self):
        # Determine the number of completed passes
        ll_benchmark = 0.8475
        df = pd.read_csv(f"{self.it_dir}/selected_efps.csv")
        hl_only = df.auc.values[0]
        efps = df.efp.values

        # Get pass data
        pass_ix = df.index.values
        auc_ix = df.auc.values
        ado_ix = df.ado.values

        x_min = pass_ix[0]
        x_max = pass_ix[-1]

        # Initialize plot
        fig, (ax0, ax1, ax2) = plt.subplots(
            3, sharex=True, figsize=(10, 8), gridspec_kw={"height_ratios": [4, 1, 1]}
        )

        ax0.hlines(
            ll_benchmark,
            x_min,
            x_max,
            label=f"LL $[AUC={ll_benchmark:.4}]$",
            color="r",
            linestyle="dashed",
        )

        ax0.hlines(
            hl_only,
            x_min,
            x_max,
            label=f"HL $[AUC={hl_only:.4}]$",
            color="k",
            linestyle="dashed",
        )

        ax0.plot(
            pass_ix,
            auc_ix,
            color="b",
            label=f"Black-box guided DNN(HL + nEFP) [AUC={max(auc_ix):.4}]",
        )
        ax0.set_ylabel("AUC")
        ax0.legend(bbox_to_anchor=(1.0, 0.05), loc="lower right", ncol=1, fontsize=12)

        ax1.plot(pass_ix, ado_ix, color="r")
        ax1.fill_between(pass_ix, ado_ix, 0.5, color="r", alpha=0.1)
        ax1.set_ylim([0.5, 1.0])
        ax1.set_ylabel("$ADO(X_i, LL_i)$")

        plt.xlim([0, x_max])
        plt.xlabel("# of EFPs")

        # plt.xticks(pass_ix)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        for efp_ix, efp in enumerate(efps):
            if efp_ix >= 1:
                # Add the graph to the plot
                # gp = efp.split("_")
                gp = efp.split("EFP_")[-1]
                ndk = gp.split("_k")[0]
                kappa = gp.split("_")[-3]
                beta = gp.split("_")[-1]
                graph_file = path.parent / "graphs" / "png" / f"{ndk}.png"
                arr_graph = mpimg.imread(graph_file)
                imagebox = OffsetImage(arr_graph, zoom=0.25)
                xlim = ax2.get_xlim()
                ylim = ax2.get_ylim()

                xpos = efp_ix
                ypos = 0.55
                xy = (xpos, ypos)
                ab = AnnotationBbox(imagebox, xy)
                ax2.add_artist(ab)

                xy = (xpos, 0)
                textarea = TextArea(f"($\\kappa$={kappa}, $\\beta$={beta})")
                ab = AnnotationBbox(textarea, xy)
                ax2.add_artist(ab)
                ax2.axis("off")

                plt.draw()
        fig.subplots_adjust(hspace=0.1)
        # plt.tight_layout()
        plt.savefig(f"{self.it_dir}/performance_plot.png")
        plt.savefig(f"{self.it_dir}/performance_plot.pdf")
        plt.clf()

    def dif_order_hist_plots(self):
        # Get indices for run
        df = pd.read_feather(f"{self.it_dir}/p{self.ix}/dif_order.feather")
        idx0 = df.idx0.values
        idx1 = df.idx1.values

        # Remove any duplicates
        idx0 = list(set(idx0))
        idx1 = list(set(idx1))

        # Plot settings
        n_bins = 50
        lw = 2
        e1 = edgecolor = (0.2, 0.2, 1, 1)
        c1 = (0.2, 0.2, 1, 0.1)

        e2 = edgecolor = (1, 0.5, 0, 1)
        c2 = (1, 0.5, 0, 0.1)

        fig_x = 6
        fig_y = 8
        alpha = 0.1

        # Initialize plot
        fig, axs = plt.subplots(2, sharex=True, figsize=(fig_x, fig_y))

        # Load EFP data from run
        efp_max = (
            pd.read_csv(f"{self.it_dir}/p{self.ix}/dif_order_ado_comparison.csv")
            .iloc[0]
            .efp
        )
        efps = h5py.File(path.parent / "data" / "efps.h5", "r")
        efp_data, targets = efps["efps"][efp_max][:], efps["targets"][:] #pd.read_feather(f"{efp_dir}/{efp_max}.feather")
        efp_data = pd.DataFrame({"features":efp_data, "targets":targets})
        
        # Select dif-order indices
        dif_X0 = np.log10(efp_data.iloc[idx0]["features"].values)
        dif_X1 = np.log10(efp_data.iloc[idx1]["features"].values)

        # Get complete EFP and group by signal/background
        efp_grp = efp_data.groupby(["targets"])
        X0 = np.log10(efp_grp.get_group(0).features.values)
        X1 = np.log10(efp_grp.get_group(1).features.values)

        upper_bound = max(max(X0), max(X1))
        lower_bound = min(min(X0), min(X1))
        if lower_bound == -np.inf:
            lower_bound = -12
        bins = np.linspace(lower_bound, upper_bound, n_bins)

        # Plot histograms
        axs[0].hist(
            X0,
            bins=bins,
            density=True,
            label="Signal in space $EFP_{i}$",
            histtype="stepfilled",
            linestyle="-",
            linewidth=lw,
            edgecolor=e1,
            fc=c1,
        )
        axs[0].hist(
            X1,
            bins=bins,
            density=True,
            label="Background in space $EFP_{i}$",
            histtype="stepfilled",
            linestyle="-",
            linewidth=lw,
            edgecolor=e2,
            fc=c2,
        )

        axs[1].hist(
            dif_X0,
            bins=bins,
            density=True,
            label="Signal in subspace $X_{i}$",
            histtype="stepfilled",
            linestyle="-",
            linewidth=lw,
            edgecolor=e1,
            fc=c1,
        )
        axs[1].hist(
            dif_X1,
            bins=bins,
            density=True,
            label="Background in subspace $X_{i}$",
            histtype="stepfilled",
            linestyle="-",
            linewidth=lw,
            edgecolor=e2,
            fc=c2,
        )

        # Increase font
        axs[0].legend(fontsize=12, loc="upper left")
        axs[1].legend(fontsize=12, loc="upper left")

        axs[0].set_xlim(left=lower_bound, right=upper_bound)
        axs[1].set_xlim(left=lower_bound, right=upper_bound)

        # Plot labels
        axs[1].set_xlabel("$\log_{10}$ [%s]" % efp_max)
        axs[0].set_ylabel("Density")
        axs[1].set_ylabel("Density")
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.subplots_adjust(hspace=0.02)

        # Add the graph to the plot
        gp = efp_max.split("EFP_")[-1]
        ndk = gp.split("_k")[0]
        kappa = gp.split("_")[-3]
        beta = gp.split("_")[-1]
        graph_label = f"{ndk}"
        graph_file = path.parent / "graphs" / "png" / f"{ndk}.png"
        arr_graph = mpimg.imread(graph_file)
        imagebox = OffsetImage(arr_graph, zoom=0.2)
        xlim = axs[0].get_xlim()
        ylim = axs[0].get_ylim()

        xpos = bins[int(n_bins / 6)]
        ypos = ((ylim[0] + ylim[1]) / 2.0) * 1.2
        xy = (xpos, ypos)
        ab = AnnotationBbox(imagebox, xy)
        axs[0].add_artist(ab)

        xy = (xpos, ypos * 0.7)
        textarea = TextArea(f"($\\kappa$={kappa}, $\\beta$={beta})")
        ab = AnnotationBbox(textarea, xy)
        axs[0].add_artist(ab)

        plt.draw()

        # Save plot
        plt.tight_layout()
        plt.savefig(f"{self.it_dir}/p{self.ix}/dif_order_plot.png")
        plt.savefig(f"{self.it_dir}/p{self.ix}/dif_order_plot.pdf")
        plt.clf()

    def clear_viz(self):
        plt.close("all")
