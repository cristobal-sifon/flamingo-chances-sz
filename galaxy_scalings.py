"""Evaluate scaling relations using the information in BoundSubhaloProperties
"""

import h5py
from icecream import ic
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import binned_statistic
import seaborn as sns
from time import time
import unyt

from plottery.plotutils import savefig, update_rcParams

import flaminkit as fkit
from flaminkit.plotting import get_axlabel

update_rcParams()
pd.options.display.float_format = "{:.2e}".format
sns.set_style("white")


def main():
    args = fkit.parse_args(
        [
            (
                "--ngal",
                dict(
                    default=None, type=int, help="Number of galaxies (for quick runs)"
                ),
            ),
            (
                "--ncl",
                dict(
                    default=None,
                    type=int,
                    help="Number of main clusters (for quick runs)",
                ),
            ),
            (
                "--min-mass-cluster",
                dict(default=1e14, type=float, help="Minimum cluster mass"),
            ),
            (
                "--min-mass-infall-group",
                dict(
                    default=1e13, type=float, help="Minimum mass for infalling groups"
                ),
            ),
        ]
    )
    halofile = fkit.halofile(args, info=True)

    # Load cluster galaxies for 10 random clusters
    cluster_galaxies = fkit.subhalos_in_clusters(
        halofile,
        cluster_mass_min=args.min_mass_cluster,
        n=args.ncl,
        subhalo_cols=[
            "StellarMass",
            "GasMass",
            "GasTemperature",
        ],
        so_cols=[
            "ComptonY",
            "TotalMass",
        ],  # maybe we can reproduce the CHANCES selection with this?
        subhalo_mask={"StellarMass": (1e10, np.inf)},
        random_seed=args.seed,
    )
    ic(cluster_galaxies)
    ic(cluster_galaxies.loc[cluster_galaxies["GasTemperature"] > 0])
    clusters = cluster_galaxies.loc[cluster_galaxies["Rank"] == 0]
    ic(clusters, clusters.shape)

    # find infalling groups
    ti = time()
    main_clusters, infallers = fkit.infalling_groups(
        halofile,
        clusters=clusters,
        group_mass_min=args.min_mass_infall_group,
        random_seed=9,
    )
    tf = time()
    ic(
        main_clusters,
        infallers,
        tf - ti,
    )
    galaxies_in_mains = cluster_galaxies.merge(
        main_clusters["HostHaloId"], on="HostHaloId", how="right"
    )
    ti = time()
    # galaxies_in_mains = fkit.subhalos_in_clusters(
    #     halofile,
    #     clusters=main_clusters,
    #     subhalo_cols=[
    #         "StellarMass",
    #         "GasMass",
    #         "GasTemperature",
    #     ],
    #     so_cols=[
    #         "StellarMass",
    #         "GasMass",
    #         "GasTemperature",
    #     ],
    #     subhalo_mask={"StellarMass": (1e10, np.inf)},
    #     random_seed=args.seed,
    # )
    galaxies_in_infallers = fkit.subhalos_in_clusters(
        halofile,
        clusters=infallers,
        subhalo_cols=[
            "StellarMass",
            "GasMass",
            "GasTemperature",
        ],
        so_cols=[
            "StellarMass",
            "GasMass",
            "GasTemperature",
        ],
        subhalo_mask={"StellarMass": (1e10, np.inf)},
        random_seed=args.seed,
    )
    galaxies_in_infallers = galaxies_in_infallers.loc[
        galaxies_in_infallers["StellarMass"] > 1e10
    ]
    tf = time()
    ic(galaxies_in_mains, galaxies_in_infallers, tf - ti)
    ic(np.sort(galaxies_in_mains.columns))
    ic(galaxies_in_infallers.loc[galaxies_in_infallers["GasTemperature"] > 0])

    # stellar mass vs temperature
    output = "plots/testing/mstar_temp.png"
    fig, ax = scatter(
        args,
        galaxies_in_mains,
        "StellarMass",
        "GasTemperature",
        label="Galaxies in main clusters",
        marker="x",
        s=25,
    )
    scatter(
        args,
        galaxies_in_infallers,
        "StellarMass",
        "GasTemperature",
        ax=ax,
        labels=False,
        label="Galaxies in infallers",
        marker="+",
        s=25,
        xscale="log",
        yscale="log",
    )
    ax.legend(fontsize=14, loc="upper left", frameon=False)
    savefig(output, fig=fig, tight=False)

    # line plot the sns way
    # this takes quite a long while for all M>1e14 clusters (500k galaxies)
    galaxies_in_mains["ClusterType"] = "Main"
    galaxies_in_infallers["ClusterType"] = "Infaller"
    # galaxies = galaxies_in_mains.join(galaxies_in_infallers, how="outer")
    galaxies = pd.concat(
        [galaxies_in_mains, galaxies_in_infallers], ignore_index=True, axis=0
    )
    # galaxies.sort_values(["ClusterType", "StellarMass"])
    # galaxies["StellarMass"] = galaxies["StellarMass"].rolling()
    ic(galaxies)
    fig, ax = plot(
        args,
        galaxies,
        "StellarMass",
        "GasTemperature",
        mask=galaxies["GasMass"] > 0,
        plottype="binned_statistic",
        hue="ClusterType",
        estimator="median",
        # errorbar=("pi", 99),
        errorbar="sd",
        bins=np.logspace(10, 13, 25),
        xscale="log",
        yscale="log",
        output="plots/testing/mstar_temp_median.png",
    )
    fig, ax = plot(
        args,
        galaxies,
        "StellarMass",
        "GasTemperature",
        mask=galaxies["GasMass"] > 0,
        plottype="sns.lineplot",
        hue="ClusterType",
        estimator="median",
        # errorbar=("pi", 99),
        errorbar="sd",
        rolling=10000,
        xscale="log",
        yscale="log",
        output="plots/testing/mstar_temp_median_sns.png",
    )

    # given that galaxies in mains and in infallers follow the same relation,
    # let's look at gas temperature vs m200

    return


def plot(
    args,
    tbl,
    xcol,
    ycol,
    mask=None,
    plottype="scatter",
    bins=None,
    estimator=None,
    # can use these with non-sns plottypes anyway! Just need to write a few more lines
    hue=None,
    rolling=None,
    labels=True,
    fig=None,
    ax=None,
    output=None,
    xscale="linear",
    yscale="linear",
    **kwargs,
):
    """We might move this to flaminkit.plotting!

    Remove sns plotting types, add "means", "medians", and options for uncertainties a la sns.lineplot (which however doesn't quite work)
    """
    if fig is None:
        fig = plt.figure(figsize=(6, 5), layout="constrained")
    if ax is None:
        ax = fig.add_subplot(111)
    if mask is None:
        mask = np.ones(tbl[xcol].size, dtype=bool)
    if mask.dtype == bool:
        tbl = tbl.loc[mask]
    else:
        tbl = tbl.iloc[mask]
    if plottype.startswith("sns") and rolling is not None:
        tbl = tbl.sort_values(xcol)
        ic(tbl[[xcol, ycol]])
        tbl[xcol] = tbl[xcol].rolling(rolling, min_periods=1).median()
        ic(tbl[[xcol, ycol]])
    if plottype == "plot":
        ax.plot(tbl[xcol], tbl[ycol], **kwargs)
    elif plottype == "scatter":
        ax.scatter(tbl[xcol], tbl[ycol], **kwargs)
    elif plottype == "binned_statistic":
        # hue not yet implemented
        stat, bin_edges, binnumber = binned_statistic(
            tbl[xcol], tbl[ycol], estimator, bins
        )
        if xscale == "log":
            bin_edges = np.log10(bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        if xscale == "log":
            bin_centers = 10**bin_centers
        ax.plot(bin_centers, stat, **kwargs)
    elif plottype[:4] == "sns.":
        f = getattr(sns, plottype[4:])
        if "rolling" not in kwargs:
            kwargs["rolling"] = bins
        f(tbl, x=xcol, y=ycol, ax=ax, **kwargs)
    else:
        raise ValueError(f"plottype {plottype} not recognized")
    if labels:
        ax.set(xlabel=get_axlabel(xcol), ylabel=get_axlabel(ycol))
    ax.set(xscale=xscale, yscale=yscale)
    if output:
        savefig(output, fig=fig, tight=False)
    return fig, ax


def scatter(*args, **kwargs):
    if "plottype" in kwargs:
        kwargs.pop("plottype")
    return plot(*args, plottype="scatter", **kwargs)


if __name__ == "__main__":
    main()
