"""Evaluate scaling relations using the information in BoundSubhaloProperties
"""

import h5py
from icecream import ic
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from time import time
import unyt

from plottery.plotutils import savefig, update_rcParams

import flaminkit as fkit

update_rcParams()
pd.options.display.float_format = "{:.2e}".format


def main():
    args = fkit.parse_args()
    halofile = fkit.halofile(args, info=True)

    # Load cluster galaxies for 10 random clusters
    cluster_galaxies = fkit.subhalos_in_clusters(
        halofile,
        cluster_mass_min=1e14,
        n=5,
        subhalo_cols=[
            "StellarMass",
            "GasMass",
            "GasTemperature",
            "AngularMomentumStars",
        ],
        so_cols="ComptonY",  # maybe we can reproduce the CHANCES selection with this?
        subhalo_min_mask={"StellarMass": 1e10},
        random_seed=1,
    )
    ic(cluster_galaxies)
    clusters = cluster_galaxies.loc[cluster_galaxies["Rank"] == 0]

    # find infalling groups
    ti = time()
    main_clusters, infallers = fkit.infalling_groups(
        halofile, clusters=main_clusters, group_mass_min=1e13, random_seed=9
    )
    tf = time()
    ic(
        main_clusters,
        infallers,
        tf - ti,
    )
    ti = time()
    galaxies_in_infallers = fkit.subhalos_in_clusters(
        halofile,
        clusters=infallers,
        subhalo_cols=["StellarMass"],
        subhalo_mass_min={"StellarMass": 1e10},
        random_seed=9,
    )
    galaxies_in_infallers = galaxies_in_infallers.loc[
        galaxies_in_infallers["StellarMass"] > 1e10
    ]
    tf = time()
    ic(galaxies_in_infallers, tf - ti)

    return

def plot_relation(args, xcol, ycol)

if __name__ == "__main__":
    main()
