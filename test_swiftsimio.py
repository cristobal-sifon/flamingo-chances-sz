import h5py
from icecream import ic
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import swiftsimio as sw
from time import time
import unyt
from unyt import Mpc

from plottery.plotutils import savefig, update_rcParams

import flaminkit as fkit

update_rcParams()
pd.options.display.float_format = "{:.3e}".format


def main():
    args = fkit.parse_args(
        [
            (
                "-n",
                dict(
                    dest="number",
                    default=None,
                    type=int,
                    help="Number of clusters (for quick runs)",
                ),
            )
        ]
    )

    ## metadata
    part = sw.load(args.snapshot_file)
    meta = part.metadata
    ic(meta.header)
    ic(meta.present_particle_types, meta.present_particle_names)
    ic(meta.gas_properties.field_names)
    ic(meta.gas_properties.field_units)
    ic(meta.stars_properties.field_names)

    ## Load cluster subhalos
    snap = f"{args.snapshot:04d}"
    halofile = os.path.join(args.path["SOAP-HBT"], f"halo_properties_{snap}.hdf5")
    # to have some info handy
    ic(halofile)
    with h5py.File(halofile) as f:
        ic(f.keys())
        for key in f.keys():
            ic(key)
            for gr in f[key].items():
                ic(gr)
                if gr[0] in ("HBTplus", "200_crit", "100kpc"):
                    for i, subgr in enumerate(gr[1].items()):
                        ic(subgr)
                        if subgr[0] in ("projx",):
                            for i, subsubgr in enumerate(subgr[1].items()):
                                ic(subsubgr)
            print()

    # Load cluster galaxies for 10 random clusters
    cluster_galaxies = fkit.subhalos_in_clusters(
        halofile,
        cluster_mass_min=1e14,
        n=args.number,
        so_cols="ComptonY",
        subhalo_mask={"StellarMass": (1e10, np.inf)},
        random_seed=1,
    )
    ic(cluster_galaxies)
    main_clusters = cluster_galaxies.loc[cluster_galaxies["Rank"] == 0]

    # cut in stellar mass (first add stellar mass and center of
    # mass which we will use below)
    with h5py.File(halofile) as file:
        gals = pd.DataFrame(
            {
                "TrackId": file.get("InputHalos/HBTplus/TrackId")[()],
                "StellarMass": file.get("BoundSubhaloProperties/StellarMass")[()],
                "GasMass": file.get("BoundSubhaloProperties/GasMass")[()],
            }
        )
        cluster_galaxies = gals.merge(cluster_galaxies, on="TrackId", how="right")
        del gals
    ic(cluster_galaxies)

    # find particles associated with each selected cluster galaxy
    xyz = cluster_galaxies[["x", "y", "z"]].to_numpy() * Mpc
    dmax = 0.1 * Mpc
    ti = time()
    gas_particles, gas_particles_around = fkit.particles_around(
        args.snapshot_file, xyz, dmax, "gas"
    )
    tf = time()
    ic(gas_particles, tf - ti)
    # must make an array to get floats, otherwise I get object
    cluster_galaxies["ComptonYSpherical"] = np.array(
        [gas_particles.compton_yparameters[p].sum() for p in gas_particles_around]
    )
    ic(cluster_galaxies[["GasMass", "StellarMass", "ComptonYSpherical"]])
    ic(cluster_galaxies.dtypes)
    ic((cluster_galaxies["GasMass"] > 0).sum())

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
        subhalo_mask={"StellarMass": (1e10, np.inf)},
        random_seed=9,
    )
    galaxies_in_infallers = galaxies_in_infallers.loc[
        galaxies_in_infallers["StellarMass"] > 1e10
    ]
    tf = time()
    ic(galaxies_in_infallers, tf - ti)
    ti = time()
    infaller_gas_particles, infaller_gas_particles_around = fkit.particles_around(
        args.snapshot_file,
        galaxies_in_infallers[["x", "y", "z"]].to_numpy() * Mpc,
        dmax,
        "gas",
    )
    tf = time()
    ic(infaller_gas_particles, tf - ti)
    galaxies_in_infallers["ComptonYSpherical"] = np.array(
        [
            infaller_gas_particles.compton_yparameters[p].sum()
            for p in infaller_gas_particles_around
        ]
    )
    ic(galaxies_in_infallers, tf - ti)
    ic(galaxies_in_infallers.dtypes)

    # plot compton y vs stellar mass or distance to main or something
    fig, ax = plt.subplots(layout="constrained")
    mbins = np.logspace(10, 13, 20)
    ybins = np.logspace(-11, -5, 20)
    h2d_infall = np.histogram2d(
        galaxies_in_infallers["ComptonYSpherical"],
        galaxies_in_infallers["StellarMass"],
        (ybins, mbins),
    )[0]
    h2d_main = np.histogram2d(
        cluster_galaxies["ComptonYSpherical"],
        cluster_galaxies["StellarMass"],
        (ybins, mbins),
    )[0]
    # remember this doesn't work for log axes, use pcolormesh instead
    ax.pcolormesh(mbins, ybins, h2d_infall, cmap="BuPu")
    y0 = 10 ** ((np.log10(ybins[1:]) + np.log10(ybins[:-1])) / 2)
    m0 = 10 ** ((np.log10(mbins[1:]) + np.log10(mbins[:-1])) / 2)
    ax.contour(m0, y0, h2d_main, colors="k")
    ax.set(xscale="log", yscale="log", xlabel="stellar mass", ylabel="Compton Y")
    output = "plots/testing/mstar_y_infallers.png"
    savefig(output, fig=fig, tight=False)

    return


if __name__ == "__main__":
    main()
