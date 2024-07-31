import h5py
from icecream import ic
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import pandas as pd
from scipy.stats import binned_statistic
import swiftsimio as sw
from time import time
import unyt
from unyt import Mpc
import sys

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

    ## Load cluster galaxies for n random clusters
    cluster_galaxies = fkit.subhalos_in_clusters(
        halofile,
        cluster_mass_min=1e15,
        n=args.number,
        so_cols="ComptonY",
        subhalo_mask={"StellarMass": (1e10, np.inf)},
        random_seed=1,
    )
    ic(cluster_galaxies)
    main_clusters = cluster_galaxies.loc[cluster_galaxies["Rank"] == 0]
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
    (gas_particles, star_particles), (gas_particles_mask, star_particles_mask) = (
        fkit.particles_around(args.snapshot_file, xyz, dmax, ["gas", "stars"])
    )
    tf = time()

    ## Load galaxies in infalling groups around those clusters
    # not using for now
    # ti = time()
    # galaxies_in_infallers = fkit.subhalos_in_clusters(
    #     halofile,
    #     clusters=infallers,
    #     subhalo_cols=["StellarMass"],
    #     subhalo_mask={"StellarMass": (1e10, np.inf)},
    #     random_seed=9,
    # )
    # galaxies_in_infallers = galaxies_in_infallers.loc[
    #     galaxies_in_infallers["StellarMass"] > 1e10
    # ]
    # tf = time()
    # ic(galaxies_in_infallers, tf - ti)
    # (infaller_gas_particles, infaller_star_particles), (
    #     infaller_gas_particles_mask,
    #     infaller_star_particles_mask,
    # ) = fkit.particles_around(
    #     args.snapshot_file,
    #     galaxies_in_infallers[["x", "y", "z"]].to_numpy() * Mpc,
    #     dmax,
    #     ["gas", "stars"],
    # )

    ic(tf - ti, gas_particles)
    ic(star_particles.luminosities, gas_particles.electron_number_densities)
    for i in range(3):
        cluster_galaxies[f"MeanGasVelocity{i}"] = fkit.subhalo_particle_statistic(
            gas_particles.velocities[:, i], gas_particles_mask, np.average
        )
        cluster_galaxies[f"MeanStarsVelocity{i}"] = fkit.subhalo_particle_statistic(
            star_particles.velocities[:, i], star_particles_mask, np.average
        )
        cluster_galaxies[f"eDensityWeightedGasVelocity{i}"] = (
            fkit.subhalo_particle_statistic(
                gas_particles.velocities[:, i],
                gas_particles_mask,
                np.average,
                weights=gas_particles.electron_number_densities.ndarray_view(),
            )
        )
        cluster_galaxies[f"LuminosityWeightedStarsVelocity{i}"] = (
            fkit.subhalo_particle_statistic(
                star_particles.velocities[:, i],
                star_particles_mask,
                np.average,
                weights=star_particles.luminosities.GAMA_r.ndarray_view(),
            )
        )
        # # load same columns for galaxies in infallinng groups
        # galaxies_in_infallers[f"MeanGasVelocity{i}"] = fkit.subhalo_particle_statistic(
        #     infaller_gas_particles.velocities[:, i],
        #     infaller_gas_particles_mask,
        #     np.average,
        # )
        # galaxies_in_infallers[f"MeanStarsVelocity{i}"] = (
        #     fkit.subhalo_particle_statistic(
        #         infaller_star_particles.velocities[:, i],
        #         infaller_star_particles_mask,
        #         np.average,
        #     )
        # )
        # galaxies_in_infallers[f"eDensityWeightedGasVelocity{i}"] = (
        #     fkit.subhalo_particle_statistic(
        #         infaller_gas_particles.velocities[:, i],
        #         infaller_gas_particles_mask,
        #         np.average,
        #         weights=infaller_gas_particles.electron_number_densities.ndarray_view(),
        #     )
        # )
        # galaxies_in_infallers[f"LuminosityWeightedStarsVelocity{i}"] = (
        #     fkit.subhalo_particle_statistic(
        #         infaller_star_particles.velocities[:, i],
        #         infaller_star_particles_mask,
        #         np.average,
        #         weights=infaller_star_particles.luminosities.GAMA_r.ndarray_view(),
        #     )
        # )
    ic(
        cluster_galaxies[
            [
                "GasMass",
                "StellarMass",
                "MeanGasVelocity0",
                "eDensityWeightedGasVelocity0",
            ]
        ],
    )
    ic(cluster_galaxies.dtypes)
    ic(
        cluster_galaxies[
            ["LuminosityWeightedStarsVelocity0", "eDensityWeightedGasVelocity0"]
        ].describe()
    )

    # maybe I can use galaxy_scalings.plot for this - should move that to fkit
    # vbins = np.logspace(1, 4, 41)
    vbins = np.linspace(-2000, 2000, 51)
    hist2d = np.histogram2d(
        cluster_galaxies["LuminosityWeightedStarsVelocity0"],
        cluster_galaxies["eDensityWeightedGasVelocity0"],
        vbins,
    )[0]
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    im = ax.pcolormesh(vbins, vbins, hist2d.T, cmap="magma_r", norm=LogNorm())
    plt.colorbar(im, ax=ax)
    ax.plot(vbins, vbins, "k-", lw=2)
    ax.set(
        xlabel="LuminosityWeightedStarsVelocity0",
        ylabel="eDensityWeightedGasVelocity0",
        # xscale="log",
        # yscale="log",
    )
    output = "plots/testing/vstars_vgas.png"
    savefig(output, fig=fig, tight=False)
    return

    # must make an array to get floats, otherwise I get object
    cluster_galaxies["ComptonYSpherical"] = np.array(
        [gas_particles.compton_yparameters[p].sum() for p in gas_particles_mask]
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
    infaller_gas_particles, infaller_gas_particles_mask = fkit.particles_around(
        args.snapshot_file,
        galaxies_in_infallers[["x", "y", "z"]].to_numpy() * Mpc,
        dmax,
        ["gas", "stars"],
    )
    tf = time()
    ic(
        tf - ti,
        galaxies_[["GasMass", "StellarMass", "GasVelocities0", "GasVelocities1"]],
    )
    ti = time()
    # gv = [gas_particles.velocities[p] for p in gas_particles_mask]
    for i in range(3):
        cluster_galaxies[f"GasVelocities{i}"] = []
        for p in gas_particles_mask:
            gv = gas_particles.velocities[p]
            cluster_galaxies[f"GasVelocities{i}"].extend(gv[:, i])
    tf = time()
    ic(
        tf - ti,
        cluster_galaxies[
            ["GasMass", "StellarMass", "GasVelocities0", "GasVelocities1"]
        ],
    )
    ic(infaller_gas_particles, tf - ti)
    galaxies_in_infallers["ComptonYSpherical"] = np.array(
        [
            infaller_gas_particles.compton_yparameters[p].sum()
            for p in infaller_gas_particles_mask
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
