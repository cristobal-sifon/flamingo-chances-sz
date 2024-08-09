from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
import h5py
from icecream import ic
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle
import numpy as np
import os
import pandas as pd
from scipy.stats import binned_statistic
import swiftsimio as sw
from swiftsimio.conversions import swift_cosmology_to_astropy
from time import time
import unyt
from unyt import Mpc
import sys
import warnings

from plottery.plotutils import savefig, update_rcParams

import flaminkit as fkit

update_rcParams()
pd.options.display.float_format = "{:.3e}".format
warnings.simplefilter("ignore", category=RuntimeWarning)


def main():
    args = fkit.parse_args()

    ## particle metadata
    part = sw.load(args.snapshot_file)
    meta = part.metadata
    ic(meta.header)
    ic(meta.present_particle_types, meta.present_particle_names)
    ic(meta.gas_properties.field_names)
    ic(meta.gas_properties.field_units)
    ic(meta.stars_properties.field_names)

    halofile = fkit.halofile(args)

    ## Load cluster galaxies for n random clusters
    cluster_galaxies = fkit.subhalos_in_clusters(
        halofile,
        cluster_mask={"SO/200_crit/TotalMass": (args.min_mass_cluster, np.inf)},
        n=args.ncl,
        so_cols="ComptonY",
        subhalo_mask={"StellarMass": (1e10, np.inf)},
        random_seed=args.seed,
    )
    ic(cluster_galaxies)
    main_clusters = cluster_galaxies.loc[cluster_galaxies["Rank"] == 0]
    # find infalling groups
    ti = time()
    main_clusters, infallers = fkit.infalling_groups(
        halofile,
        clusters=main_clusters,
        group_mass_min=args.min_mass_group,
        random_seed=args.seed,
    )
    tf = time()
    ic(main_clusters, infallers, tf - ti)

    plot_positions(args, main_clusters, infallers)

    # find particles associated with each selected cluster galaxy
    xyz = cluster_galaxies[["x", "y", "z"]].to_numpy() * Mpc
    dmax = 0.1 * Mpc
    ti = time()
    (gas_particles, star_particles), (gas_particles_mask, star_particles_mask) = (
        fkit.particles_around(
            args.snapshot_file, xyz, dmax, ["gas", "stars"], nthreads=args.nthreads
        )
    )
    tf = time()
    ic("gas and star particles", tf - ti)
    ic(star_particles.luminosities, gas_particles.electron_number_densities)
    # for testing:
    with open(
        f"testing/particles_around__{args.ncl}__{args.seed}__{args.nthreads}.txt", "w"
    ) as f:
        for m in star_particles_mask:
            mstr = " | ".join([f"{i:7d}" for i in m[:10]])
            print(mstr, file=f)

    ## Load galaxies in infalling groups around those clusters
    # not using for now
    ti = time()
    galaxies_in_infallers = fkit.subhalos_in_clusters(
        halofile,
        clusters=infallers,
        subhalo_cols=["StellarMass"],
        subhalo_mask={"StellarMass": (1e10, np.inf)},
        random_seed=args.seed,
    )
    galaxies_in_infallers = galaxies_in_infallers.loc[
        galaxies_in_infallers["StellarMass"] > 1e10
    ]
    tf = time()
    ic(galaxies_in_infallers, tf - ti)
    (infaller_gas_particles, infaller_star_particles), (
        infaller_gas_particles_mask,
        infaller_star_particles_mask,
    ) = fkit.particles_around(
        args.snapshot_file,
        galaxies_in_infallers[["x", "y", "z"]].to_numpy() * Mpc,
        dmax,
        ["gas", "stars"],
        nthreads=args.nthreads,
    )

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
        # load same columns for galaxies in infallinng groups
        galaxies_in_infallers[f"MeanGasVelocity{i}"] = fkit.subhalo_particle_statistic(
            infaller_gas_particles.velocities[:, i],
            infaller_gas_particles_mask,
            np.average,
        )
        galaxies_in_infallers[f"MeanStarsVelocity{i}"] = (
            fkit.subhalo_particle_statistic(
                infaller_star_particles.velocities[:, i],
                infaller_star_particles_mask,
                np.average,
            )
        )
        galaxies_in_infallers[f"eDensityWeightedGasVelocity{i}"] = (
            fkit.subhalo_particle_statistic(
                infaller_gas_particles.velocities[:, i],
                infaller_gas_particles_mask,
                np.average,
                weights=infaller_gas_particles.electron_number_densities.ndarray_view(),
            )
        )
        galaxies_in_infallers[f"LuminosityWeightedStarsVelocity{i}"] = (
            fkit.subhalo_particle_statistic(
                infaller_star_particles.velocities[:, i],
                infaller_star_particles_mask,
                np.average,
                weights=infaller_star_particles.luminosities.GAMA_r.ndarray_view(),
            )
        )
    ic(cluster_galaxies.dtypes)
    ic(
        cluster_galaxies[
            ["LuminosityWeightedStarsVelocity0", "eDensityWeightedGasVelocity0"]
        ].describe()
    )

    # maybe I can use galaxy_scalings.plot for this - should move that to fkit
    # vbins = np.logspace(1, 4, 41)
    # cluster galaxies
    vbins = np.linspace(-2000, 2000, 31)
    plot_vstars_vgas(
        args,
        cluster_galaxies,
        "LuminosityWeightedStarsVelocity0",
        "eDensityWeightedGasVelocity0",
        vbins,
    )
    # infaller galaxies
    vbins = np.linspace(-10000, 10000, 51)
    plot_vstars_vgas(
        args,
        galaxies_in_infallers,
        "LuminosityWeightedStarsVelocity0",
        "eDensityWeightedGasVelocity0",
        vbins,
    )
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
        halofile, clusters=main_clusters, group_mass_min=1e13, random_seed=args.seed
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


def plot_positions(args, main_clusters, infallers, cluster_detail=None, z_ref=0.05):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # there's no point in showing these in the left panel
    # axes[1].plot(infallers.CentreOfMass_x, infallers.CentreOfMass_y, "C1.")
    main_r200 = PatchCollection(
        main_clusters.apply(
            lambda cl: Circle(
                (cl["CentreOfMass_x"], cl["CentreOfMass_y"]), cl["SORadius"]
            ),
            axis=1,
        ),
        fc="none",
        ec="C0",
        lw=1,
        zorder=1,
    )
    axes[1].add_collection(main_r200)
    fmts = (",", "+")
    for i, (ax, fmt) in enumerate(zip(axes, fmts)):
        main_5r200 = PatchCollection(
            main_clusters.apply(
                lambda cl: Circle(
                    (cl["CentreOfMass_x"], cl["CentreOfMass_y"]),
                    5 * cl["SORadius"],
                ),
                axis=1,
            ),
            fc="none",
            ec="C0",
            lw=1 + i,
            zorder=3,
        )
        infaller_r200 = PatchCollection(
            infallers.apply(
                lambda cl: Circle(
                    (cl["CentreOfMass_x"], cl["CentreOfMass_y"]),
                    cl["SORadius"],
                ),
                axis=1,
            ),
            fc="none",
            ec="C1",
            lw=1 + i,
            zorder=2,
        )
        ax.add_collection(main_5r200)
        ax.add_collection(infaller_r200)
        ax.plot(main_clusters.CentreOfMass_x, main_clusters.CentreOfMass_y, f"C0{fmt}")
        # ax.grid(zorder=-1)
        ax.set(xlabel="x (Mpc)", ylabel="y (Mpc)")
    # for zoom on right panel
    if args.ncl:
        cl = main_clusters.iloc[0]
        s = 6 * cl["SORadius"]
        xlim = cl["CentreOfMass_x"]
        xlim = (xlim - s, xlim + s)
        ylim = cl["CentreOfMass_y"]
        ylim = (ylim - s, ylim + s)
    else:
        xlim = (590, 690)
        ylim = (730, 830)
    axes[1].set(xlim=xlim, ylim=ylim)
    zoom = Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        ec="0.4",
        fc="none",
        lw=3,
        zorder=-1,
    )
    axes[0].add_patch(zoom)
    for ax in axes:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        angsizex = (
            ((xlim[1] - xlim[0]) * u.Mpc * cosmo.arcsec_per_kpc_comoving(z_ref))
            .to("deg")
            .value
        )
        angsizey = (
            ((ylim[1] - ylim[0]) * u.Mpc * cosmo.arcsec_per_kpc_comoving(z_ref))
            .to("deg")
            .value
        )
        tax = ax.twiny()
        tax.set(
            xlim=(-angsizex / 2, angsizex / 2), xlabel=f"$\\theta_x$ (deg @ z={z_ref})"
        )
        rax = ax.twinx()
        rax.set(
            ylim=(-angsizey / 2, angsizey / 2), ylabel=f"$\\theta_y$ (deg @ z={z_ref})"
        )
    output = "plots/testing/clusters_and_infallers.png"
    if args.ncl:
        output = output.replace(".png", f"_{args.ncl}.png")
    savefig(output, fig=fig)
    return


def plot_vstars_vgas(args, galaxies, xcol, ycol, vbins):
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    hist2d = np.histogram2d(galaxies[xcol], galaxies[ycol], vbins)[0]
    im = ax.pcolormesh(vbins, vbins, hist2d.T, cmap="BuPu", norm=LogNorm())
    plt.colorbar(im, ax=ax, label="N galaxies per cell")
    ax.plot(vbins, vbins, "k-", lw=2)
    ax.set(
        xlabel=xcol,
        ylabel=ycol,
        # xscale="log",
        # yscale="log",
    )
    output = "plots/testing/vstars_vgas_infallers.png"
    if args.ncl:
        output = output.replace(".png", f"_{args.ncl}.png")
    savefig(output, fig=fig, tight=False)
    return


if __name__ == "__main__":
    main()
