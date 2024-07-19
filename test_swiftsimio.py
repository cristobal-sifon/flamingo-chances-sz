import h5py
from icecream import ic
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.integrate import trapezoid
import swiftsimio as sw
from time import time
import unyt
from unyt import Mpc

from plottery.plotutils import savefig, update_rcParams

from testing import parse_args
import flamingo_tools as ftools

update_rcParams()


def main():
    args = parse_args()

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
            print()

    # Load cluster galaxies for 10 random clusters
    cluster_galaxies = ftools.galaxies_in_clusters(
        halofile, cluster_mass_min=1e14, n=3, so_cols="ComptonY", random_seed=1
    )
    ic(cluster_galaxies)

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
        com = file.get("BoundSubhaloProperties/CentreOfMass")
        for i, coord in enumerate("xyz"):
            gals[coord] = com[:, i]
        cluster_galaxies = gals.merge(cluster_galaxies, on="TrackId", how="right")
        del gals
    ic(cluster_galaxies)
    cluster_galaxies = cluster_galaxies.loc[cluster_galaxies["StellarMass"] > 1e10]
    ic(cluster_galaxies)

    # find particles associated with each selected cluster galaxy
    mask = sw.mask(args.snapshot_file)
    dmax = 0.1 * Mpc
    xyz = cluster_galaxies[["x", "y", "z"]].to_numpy() * Mpc
    ic(dmax, xyz)
    ti = time()
    regions = (
        np.transpose(
            [[xyz[:, i] - dmax, xyz[:, i] + dmax] for i in range(xyz.shape[1])],
            axes=(2, 0, 1),
        )
        * Mpc
    )
    tf = time()
    ic(regions, regions.shape, tf - ti)
    mask.constrain_spatial(regions[0])
    particles_test = sw.load(args.snapshot_file, mask=mask)
    ic(particles_test.gas.masses.shape)
    for region in regions[1:]:
        mask.constrain_spatial(region, intersect=True)
    particles = sw.load(args.snapshot_file, mask=mask)
    # are the particles sorted following the masks?
    ic(
        particles.gas.coordinates.shape,
        particles.gas.masses.shape,
        particles.gas.temperatures.shape,
    )
    # match particles to galaxies
    rng_gas = np.arange(particles.gas.masses.size, dtype=int)
    ti = time()
    matching_particles = [
        rng_gas[((particles.gas.coordinates - xyz_i) ** 2).sum(axis=1) ** 0.5 < dmax]
        for xyz_i in xyz
    ]
    tf = time()
    ic(tf - ti)
    y_tot = np.zeros(len(cluster_galaxies))
    for i, p in enumerate(matching_particles):
        y_tot[i] = particles.gas.compton_yparameters[p].sum()
        # if cluster_galaxies["GasMass"].iloc[i] > 0:
        if i % 100 == 0:
            ic(
                i,
                cluster_galaxies[["GasMass", "x", "y", "z"]].iloc[i],
                p,
                particles.gas.coordinates[p],
                y_tot[i],
            )
    cluster_galaxies["ComptonY"] = y_tot
    ic(cluster_galaxies[["GasMass", "StellarMass", "ComptonY"]])
    ic((cluster_galaxies["GasMass"] > 0).sum())

    # find infalling groups
    ti = time()
    main_clusters, infallers = ftools.infalling_groups(
        halofile, cluster_mass_min=1e14, group_mass_min=1e13
    )
    tf = time()
    ic(
        main_clusters,
        infallers,
        tf - ti,
    )
    ti = time()
    galaxies_in_infallers = ftools.galaxies_in_clusters(halofile, clusters=infallers)
    tf = time()
    ic(galaxies_in_infallers, tf - ti)

    
    return


if __name__ == "__main__":
    main()
