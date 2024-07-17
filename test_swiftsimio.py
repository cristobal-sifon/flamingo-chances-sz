import h5py
from icecream import ic
from matplotlib import pyplot as plt
import numpy as np
import os
import swiftsimio as sw
from testing import parse_args

from plottery.plotutils import savefig, update_rcParams

import flamingo_tools as ftools

update_rcParams()


def main():
    args = parse_args()

    ## Load particles
    part = sw.load(args.snapshot_file)
    meta = part.metadata
    ic(meta.header)
    ic(meta.present_particle_types, meta.present_particle_names)
    ic(meta.gas_properties.field_names)
    ic(meta.gas_properties.field_units)
    ic(meta.stars_properties.field_names)

    ## Load subhalos
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
        halofile, cluster_mass_min=1e14, n=10, so_cols="ComptonY"
    )
    ic(cluster_galaxies)
    # with h5py.File(halofile) as f:
    #     com = f.get("BoundSubhaloProperties/CentreOfMass").value
    return


if __name__ == "__main__":
    main()
