from argparse import ArgumentParser
import h5py
from icecream import ic
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd

from plottery.plotutils import savefig, update_rcParams
from velociraptor import load
from velociraptor.tools import get_full_label

update_rcParams()

"""Notes
- I probably need separate variables for:
    * main directory, e.g., /cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/
    * particles directory, e.g., /cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots_downsampled/
    * HBT+ subhaloes directory, e.g., /cosma8/data/dp004/dc-foro1/HBT_SOAP/L1000N1800/HYDRO_FIDUCIAL/SOAP_uncompressed/HBTplus/
        (only available for few sims/snaps so far)
    * VR subhaloes directory, e.g.,  /cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP/
        (to be replaced by HBT+ subhaloes as they become available)


"""


def main():
    args = parse_args()

    snap = f"{args.snapshot:04d}"
    halofile = os.path.join(args.path["SOAP-HBT"], f"halo_properties_{snap}.hdf5")
    mempath = os.path.join(args.path["SOAP-HBT"], f"membership_{snap}")

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
    ic(mempath, os.listdir(mempath))
    fmem = os.path.join(mempath, f"membership_{snap}.0.hdf5")
    ic(fmem)
    with h5py.File(fmem) as f:
        ic(f.keys())
        for key in f.keys():
            ic(key)
            for gr in f[key].items():
                ic(gr)
            print()

    ic(os.listdir(args.path["particles"]))
    fpart = os.path.join(args.path["particles"], "flamingo_0000.hdf5")
    with h5py.File(fpart) as f:
        ic(f.keys())
        for key in f.keys():
            ic(key)
            for gr in f[key].items():
                ic(gr)
            print()

    ## compare InputHalos/is_central with InputHalos/HBTplus/Rank
    with h5py.File(halofile) as f:
        is_central = np.array(f["InputHalos/is_central"], dtype=bool)
        rank = np.array(f["InputHalos/HBTplus/Rank"])
        hostid = np.array(f["InputHalos/HBTplus/HostHaloId"])
        ic(np.array_equal(is_central, rank == 0))
    # they are the same!

    ## select satellite galaxies with log mstar > 1e11 in M200c>1e14 clusters
    with h5py.File(halofile) as f:
        good = hostid > -1
        sat = ~is_central & good
        cen = is_central & good
        ic(sat.sum())
        mstar = np.array(f["BoundSubhaloProperties/StellarMass"])
        massive = mstar > 1e11
        ic(massive.sum(), (sat & massive).sum())
        # this is >0 only for centrals -- use InputHalos/HBTplus/HostHaloId to match
        m200 = np.array(f["SO/200_crit/TotalMass"])
        ic(is_central[:100], np.log10(m200[:100]))
        ic(m200, m200.shape)
        # this automatically only considers centrals
        bcg = good & (m200 > 1e14)
        ic(bcg.shape, bcg.sum())
        # match by hostid to assign m200 to satellites too
        trackid = np.array(f["InputHalos/HBTplus/TrackId"])
        galaxies = pd.DataFrame(
            {
                "TrackId": trackid,
                "HostHaloId": np.array(f["InputHalos/HBTplus/HostHaloId"]),
                "Rank": rank,
                "StellarMass": mstar,
            }
        )
        clusters = pd.DataFrame(
            {
                "HostHaloId": galaxies["HostHaloId"][bcg],
                "m200crit": m200[bcg],
                "ySZ": np.array(f["SO/200_crit/ComptonY"])[bcg],
            }
        )
        galaxies = galaxies[good]
    ic(galaxies, clusters)
    ic(np.sort(galaxies["HostHaloId"]))
    ic(np.sort(clusters["HostHaloId"]))
    m200 = clusters.merge(galaxies, how="inner", on="HostHaloId")
    ic(m200)
    # now that we have m200 assigned to satellites
    cluster = np.isin(trackid, m200["TrackId"])
    ic(cluster.shape, cluster.sum(), (cluster & cen).sum(), (cluster & sat).sum())
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    msbins = np.logspace(9, 14, 100)
    ax.hist(
        m200["StellarMass"][m200["Rank"] > 0], msbins, label="Satellites", alpha=0.5
    )
    ax.hist(m200["StellarMass"][m200["Rank"] == 0], msbins, label="Centrals", alpha=0.5)
    ax.legend()
    ax.set(xlabel="$m_\star$", xscale="log", yscale="log")
    output = "plots/testing/hist_mstar_cluster.png"
    savefig(output, fig=fig, tight=False)

    ## Compton-y vs number of satellites
    # hhid, nsat = np.unique(galaxies["HostHaloId"], return_counts=True)
    # df_satellites = pd.DataFrame("HostHaloId")
    nsat = galaxies["HostHaloId"].value_counts()
    ic(nsat)
    # clusters = clusters.merge(pd.DataFrame({}))

    ## plot Compton y vs mstar for (a few) massive cluster galaxies
    with h5py.File(halofile) as f:
        com = np.array(f["BoundSubhaloProperties/CentreOfMass"])[cluster & sat]
        ic(com.min(axis=0), com.max(axis=0), com.shape)
    chunk = 10
    with h5py.File(fpart) as fp:
        xyz = fp["DMParticles/Coordinates"]
        ic(xyz.shape)
        for i in range(com.size // chunk):
            xyz_gal = com[i * chunk : (i + 1) * chunk]
            dist = ((xyz - xyz_gal[:, None]) ** 2).sum(axis=2) ** 0.5
            near = np.min(dist, axis=0) < 0.1  # Mpc?
            ic(i, xyz_gal, dist.shape, dist.min(), dist.max(), near.shape, near.sum())
            if i >= 1:
                break
    # with h5py.File(fpart) as f:
    #     comptony = np.array(f["BoundSubhaloProperties/"])


def parse_args():
    parser = ArgumentParser()
    add = parser.add_argument
    add("-b", "--box", default="L1000N1800")
    add("-s", "--sim", default="HYDRO_FIDUCIAL")
    add("-z", "--snapshot", default=77, type=int)
    args = parser.parse_args()
    # for now
    args.path = dict(main=os.path.join(os.environ.get("FLAMINGO"), args.box, args.sim))
    args.path["particles"] = os.path.join(
        args.path.get("main"), "snapshots_downsampled"
    )
    args.path["SOAP-HBT"] = os.path.join(
        "/cosma8/data/dp004/dc-foro1/HBT_SOAP",
        args.box,
        args.sim,
        "SOAP_uncompressed",
        "HBTplus",
    )
    args.path["SOAP-VR"] = os.path.join(args.path.get("main"), "SOAP")
    if args.snapshot is not None:
        args.path["snapshot"] = os.path.join(
            args.path.get("main"), "snapshots", f"flamingo_{args.snapshot:04d}"
        )
        args.snapshot_file = os.path.join(
            args.path.get("snapshot"), f"flamingo_{args.snapshot:04d}.hdf5"
        )
    return args


if __name__ == "__main__":
    main()
