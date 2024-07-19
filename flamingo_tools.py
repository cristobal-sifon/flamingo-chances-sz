import h5py
import numpy as np
import pandas as pd
import unyt

# debugging
from icecream import ic
from time import time

# see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html
pd.options.mode.copy_on_write = True


def galaxies_in_clusters(
    halofile,
    cluster_mass_min=0,
    cluster_mass_max=np.inf,
    clusters=None,
    n=None,
    overdensity="200_crit",
    so_cols=None,
    random_seed=None,
):
    """Find cluster galaxies within a given cluster mass range

    NOTE: only works with HBT+ catalogs for now

    Must provide either ``halofile`` or ``clusters``.

    Parameters
    ----------
    halofile : str
        hdf5 file name
    cluster_mass_min, cluster_mass_max : float, optional
        minimum and maximum spherical overdensity cluster mass, in Msun
    clusters : pd.DataFrame, optional
        cluster sample. Must contain at least the columns ``(HostHaloId,CentreOfMass_x,CentreOfMass_y,CentreOfMass_z)``
    n : int, optional
        number of clusters to return, chosen randomly given the mass range.
        If not specified all clusters are returned
    overdensity : str
        spherical overdensity as named in ``halofile``
    so_cols : list, optional
        list of spherical overdensity columns to include in addition to
        ``TotalMass``. Ignored if ``clusters`` is provided

    Returns
    -------
    cluster_galaxies : pd.DataFrame
        galaxies within clusters
    """
    with h5py.File(halofile) as file:
        hostid = file.get("InputHalos/HBTplus/HostHaloId")[()]
        # merge centrals and satellites as in testing.py
        galaxies = pd.DataFrame(
            {
                "TrackId": file.get("InputHalos/HBTplus/TrackId")[()],
                "HostHaloId": hostid,
                "Rank": file.get("InputHalos/HBTplus/Rank")[()],
            }
        )
        if clusters is None:
            mcl = file.get(f"SO/{overdensity}/TotalMass")[()]
            bcg = (mcl > cluster_mass_min) & (mcl < cluster_mass_max)
            clusters = pd.DataFrame(
                {
                    "HostHaloId": galaxies["HostHaloId"][bcg],
                    "TotalMass": mcl[bcg],
                }
            )
            if so_cols is not None:
                if isinstance(so_cols, str):
                    so_cols = (so_cols,)
                for col in so_cols:
                    clusters[col] = file.get(f"SO/{overdensity}/{col}")[()][bcg]
        else:
            # let's just make sure it contains everything we need
            assert isinstance(clusters, pd.DataFrame)
            assert (
                "HostHaloId" in clusters.columns
            ), "clusters must contain column HostHaloId"
    galaxies = galaxies.loc[galaxies["HostHaloId"] > -1]
    if n is not None:
        rdm = np.random.default_rng(random_seed)
        n = rdm.choice(
            clusters["HostHaloId"].size,
            n,
            replace=False,
            shuffle=False,
        )
        clusters = clusters.iloc[n]
    # we don't need this
    if "TrackId" in clusters.columns:
        clusters.pop("TrackId")
    cluster_galaxies = clusters.merge(
        galaxies, how="inner", on="HostHaloId", suffixes=("_cl", "_gal")
    )
    return cluster_galaxies.sort_values(["HostHaloId", "Rank"], ignore_index=True)


def infalling_groups(
    halofile,
    cluster_mass_min=0,
    cluster_mass_max=np.inf,
    distance_max=5,
    group_mass_min=0,
    n=None,
    overdensity="200_crit",
    so_cols=None,
    random_seed=None,
):
    """Find groups falling into a sample of clusters

    This function looks for all clusters in the chosen mass range that are the
    most massive cluster within ``distance_max`` and then identifies all groups
    around them. Here, groups are defined simply by a lower mass cut. The cluster
    center is taken as the center of mass.

    Parameters
    ----------
    halofile : str
        hdf5 file name
    cluster_mass_min, cluster_mass_max : float, optional
        minimum and maximum spherical overdensity cluster mass, in Msun
    n : int, optional
        number of clusters to return, chosen randomly given the mass range.
        If not specified all clusters are returned
    distance_max : float
        maximum 3d distance around which to search for groups, in units of
        the ``SORadius`` specified by ``overdensity``
    group_mass_min : float, optional
    overdensity : str
        spherical overdensity as named in ``halofile``
    so_cols : list, optional
        list of spherical overdensity columns to include in addition to
        ``TotalMass``

    Returns
    -------
    main_clusters: pd.DataFrame
        most massive clusters within their own ``distance_max * SORadius``
    infallers : pd.DataFrame
        infalling groups
    """
    with h5py.File(halofile) as file:
        hostid = file.get("InputHalos/HBTplus/HostHaloId")[()]
        mass = file.get(f"SO/{overdensity}/TotalMass")[()]
        # working with central galaxies is enough here
        good = (hostid > -1) & (file.get("InputHalos/HBTplus/Rank")[()] == 0)
        df = pd.DataFrame(
            {
                "TrackId": file.get("InputHalos/HBTplus/TrackId")[()][good],
                "HostHaloId": hostid[good],
                "TotalMass": mass[good],
                "SORadius": file.get(f"SO/{overdensity}/SORadius")[()][good],
            }
        )
        clusters = (mass[good] > cluster_mass_min) & (mass[good] < cluster_mass_max)
        com = file.get(f"SO/{overdensity}/CentreOfMass")[()][good]
        for i, x in enumerate("xyz"):
            df[f"CentreOfMass_{x}"] = com[:, i]
    clusters = df.loc[clusters]
    # subsample?
    if n is not None:
        rdm = np.random.default_rng(random_seed)
        n = rdm.choice(
            clusters["TotalMass"].size,
            n,
            replace=False,
            shuffle=False,
        )
        clusters = clusters.iloc[n]
    groups = df.loc[df["TotalMass"] > group_mass_min]
    # main_clusters = []
    infallers = {
        "TrackId": [],
        "HostHaloId": [],
        "TotalMass": [],
        "SORadius": [],
        "MainTrackId": [],
        "DistanceToMain": [],
    }
    for cl in clusters.itertuples():
        dist = (
            (cl.CentreOfMass_x - groups.CentreOfMass_x) ** 2
            + (cl.CentreOfMass_y - groups.CentreOfMass_y) ** 2
            + (cl.CentreOfMass_z - groups.CentreOfMass_z) ** 2
        ) ** 0.5
        # we need the main clusters in here too, to do the merging below
        near = dist < distance_max * cl.SORadius
        if not np.any(clusters["TotalMass"][near] > cl.TotalMass):
            # main_clusters.append(cl.Index)
            infallers["MainTrackId"].extend(near.sum() * [cl.TrackId])
            for col in ("TrackId", "HostHaloId", "TotalMass", "SORadius"):
                infallers[col].extend(groups[col].loc[near])
            infallers["DistanceToMain"].extend(dist[near])
    infallers = pd.DataFrame(infallers)
    main_clusters = clusters.loc[np.isin(clusters["TrackId"], infallers["MainTrackId"])]
    # now keep only infallers
    infallers = infallers.loc[infallers["DistanceToMain"] > 0]
    infallers.sort_values(["MainTrackId", "TotalMass"])
    return main_clusters, infallers
