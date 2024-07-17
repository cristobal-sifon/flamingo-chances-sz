import h5py
import numpy as np
import pandas as pd


def galaxies_in_clusters(
    halofile,
    cluster_mass_min=0,
    cluster_mass_max=np.inf,
    n=None,
    overdensity="200_crit",
    so_cols=None,
    random_seed=None,
):
    """Find cluster galaxies within a given cluster mass range

    NOTE: only works with HBT+ catalogs for now

    Parameters
    ----------
    halofile : str
        hdf5 file name
    cluster_mass_min, cluster_mass_max : float, optional
        minimum and maximum spherical overdensity cluster mass
    n : int, optional
        number of clusters to use, chosen randomly given the mass range.
        If not specified all clusters are used
    overdensity : str
        spherical overdensity as named in ``halofile``
    so_cols : list, optional
        list of spherical overdensity columns to include in addition to
        ``TotalMass``

    Returns
    -------
    cluster_galaxies : pd.DataFrame
        galaxies within clusters, including and index matching to the
        contents of ``halofile`` and the spherical overdensity mass
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
    cluster_galaxies = clusters.merge(galaxies, how="inner", on="HostHaloId")
    return cluster_galaxies.sort_values(["HostHaloId", "Rank"], ignore_index=True)
