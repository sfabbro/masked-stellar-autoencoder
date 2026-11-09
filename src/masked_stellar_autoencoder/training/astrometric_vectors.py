"""Unit-vector propagation for local QA matching; pmra is mu_alpha*cos(dec)."""

import numpy as np


def directions(ra, dec, pmra=None, pmdec=None, dt=0.0):
    ra, dec = np.asarray(ra, dtype="f8"), np.asarray(dec, dtype="f8")
    a, d = np.deg2rad(ra), np.deg2rad(dec)
    r = np.column_stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])
    if dt:
        pmra, pmdec = np.asarray(pmra, dtype="f8"), np.asarray(pmdec, dtype="f8")
        east = np.column_stack([-np.sin(a), np.cos(a), np.zeros(len(a))])
        north = np.column_stack(
            [-np.cos(a) * np.sin(d), -np.sin(a) * np.sin(d), np.cos(d)]
        )
        r += (
            dt
            * np.deg2rad(1 / 3600000)
            * (pmra[:, None] * east + pmdec[:, None] * north)
        )
        r /= np.linalg.norm(r, axis=1)[:, None]
    return r
