"""Training [Fe/H] prior of a fine-tune table, for prior-free predictions.

usage: make_feh_prior.py RECORDS.npz OUTPUT.npz

Per star (table row), the median [Fe/H] of its non-anchor survey records;
their smoothed histogram on ``feh_prior.EDGES`` is the prior the head learned
under (``predict_fits_rows --feh-prior``).
"""

import sys

import numpy as np
import pandas as pd

from masked_stellar_autoencoder.training.feh_prior import (
    CENTRES,
    EDGES,
    smoothed_histogram,
)


def main():
    rec = np.load(sys.argv[1], allow_pickle=False)
    names = [str(n) for n in rec["survey_names"]]
    fe = list(rec["parameters"]).index("fe_h")
    val = rec["value"][:, fe]
    anchor = [i for i, n in enumerate(names) if n.startswith("anchor")]
    ok = np.isfinite(val) & ~np.isin(rec["survey"], anchor)
    per_star = pd.Series(val[ok]).groupby(rec["row"][ok]).median().to_numpy()
    prior = smoothed_histogram(per_star)
    np.savez(sys.argv[2], centres=CENTRES, edges=EDGES, prior=prior,
             n_stars=len(per_star), source=sys.argv[1])  # fmt: skip
    print(
        f"{len(per_star)} stars; prior mass below -2: {prior[CENTRES < -2].sum():.4f}"
    )


if __name__ == "__main__":
    main()
