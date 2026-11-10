# Master label census: new sources, 2026-11-10

This was a local run in a cloud session, not on CANFAR, using
`scripts/run_master_label_census.py`. Native sources came from SDSS SAS (with
publisher SHA1/size checks), the VizieR CfA mirror and the ESA Gaia archive.
`manifest.json` holds per-source receipts (sha256), policies and the full
census. No roles were assigned and no training was performed.

**Scope:** only the sources added by the master-label plan. The join against
the eight survey groups already staged on ARC (18,191,859 Gaia-linked stars)
needs the CANFAR job (`scripts/launch_master_label_census.py`). Until it runs,
"Only here" means exclusive *among these new sources*, not against the
existing union.

## Union of the new sources

- **1,549,757 unique Gaia DR3 stars**; 17,618 appear in more than one new source.
- Metal-poor stars use the quality-passing, grid-edge-free median [Fe/H] per
  star:

| [Fe/H] < | −1.0 | −1.5 | −2.0 | −2.5 | −3.0 | −3.5 | −4.0 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Any new source | 226,874 | 122,705 | 52,423 | 12,670 | 3,392 | 828 | 190 |
| High-resolution reference tier | 2,591 | 1,898 | 1,305 | 836 | 359 | 89 | 21 |

- 8,465 stars have only grid-edge (censored) metallicities. These are DR19
  ASPCAP stars piled up at [Fe/H] = −2.50.

## Per source

| Source | Tier | Rows | Gaia stars | Quality stars | Only here | Ambiguous / unmatched rows | <−2.0 | <−2.5 | <−3.0 | <−3.5 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| sdss_dr19_astra_aspcap | survey | 1,095,480 | 968,486 | 696,330 | 955,588 | 1,775 / 6,419 | 1,218 | 0 | 0 | 0 |
| sdss_dr12_sspp | survey | 386,337 | 384,104 | 370,437 | 379,715 | 10,433 / 1,810 | 28,763 | 7,038 | 2,055 | 585 |
| gaia_eso_dr5_1 | survey | 114,916 | 114,721 | 94,145 | 111,134 | 5,787 / 195 | 1,169 | 232 | 66 | 0 |
| s5_dr1 | metal poor survey | 31,777 | 29,595 | 29,581 | 28,131 | 0 / 3 | 2,540 | 937 | 273 | 63 |
| pigs_aat | metal poor survey | 13,235 | 13,231 | 13,231 | 13,002 | 0 / 0 | 2,349 | 315 | 49 | 21 |
| lamost_vmp_li2018 | metal poor survey | 10,011 | 9,674 | 9,674 | 6,778 | 60 / 31 | 9,614 | 2,706 | 621 | 83 |
| lamost_cat_li2025 | metal poor survey | 8,440 | 8,378 | 8,378 | 5,392 | 0 / 0 | 8,362 | 1,070 | 24 | 0 |
| pristine_aguado2019 | metal poor survey | 1,007 | 990 | 990 | 802 | 12 / 17 | 781 | 453 | 86 | 0 |
| bestbrightest_placco2019 | metal poor survey | 857 | 857 | 796 | 743 | 38 / 0 | 311 | 121 | 17 | 1 |
| smss_dacosta2019_t2 | metal poor survey | 142 | 142 | 142 | 56 | 2 / 0 | 142 | 142 | 142 | 28 |
| smss_dacosta2019_t3 | metal poor survey | 29 | 29 | 29 | 17 | 0 / 0 | 29 | 29 | 29 | 4 |
| pastel_2020 | highres reference | 81,362 | 36,966 | 17,943 | 30,466 | 6,809 / 1,607 | 886 | 559 | 252 | 69 |
| roederer2014 | highres reference | 313 | 295 | 295 | 1 | 23 / 18 | 249 | 194 | 76 | 15 |
| rpa_holmbeck2020 | highres reference | 233 | 232 | 232 | 158 | 3 / 0 | 151 | 60 | 8 | 1 |
| yong2013_literature | highres reference | 152 | 152 | 152 | 32 | 2 / 0 | 147 | 132 | 61 | 18 |
| smss_yong2021 | highres reference | 150 | 150 | 150 | 45 | 8 / 0 | 150 | 132 | 86 | 14 |
| rpa_hansen2018 | highres reference | 107 | 107 | 107 | 65 | 3 / 0 | 86 | 44 | 3 | 0 |
| yong2013_program | highres reference | 47 | 38 | 38 | 14 | 2 / 0 | 38 | 38 | 23 | 13 |

The metal-poor columns count unique quality-passing stars whose edge-free
median [Fe/H] is below each threshold.

## Findings

1. **APOGEE cannot supply the metal-poor tail.** DR19 Astra ASPCAP has no
   [Fe/H] below −2.50, and 652 stars sit on that floor. They are flagged as
   censored and excluded from the metal-poor bins. Treat them as upper limits
   (or drop them) when used as labels.
2. **SEGUE/SSPP is the largest new metal-poor source:** 28.8k stars below −2,
   2.1k below −3 and 585 below −3.5. It contributes 380k stars absent from
   every other new source.
3. **DR19 ASPCAP adds 329k SDSS-V stars** with no DR17 row in the same file,
   on top of 639k DR17 stars.
4. **Gaia-ESO is resolved:** 114,721 of 114,916 rows are matched to Gaia DR3.
   5,787 are ambiguous (crowded clusters/bulge; `match_n_within > 1`) and 195
   are unmatched.
5. **Dedicated metal-poor surveys add depth, not volume:** S5, PIGS, LAMOST
   Li+2018/Li+2025, Pristine and SkyMapper together give about 24k stars below
   −2 and 1.2k below −3. These are per-source sums; Li+2018 and Li+2025 share
   LAMOST stars.
6. **The high-resolution reference tier** (RPA, Yong, Roederer, PASTEL) gives
   359 stars below −3 for calibrating the metal-poor scales of the low-resolution
   surveys. Reserve these stars from FIT.
