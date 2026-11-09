# Seismic mapping and older-model provenance — clarification

The earlier statement that learned seismic mappings did not help was too broad. It referred to common-reference gravity agreement with untransformed seismic catalogue labels. This is not the same as evaluating the seismic observation model in its own coordinates.

If the observation channel is `y_seis = h_seis(theta) + noise`, compare `h_seis(theta_pred)` with held-out `y_seis` to assess source-space prediction. Comparing `theta_pred` directly with `y_seis` instead assesses agreement with that catalogue's adopted scale. A learned calibration can improve the first while changing the second. Neither comparison alone certifies physical latent gravity.

Reanalysis of the saved, source-projected terminal readouts gives these two-joint-seed DEV means (737 held-out APOKASC stars):

| Seismic observation map | Source-projected MAE [dex] | Source-predictive CRPS [dex] | Common gravity versus raw seismic MAE [dex] |
|---|---:|---:|---:|
| Identity | 0.080534 | 0.058713 | 0.080534 |
| Affine | 0.080994 | 0.059014 | 0.083690 |
| Teff/FeH contextual | 0.079006 | 0.057919 | 0.087523 |

Thus the contextual map modestly improves source-projected MAE (~1.9%) and CRPS (~1.35%) on average, while common-to-raw-reference error worsens. The source-space benefit is driven mostly by seed one: paired DEV MAE changes are −0.002391 dex (star-bootstrap 95% interval [−0.003191, −0.001682]) and −0.000667 dex ([−0.001376, +0.000013]). TUNE source-space mean MAE changes from 0.074534 to 0.073456 dex, but seed two has no clear improvement. These intervals condition on the trained seeds and omit reference systematic uncertainty. This is a small candidate improvement, not a decisive architecture win. The affine-only mapping does not show the same source-space gain.

The older model was `decision6_nsf_energy_parallax_20261106_v1`, checkpoint `c59ea40759bfe8f16e0f5553fdb6f7e99796fd2d146eac6f452d151daf0adff1`. Its recorded gravity-label presentations are `logg:apogee` and `logg:galah`, with no explicit APOKASC channel. It did not receive an explicit APOKASC seismic loss in that supervised stage. However, the ingestion code selects APOGEE `LOGG`, not `LOGG_SPEC`; SDSS documents that these calibrated gravities use asteroseismic calibration. Consequently the older model was not free of indirect seismic information. See [APOGEE parameter definitions](https://www.sdss4.org/dr17/irspec/parameters/) and [ASPCAP calibration](https://www.sdss4.org/dr17/irspec/aspcap/).

“Adaptation” previously meant continued training on the expanded source-native observation objective, not fitting observation maps alone. In the joint control arms, the encoder/decoder, NSF head, native-survey maps and optional seismic map were all updated, with retained MAE reconstruction. In frozen controls, the encoder/decoder stayed fixed but the NSF head and maps were updated. The production run also inherited a seismic-trained mapping-pilot head; the matched controls instead start from the earlier pooled NSF before that explicit seismic stage.

The common reference in these controls is fixed to the APOGEE gravity convention, while the production run uses a fixed seismic identity reference. An independent physical comparison needs its own uncertainty/systematic model; TESS and APOKASC are both seismic-derived references. Current evidence supports retaining a learned contextual seismic channel as a candidate and distinguishing source-space fit from common-scale agreement. It does not establish an absolute common gravity zero point.

Saved readouts, paired bootstrap calculations and older-model provenance are beside this note under `projected-seismic/`.

## Encoder lineage clarification

The strong older baseline is **not frozen SSL + NSF**. Its recorded arm is `encoder: finetune`, with masked MAE reconstruction weight 0.03. It is SSL pretraining followed by joint supervised encoder/NSF fine-tuning, selected at update 8,000. The frozen controls freeze this already fine-tuned encoder during the subsequent expanded-source experiment; they do not start from a purely SSL-trained frozen encoder. Thus these results do not establish that a frozen SSL representation beats an unfrozen one. At matched seed one, frozen identity-seismic adaptation gives TESS/APOKASC common MAE 0.10329/0.08343 dex, versus joint adaptation 0.10919/0.08784. The second joint seed gives 0.08641/0.07323; there is no second frozen seed in this wave. Both matched frozen and joint conditions contain survey mappings and explicit seismic supervision.
