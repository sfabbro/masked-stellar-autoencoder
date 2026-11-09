# Full label-union common-reference QA

Complete usable labelled-source registry, immutable first large checkpoint. All histogram counts and residual statistics use all eligible measurements; scatter overlays use fixed uniform samples. Inverse-mapped noisy survey observations are references, not physical truth. No averaging of overlapping surveys. CAL is diagnostic only here; no new correction is fitted.

Prediction stars: 10,427,497; unique usable spectroscopic-union stars: 10,403,916; checkpoint update: 2000.

An outer join retains all available labels; it does not define a unique true label. Each source measurement is transformed with the exact inverse of its learned monotone observation map. Overlap remains multiple measurements. Individual source maps do not establish an absolute physical reference.

Kiel panels require Teff and logg actually observed in the same survey. Predictions are shown on identical stars. Density contours use every in-range record; scatter is a fixed uniform overlay. Tail counts are recorded in the manifest. Teacher age/distance channels are shown separately in residual statistics and are not spectroscopic age/distance truth.

FIT residuals include optimization exposure; TUNE has selection exposure; DEV has development exposure. CAL is not used to adjust this model or intervals. Raw interval-reference inclusion omits measurement noise and is not a calibrated coverage claim. Full statistics: [CSV](residual-statistics.csv).

Cluster median distributions and equal-star posterior mixtures are different quantities. Narrower distributions alone do not establish better physical calibration.

![kiel-survey-colors-all.png](kiel-survey-colors-all.png)

![kiel-survey-colors-fit.png](kiel-survey-colors-fit.png)

![kiel-survey-colors-tune.png](kiel-survey-colors-tune.png)

![kiel-survey-colors-cal.png](kiel-survey-colors-cal.png)

![kiel-survey-colors-devtest.png](kiel-survey-colors-devtest.png)

![kiel-unique-union-density.png](kiel-unique-union-density.png)

![residuals-all-splits-teff.png](residuals-all-splits-teff.png)

![population-distributions-teff.png](population-distributions-teff.png)

![residuals-all-splits-logg.png](residuals-all-splits-logg.png)

![population-distributions-logg.png](population-distributions-logg.png)

![residuals-all-splits-fe_h.png](residuals-all-splits-fe_h.png)

![population-distributions-fe_h.png](population-distributions-fe_h.png)

![residuals-all-splits-alpha.png](residuals-all-splits-alpha.png)

![population-distributions-alpha.png](population-distributions-alpha.png)

![residuals-all-splits-age.png](residuals-all-splits-age.png)

![population-distributions-age.png](population-distributions-age.png)

![residuals-all-splits-distance.png](residuals-all-splits-distance.png)

![population-distributions-distance.png](population-distributions-distance.png)

![overlap-agreement-teff.png](overlap-agreement-teff.png)

![overlap-agreement-logg.png](overlap-agreement-logg.png)

![overlap-agreement-fe_h.png](overlap-agreement-fe_h.png)

![cluster-NGC_5139_oCen-distributions.png](cluster-NGC_5139_oCen-distributions.png)

![cluster-NGC_104_47Tuc-distributions.png](cluster-NGC_104_47Tuc-distributions.png)

![cluster-NGC_3201-distributions.png](cluster-NGC_3201-distributions.png)

![cluster-NGC_6205_M_13-distributions.png](cluster-NGC_6205_M_13-distributions.png)

![cluster-NGC_6752-distributions.png](cluster-NGC_6752-distributions.png)

![cluster-NGC_5272_M_3-distributions.png](cluster-NGC_5272_M_3-distributions.png)

![cluster-NGC_5904_M_5-distributions.png](cluster-NGC_5904_M_5-distributions.png)

![cluster-NGC_2808-distributions.png](cluster-NGC_2808-distributions.png)

![comparator-Andrae_MH_proxy-teff.png](comparator-Andrae_MH_proxy-teff.png)

![comparator-Andrae_MH_proxy-logg.png](comparator-Andrae_MH_proxy-logg.png)

![comparator-Andrae_MH_proxy-fe_h.png](comparator-Andrae_MH_proxy-fe_h.png)

![comparator-Andrae_MH_proxy-matched-kiel.png](comparator-Andrae_MH_proxy-matched-kiel.png)

![comparator-GaiaNet-teff.png](comparator-GaiaNet-teff.png)

![comparator-GaiaNet-logg.png](comparator-GaiaNet-logg.png)

![comparator-GaiaNet-fe_h.png](comparator-GaiaNet-fe_h.png)

![comparator-GaiaNet-matched-kiel.png](comparator-GaiaNet-matched-kiel.png)
