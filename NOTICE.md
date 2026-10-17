# Third-party data and components

## Data: IBTrACS
Track and intensity data come from NOAA's International Best Track Archive for Climate Stewardship (IBTrACS) v04r01,
https://www.ncei.noaa.gov/products/international-best-track-archive. The data are not redistributed in this repository;
the pipeline downloads them. Check NOAA/NCEI's current terms of use and cite:

> Knapp, K. R., M. C. Kruk, D. H. Levinson, H. J. Diamond, and C. J. Neumann, 2010: The International Best Track Archive
> for Climate Stewardship (IBTrACS): Unifying tropical cyclone best track data. *Bull. Amer. Meteor. Soc.*, 91, 363-376,
> doi:10.1175/2009BAMS2755.1

Also credit the underlying agencies (e.g. NHC, JTWC) and state the data version and access date.
`DEMO_synthetic_data_app.html` contains computer-generated fake storms, not real observations.

## Libraries loaded by the app (from CDNs)
- Leaflet (BSD-2-Clause) https://leafletjs.com
- Chart.js (MIT) https://www.chartjs.org
- Map tiles: Esri World Light Gray Base, "Tiles (c) Esri". Subject to Esri terms; fine for personal and portfolio use.
  Swap the tile URL in `app/hurricane_app.html` if you deploy commercially.

## Python dependencies
numpy, pandas, scikit-learn, xgboost, matplotlib, joblib (and optionally cartopy) under their own open-source licenses.

## Disclaimer
This software is for research and education. It is not an operational forecast and must not be used for safety-critical decisions.
