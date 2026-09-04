# Code Collection – Master's Thesis
## Capital Misallocation and Productivity: Southern European Evidence Revisited

Master's thesis by Curdin Lieberherr for the attainment of the degree: Master of Arts HSG in Economics at the University of St.Gallen.

This repository contains the code for the analysis, centered on `MisallocationAnalysis.py`, which implements the core analysis object used throughout. Its application and results are presented across several notebooks: `replicate_extend.ipynb` replicates Gopinath et al. (2017)'s analysis of Spain and extends it to the 2014–2024 period; `final_countries.ipynb` presents corresponding results for Italy, Belgium, Portugal, and Sweden; `capital_wedges.ipynb` reports the capital wedges analysis; `mismeasurement.ipynb` covers the measurement correction of the dispersion measures; and `CompareCountries.ipynb` implements an object for comparing results across multiple `MisallocationAnalysis` instances.

The `modules` folder contains Python objects and functions used to extend or support the analysis. The `data` folder contains public data from Eurostat. Note that Moody's Orbis data cannot be published due to licensing restrictions; the data used in this thesis was obtained through institutional access.

Python 3.12.13 was used for this project, and all dependencies are listed in `requirements.txt`.