# File contents

There are currently two files: `map_cov_WISC_2MPZ_smooth1p0_not_weighted_corrected.npz` and `map_cov_WISC_2MPZ_smooth1p0_weighted_corrected.npz`.

They contain maps of the galaxy overdensity and its inverse covariance from the combination of WISExSuperCOSMOS and 2MPZ.

The "weighted" version has been computed by assigning each galaxy a weight aimed mimicking the SGWB radial kernel.

Each file contains two arrays:
- `map`: the overdensity map (not inverse-variance weighted, unlike Arianna's maps)
- `inv_cov`: the inverse covariance matrix.
