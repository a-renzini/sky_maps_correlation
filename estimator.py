import numpy as np
import healpy as hp
from scipy.special import legendre
import time

# File names
fn_gal = 'galaxies/map_cov_WISC_2MPZ_smooth1p0_weighted_corrected.npz'
fn_SGWB = 'GWB_covariance.npz'


# Load maps
map_g = np.load(fn_gal)['map']
inv_cov_gg = np.load(fn_gal)['inv_cov']
z_g = np.dot(inv_cov_gg, map_g)
z_w = np.load(fn_SGWB)['Z_p'][:, 0].astype('float64')
inv_cov_ww = np.load(fn_SGWB)['M_p_pp'][:, :, 0, 0].astype('float64')

# Number of pixels
npix = len(map_g)
# Corresponding Nside resolution parameter
nside = hp.npix2nside(npix)
# This is the maximum ell you should use for a map of size nside
lmax = 3*nside-1
# This gives you the unit vector for each pixel in the map
# (i.e. it's a [3, npix] array).
u = np.array(hp.pix2vec(nside, np.arange(npix)))
# This is the cosine of the angle between all pairs of pixels
# (i.e. an [npix, npix] array)
mu_ij = np.dot(u.T, u)
# Then you calculate the matrix of legendre polynomials
# Q matrices (shape [N_ell, npix, npix])
Qmats = np.array([(2*l+1)*legendre(l)(mu_ij)/(4*np.pi) for l in range(lmax+1)])

# Inverse noise covariances
inv_noise_ww = inv_cov_ww
inv_noise_gg = np.identity(npix)  # TODO: calculate it

# x's
x_w = np.array([x.dot(inv_cov_ww) for x in Qmats])
x_g = np.array([x.dot(inv_cov_gg) for x in Qmats])

# c's
c_ww = z_w.dot(Qmats).dot(z_w)
c_wg = z_w.dot(Qmats).dot(z_g)
c_gg = z_g.dot(Qmats).dot(z_g)

# F's
F_gg = np.array([[np.einsum('ij,ji', x1, x2) for x1 in x_g] for x2 in x_g])
F_wg = np.array([[np.einsum('ij,ji', x1, x2) for x1 in x_w] for x2 in x_g])
F_wg = (F_wg + F_wg.T)/2.
F_ww = np.array([[np.einsum('ij,ji', x1, x2) for x1 in x_w] for x2 in x_w])

# inv_F's
inv_F_gg = np.linalg.inv(F_gg)
inv_F_wg = np.linalg.inv(F_wg)
inv_F_ww = np.linalg.inv(F_ww)

# b's
b_gg = np.array([x.dot(inv_cov_gg).dot(inv_noise_gg) for x in x_g])
b_gg = b_gg.trace(axis1=1, axis2=2)
b_wg = np.zeros(lmax+1)
b_ww = x_w.trace(axis1=1, axis2=2)

# Cell's
cell_wg = inv_F_wg.dot(c_wg-b_wg)
cell_ww = inv_F_ww.dot(c_ww-b_ww)
cell_gg = inv_F_gg.dot(c_gg-b_gg)

print(cell_ww)
print(cell_wg)
print(cell_gg)
