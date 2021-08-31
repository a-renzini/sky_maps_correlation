import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.special import legendre


# Parse arguments
parser = argparse.ArgumentParser(
    'Calculate galaxy and SGWB auto- and cross-correlations.'
)
parser.add_argument('--w_file', '-w', type=str, default=None, required=True,
                    help='SGWB maps file')
parser.add_argument('--g_file', '-g', type=str, default=None, required=True,
                    help='Galaxy maps file')
parser.add_argument('--s_file', '-s', type=str, default=None, required=True,
                    help='Galaxy simulation maps file')
parser.add_argument('--verbose', '-v', default=False, action='store_true',
                    help='Verbose mode')
args = parser.parse_args()

if args.verbose:
    time_start = time.time()


# Load maps covariances and maps
inv_cov_ww = np.load(args.w_file)['M_p_pp'][:, :, 0, 0].astype('float64')
inv_cov_gg = np.load(args.g_file)['inv_cov']
z_w = np.load(args.w_file)['Z_p'][:, 0].astype('float64')
z_g = np.dot(inv_cov_gg, np.load(args.g_file)['map'])
zs_g = np.dot(inv_cov_gg, np.load(args.s_file)['maps'].T).T
nmaps_g = len(zs_g)

# To calculate the Qmats we assume that all the maps have the same resolution
# Number of pixels
npix = len(z_w)
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
inv_noise_gg = np.identity(npix)  # TODO: estimate it

# x's
x_w = np.array([x.dot(inv_cov_ww) for x in Qmats])
x_g = np.array([x.dot(inv_cov_gg) for x in Qmats])

# F's
F_ww = np.array([[np.einsum('ij,ji', x1, x2) for x1 in x_w] for x2 in x_w])
F_wg = np.array([[np.einsum('ij,ji', x1, x2) for x1 in x_w] for x2 in x_g])
F_wg = (F_wg + F_wg.T)/2.
F_gg = np.array([[np.einsum('ij,ji', x1, x2) for x1 in x_g] for x2 in x_g])

# inv_F's
inv_F_ww = np.linalg.inv(F_ww)
inv_F_wg = np.linalg.inv(F_wg)
inv_F_gg = np.linalg.inv(F_gg)

# b's
b_ww = x_w.trace(axis1=1, axis2=2)
b_wg = np.zeros(lmax+1)
b_gg = np.array([x.dot(inv_cov_gg).dot(inv_noise_gg) for x in x_g])
b_gg = b_gg.trace(axis1=1, axis2=2)

if args.verbose:
    dt = time.time() - time_start
    print('Preliminary stuff computed in {:.4f} sec'.format(dt))


def get_cls(z_w, z_g, Q=Qmats,
            b_ww=b_ww, b_wg=b_wg, b_gg=b_gg,
            iF_ww=inv_F_ww, iF_wg=inv_F_wg, iF_gg=inv_F_gg):
    # c's
    c_ww = z_w.dot(Q).dot(z_w)
    c_wg = z_w.dot(Q).dot(z_g)
    c_gg = z_g.dot(Q).dot(z_g)
    # Cell's
    cell_ww = iF_ww.dot(c_ww-b_ww)
    cell_wg = iF_wg.dot(c_wg-b_wg)
    cell_gg = iF_gg.dot(c_gg-b_gg)
    return cell_ww, cell_wg, cell_gg


# Get Cl's
cell_ww, cell_wg, cell_gg = get_cls(z_w, z_g)

# Main loop for the simulations
cell_ww_sim = np.zeros((nmaps_g, lmax+1))
cell_wg_sim = np.zeros((nmaps_g, lmax+1))
cell_gg_sim = np.zeros((nmaps_g, lmax+1))
time_start = time.time()
for n, z_g in enumerate(zs_g):
    cell_ww_sim[n], cell_wg_sim[n], cell_gg_sim[n] = get_cls(z_w, z_g)
    if args.verbose:
        if np.mod(n, 50) == 0:
            dt = time.time() - time_start
            print('----> Cells {}/{} in {:.4f} sec'.format(n+1, len(zs_g), dt))
            time_start = time.time()


if args.verbose:
    time_start = time.time()

# Get mean and variance
ells = np.arange(lmax+1)
cell_ww_mean = cell_ww_sim.mean(axis=0)
cell_wg_mean = cell_wg_sim.mean(axis=0)
cell_gg_mean = cell_gg_sim.mean(axis=0)
cell_ww_std = cell_ww_sim.std(axis=0)
cell_wg_std = cell_wg_sim.std(axis=0)
cell_gg_std = cell_gg_sim.std(axis=0)
cell_wg_cov = np.cov(cell_wg_sim.T)

if args.verbose:
    dt = time.time() - time_start
    print('Mean and variance computed in {:.4f} seconds'.format(dt))

plt.figure(figsize=(10, 7))
plt.errorbar(ells, cell_wg, yerr=cell_wg_std, fmt='r.')
plt.errorbar(ells, -cell_wg, yerr=cell_wg_std, fmt='rv')
plt.plot(ells, cell_wg_mean, 'b.', label='mean sims')
plt.plot(ells, -cell_wg_mean, 'bv')
plt.yscale('log')
plt.xlabel(r'$\ell$', fontsize=15)
plt.ylabel(r'$C^{g\Omega}_\ell$', fontsize=15)
plt.legend(loc='best')
plt.savefig('./cl_wg.pdf')
print('Success!')
