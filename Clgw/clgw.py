import numpy as np
import healpy as hp
from scipy.special import legendre


def get_Q(nside):
    """ Q matrices for maps with resolution `nside`.

    Args:
        nside (int): resolution parameter.

    Returns:
        Array of shape [n_ell, npix, npix]
    """
    npix = hp.nside2npix(nside)
    # Compute q matrices
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
    Qmats = np.array([(2*ll+1)*legendre(ll)(mu_ij)/(4*np.pi)
                      for ll in range(lmax+1)])
    return Qmats


def get_fisher_and_noise_bias(invcov_1, invcov_2=None,
                              noicov=None, Qmats=None):
    """ Computes power spectrum Fisher matrix and noise bias (if needed)

    Args:
        invcov_1 (array_like): inverse covariance matrix for first map.
        invcov_2 (array_like): inverse covariance matrix for second map.
            I `None`, `invcov_2` will be used instead.
        noicov (array_like): noise covariance. Only needed if the noise
            bias is required. Otherwise the noise bias will be zero.
        Qmats (array_like): precomputed Q matrices (see `get_Q`). If `None`,
            they will be computed.

    Returns:
        Dictionary with entries `'fisher'` and `'pcl_nl'` containing the
        power spectrum Fisher matrix and the noise pseudo-Cl (i.e. before
        multiplying by the inverse Fisher matrix).
    """
    npix = invcov_1.shape[0]
    nside = hp.npix2nside(npix)

    if Qmats is None:
        Qmats = get_Q(nside)
    n_ell = len(Qmats)

    # These matrices contain Q.C_i^-1
    QiC1 = np.array([q.dot(invcov_1) for q in Qmats])
    if invcov_2 is None:
        invcov_2 = invcov_1
        QiC2 = QiC1
    else:
        QiC2 = np.array([q.dot(invcov_2) for q in Qmats])

    # Compute Fisher matrix
    fisher = np.array([[np.einsum('ij,ji', x1, x2)
                        for x1 in QiC1] for x2 in QiC2])
    fisher = 0.5*(fisher + fisher.T)

    # Copute noise bias if needed
    if noicov is None:
        nl = np.zeros(n_ell)
    else:
        NiC1 = np.einsum('ij,jk', invcov_1, noicov)
        nl = np.array([np.einsum('ij,ji', x1, NiC1) for x1 in QiC1])

    return {'fisher': fisher, 'pcl_nl': nl}


def get_pcl(iv_map1, iv_map2=None, Qmats=None):
    """ Returns the pseudo-Cl (i.e. before multiplying by the
    inverse Fisher matrix) of two maps.

    Arguments:
        iv_map1 (array_like): inverse-variance-weighted map.
        iv_map2 (array_like): second IV-weighted map. If `None`
            `iv_map1` will be used.
        Qmats (array_like): precomputed Q matrices (see `get_Q`).
            If `None`, they will be computed.

    Returns:
        Dictionary with a single entry `'pcl_cl'` containing
        the pseudo-Cl.
    """
    if iv_map2 is None:
        iv_map2 = iv_map1
    if Qmats is None:
        nside = hp.npix2nside(iv_map1.size)
        Qmats = get_Q(nside)
    cl = iv_map1.dot(Qmats).dot(iv_map2)
    return {'pcl_cl': cl}


def get_IV_map(mp, cov, is_map_invw=False,
               nmode_map_remove=0, is_cov_inv=False):
    """ Returns the information needed to compute a map's
    power spectrum.

    Arguments:
        mp (array_like): map. It may be the map itself or the
            inverse-variance-weighted map.
        cov (array_like): map covariance. It may be the
            covariance itself or its inverse.
        is_map_invw (bool): if `True`, `mp` contains the inverse-
            variance-weighted map.
        is_cov_inv (bool): if `True`, `cov` is actually the
            inverse covariance.
        nmode_map_remove (int): number of PCA components to
            remove from the map and covariance.

    Returns:
        iv_mp, inv_cov: inverse-variance-weighted map and
            corresponding inverse covariance to be used when
            computing power spectra.
    """
    if is_cov_inv:
        invcov = cov.copy()
    else:
        invcov = np.linalg.inv(cov)

    if nmode_map_remove > 0:
        w, v = np.linalg.eigh(invcov)
        filt = np.ones_like(w)
        filt[:nmode_map_remove] = 0
        # N^-1 = v.W.vT = sum_a v_ia w_a v_ja
        # N' = v.(F/W).vT
        # N'^-1 = v.(F*W).vT
        # mc = N' miv
        #    = v.(F/W).vT miv
        # <mc mcT> = N'<miv mivT> N'
        #          = v.(F/W).vT v.W.vT v.(F/W).vT
        #          = N'
        # mciv = N'^-1 mp_clean
        #      = v.(FW).vT v.(F/W).vT miv
        #      = v.F.vT miv
        invcov = np.einsum('ia,a,ja', v, w*filt, v)
        mp_filt = np.dot(v, filt*np.dot(mp, v))
    else:
        mp_filt = mp

    if is_map_invw:
        mp_inv = mp_filt
    else:
        mp_inv = np.dot(invcov, mp_filt)
    return mp_inv, invcov
