import numpy as np
import matplotlib.pyplot as plt

comp_1 = {
    'ells': np.load('comp_1.npz')['ells'],
    'cell_ww': np.load('comp_1.npz')['cell_ww'],
    'cell_ww_std': np.load('comp_1.npz')['cell_ww_std'],
}

comp_2 = {
    'ells': np.load('comp_2.npz')['ells'],
    'cell_ww': np.load('comp_2.npz')['cell_ww'],
    'cell_ww_std': np.load('comp_2.npz')['cell_ww_std'],
}


plt.figure(figsize=(10, 7))
plt.errorbar(comp_1['ells'], comp_1['cell_ww'], yerr=comp_1['cell_ww_std'], fmt='r.', label='comp_1')
plt.errorbar(comp_1['ells'], -comp_1['cell_ww'], yerr=comp_1['cell_ww_std'], fmt='rv', label='comp_1')
plt.errorbar(comp_2['ells'], comp_2['cell_ww'], yerr=comp_2['cell_ww_std'], fmt='b.', label='comp_2')
plt.errorbar(comp_2['ells'], -comp_2['cell_ww'], yerr=comp_2['cell_ww_std'], fmt='bv', label='comp_2')
plt.yscale('log')
plt.xlabel(r'$\ell$', fontsize=15)
plt.ylabel(r'$C^{ww}_\ell$', fontsize=15)
plt.legend(loc='best')
plt.savefig('./comp_cl_ww.pdf')

print('This should be zero (or constant if there is an overall factor difference):')
print(comp_1['cell_ww']/comp_2['cell_ww']-1.)
