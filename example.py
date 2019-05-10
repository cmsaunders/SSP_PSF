# The following shows a simple example of how to use psf.py to simulate
# star vignettes.

import psf
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Use the output of the psf fitting code to gernerate realistic vignettes:
psf_res = '/sps/snls13/HSC/prod.2017-11/dbimfast_OSHOYAY/psfpy_42UNBAI/data/332672_SSP_UDEEP_COSMOS_57842_0_i2_108212/108212p042/psf.dat'

psf_dict_file = '/sps/snls13/HSC/prod.2017-11/dbimfast_OSHOYAY/psfpy_42UNBAI/data/332672_SSP_UDEEP_COSMOS_57842_0_i2_108212/108212p042/test_out.pkl'

with open(psf_res, 'r') as f:
    psf_params = f.readlines()

with open(psf_dict_file, 'r') as f:
    psf_res = pickle.load(f)

hsize = int(psf_params[2].split()[0])
xsize, ysize = psf_params[4].strip().split()
xsize = int(xsize)
ysize = int(ysize)

# Number of stars to simulate:
n_stars = len(psf_res['xc'])#10

# Put the location of your simulated stars here:
xc = np.random.randn(n_stars) + np.random.choice(xsize, size=n_stars)
yc = np.random.randn(n_stars) + np.random.choice(ysize, size=n_stars)

# Generate Moffat PSF vignettes:
w_vars = np.hstack([psf_res['wxx'], psf_res['wyy'], psf_res['wxy']])
m = psf.MoffatPSF(hsize, xc, yc, xmax=xsize, ymax=ysize)
m.set_moffat(w_vars)
m.set_phi()
# The Moffat PSF vignettes:
phi = m.phi

# Generate Moffat + Discrete Residual vignettes:
resid_vars = np.hstack([psf_res['r'].ravel(), psf_res['c']]) 
p = psf.PSFResiduals(hsize, xc, yc, phi.reshape(n_stars, -1), xmax=xsize,
                     ymax=ysize)
# The final PSF vignettes:
psi = p.psi_model(resid_vars).reshape(n_stars, p.nd, p.nd)
