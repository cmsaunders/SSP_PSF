#!/usr/bin/env python

import numpy as np
from astropy.io import fits
import pickle
import os.path as op
import time
import matplotlib.pyplot as plt
from croaks import NTuple
import scipy.linalg as SL
import scipy.sparse as SS
import scipy.optimize as SO
from scipy.sparse import linalg as SSLA
from scipy.interpolate import RectBivariateSpline as RBS
import sksparse.cholmod 
from numpy.polynomial.legendre import leggauss
import argparse
import psutil
import os

cholesky = sksparse.cholmod.cholesky
process = psutil.Process(os.getpid())


def get_sxy(path):
    
    sstars = NTuple.fromfile(op.join(path, 'standalone_stars.list'))
    seeing = float(sstars.keys['SEEING'])
    x = sstars['x']
    y = sstars['y']
    f = sstars['flux']
    fluxmax = sstars['fluxmax']
    return seeing, x, y, f, fluxmax


def get_data(path, xc, yc, hsize):

    data_file = fits.getdata(op.join(path, 'calibrated.fz'))
    weight_file = fits.getdata(op.join(path, 'weight.fz'))
    if op.isfile(op.join(path, 'satur.fits.gz')):
        satur = fits.getdata(op.join(path, 'satur.fits.gz'))
        weight_file *= (1 - satur)

    gain = fits.getval(op.join(path, 'calibrated.fz'), 'Gain', 1)

    x_int = np.round(xc).astype(int)
    y_int = np.round(yc).astype(int)

    n_stars = len(xc)
    vsize = 2 * hsize + 1

    ymax, xmax = data_file.shape

    z = np.zeros(n_stars, dtype=int)
    ymins = np.max(np.vstack([(y_int - hsize), z]), axis=0)
    ymaxs = np.min(np.vstack([(y_int + hsize) - ymax, z]), axis=0) + ymax + 1
    yminp = -1 * (np.min(np.vstack([(y_int - hsize), z]), axis=0))
    ymaxp = (-1 * np.max(np.vstack([(y_int + hsize) +1 - ymax, z]), axis=0)
                  + vsize)
    xmins = np.max(np.vstack([(x_int - hsize), z]), axis=0)
    xmaxs = np.min(np.vstack([(x_int + hsize) - xmax, z]), axis=0) + xmax + 1
    xminp = -1 * (np.min(np.vstack([(x_int - hsize), z]), axis=0))
    xmaxp = (-1 * np.max(np.vstack([(x_int + hsize) +1 - xmax, z]), axis=0)
                  + vsize)

    data_cube = np.zeros((n_stars, vsize, vsize))
    weight_cube = np.zeros((n_stars, vsize, vsize))

    for i in range(n_stars):
        data_cube[i, yminp[i]:ymaxp[i],
                  xminp[i]:xmaxp[i]] = data_file[ymins[i]:ymaxs[i],
                                                 xmins[i]:xmaxs[i]]
        weight_cube[i, yminp[i]:ymaxp[i],
                    xminp[i]:xmaxp[i]] = weight_file[ymins[i]:ymaxs[i],
                                                     xmins[i]:xmaxs[i]]

    n_stars = len(xc)

    data = data_cube.reshape(n_stars,-1) 
    weights = weight_cube.reshape(n_stars, -1) 

    corners = np.array([[0, 0], [0, xmax], [ymax,ymax], [ymax, 0]])
    
    return data, weights, corners, gain, data_file, weight_file

def vignette(data, weight, hsize, x, y):

    hsize = int(hsize)
    y_int = int(np.round(y))
    x_int = int(np.round(x))

    vsize = 2 * hsize + 1

    y_mx, x_mx = data.shape

    z = 0
    ymin = np.max([(y_int - hsize), z])
    ymax = np.min([(y_int + hsize) - y_mx, z]) + y_mx + 1
    yminp = -1 * np.min([(y_int - hsize), z])
    ymaxp = -1 * np.max([(y_int + hsize) +1 - y_mx, z]) + vsize
    xmin = np.max([(x_int - hsize), z])
    xmax = np.min([(x_int + hsize) - x_mx, z]) + x_mx + 1
    xminp = -1 * np.min([(x_int - hsize), z])
    xmaxp = (-1 * np.max([(x_int + hsize) +1 - x_mx, z]) + vsize)

    data_vign = np.zeros((vsize, vsize))
    weight_vign = np.zeros((vsize, vsize))
    
    try:
        data_vign[yminp:ymaxp, xminp:xmaxp] = data[ymin:ymax,xmin:xmax]
    except:
        print x, y
        print ymin, ymax, xmin, xmax
        print yminp, ymaxp, xminp, xmaxp    
        1/0
    weight_vign[yminp:ymaxp, xminp:xmaxp] = weight[ymin:ymax, xmin:xmax]

    return data_vign.ravel(), weight_vign.ravel()


quadrature_order = 3
intp,intw = leggauss(quadrature_order)
intp *= 0.5 ; intw *= 0.5
intgx,intgy = np.meshgrid(intp,intp)
intwx,intwy = np.meshgrid(intw,intw)
intww = (intwx*intwy).ravel()
def integ_in_pixels(x, y, f):
    """Integrate a 2D function in pixels
    (Totally borrowed from saunerie)
    Given a function f (typically, a psf profile), and a series of
    positions (x,y), evaluate the integral of the function on the
    pixels corresponding to each of the positions (x,y) specified in
    argument.

    Args
       x (ndarray of floats): x-positions, in pixels.
       y (ndarray of floats): y-positions, in pixels.
       f (callable): function to integrate
       quadrature_order (int, optional): order of the gaussian quadrature to use.

    Returns:
       ndarray of floats: integrated PSF value.

    .. note: 
       the function is written to preserve the shape of x and y
    """
    # put this integration code in cache
    """
    p,w = leggauss(quadrature_order)
    p *= 0.5 ; w *= 0.5
    gx,gy = np.meshgrid(p,p)
    wx,wy = np.meshgrid(w,w)
    ww = (wx*wy).ravel()
    xc = x.reshape(x.shape + (1,1,)) + gx
    yc = y.reshape(y.shape + (1,1,)) + gy
    """
    xc = x.reshape(x.shape + (1,1,)) + intgx
    yc = y.reshape(y.shape + (1,1,)) + intgy
    v = f(xc,yc)
    v = v.reshape(v.shape[0], v.shape[1], -1)

    return np.dot(v, intww)

class Star(object):

    def __init__(self, data, weight, inv_weight, x, y, gain, fluxmax, nd):

        self.data = data
        self.weight = weight
        self.init_weight_inv = inv_weight
        
        self.hsize = (nd - 1)/2.
        self.gain = gain
        self.fluxmax = fluxmax

        self.x = x
        self.y = y

        self.xi_int = np.tile(np.arange(-hsize, hsize+1), (nd, 1))
        self.yi_int = np.tile(np.arange(-hsize, hsize+1)[:, None], (1, nd))
        self.xi = self.xi_int.ravel()
        self.yi = self.yi_int.ravel()

        ymax, xmax = data.shape
        self.ax = 2./ xmax
        self.bx = 1 - self.ax * xmax
        self.ay = 2./ ymax
        self.by = 1 - self.ay * ymax

    def set_globalpsfparams(self, params):

        self.g_wxx = params[0]
        self.g_wyy = params[1]
        self.g_wxy = params[2]

    def phi_grad_pos(self, dx, dy, wxx, wyy, wxy):

        norm = (wxx * wyy - wxy**2)**0.5 * (2.5 - 1)/np.pi
        fact = (1 + wxx * dx**2 + wyy * dy**2 + 2 * wxy * dx * dy)
        grad1 = -2.5 * norm * fact**-3.5  
        gradx = -2 * wxx * dx - 2 * wxy * dy
        grady = -2 * wyy * dy - 2 * wxy * dx
        return grad1 * np.array([gradx, grady])

    def phi_grad_par(self, dx, dy, wxx, wyy, wxy):

        norm_arg = (wxx * wyy - wxy**2)**0.5
        norm_c = (2.5 - 1)/np.pi
        fact = (1 + wxx * dx**2 + wyy * dy**2 + 2 * wxy * dx * dy)
        grad1 = -2.5 * norm_c * norm_arg * fact**-3.5 
        grad2 = 0.5 * norm_c * norm_arg**-1 * fact**-2.5
        dwxx = grad1 * dx**2 + grad2 * wyy
        dwyy = grad1 * dy**2 + grad2 * wyy
        dwxy = grad1 * 2 * dx * dy + grad2 * -2 * wxy
        return np.array([dwxx, dwyy, dwxy])

    def computeABchi2(self, vars, only_chi=False,plot=False):
        wxx, wyy, wxy, flux, new_x, new_y, fluxforweight = vars

        xc = new_x - np.round(new_x)
        yc = new_y - np.round(new_y)

        dx = self.xi - xc
        dy = self.yi - yc
        dxi = self.xi_int - xc
        dyi = self.yi_int - yc
        fact = (1 + wxx * dx**2 + wyy * dy**2 + 2 * wxy * dx * dy)
        norm = (wxx * wyy - wxy**2)**0.5 * (2.5 - 1)/np.pi
        def tmp(x, y):
            ft = (1 + wxx * x**2 + wyy * y**2 + 2 * wxy * x * y)
            return norm * ft**-2.5

        phi = integ_in_pixels(dxi, dyi, tmp).ravel()
        self.phi = phi
        
        self.data_slice, self.weight_slice = vignette(self.data, self.weight, 
                                                      self.hsize, new_x, new_y)

        if plot:
            nd = int(len(dx)**0.5)
            d_s = self.data_slice.reshape(nd, nd)
            m_s = (flux * phi).reshape(nd, nd)
            r_s = (d_s - m_s)
            plt.imshow(np.hstack([d_s, m_s, r_s]), interpolation='nearest')
            plt.show()

        res = (self.data_slice - flux * phi)

        weight = (self.weight_slice**-1 + fluxforweight * phi /self.gain)**-1
        dweight = - weight**2 * res**2 * fluxforweight / gain

        chi2 = (weight * res**2).sum()

        if only_chi:
            return chi2

        def tmp_xpos(x, y):
            return self.phi_grad_pos(x, y, wxx, wyy, wxy)[0]
        def tmp_ypos(x, y):
            return self.phi_grad_pos(x, y, wxx, wyy, wxy)[1]
        def tmp_wxx(x, y):
            return self.phi_grad_par(x, y, wxx, wyy, wxy)[0]
        def tmp_wyy(x, y):
            return self.phi_grad_par(x, y, wxx, wyy, wxy)[1]
        def tmp_wxy(x, y):
            return self.phi_grad_par(x, y, wxx, wyy, wxy)[2]

        dpos = np.array([integ_in_pixels(dxi, dyi, tmp_xpos).ravel(),
                         integ_in_pixels(dxi, dyi, tmp_ypos).ravel()])
        dpar = np.array([integ_in_pixels(dxi, dyi, tmp_wxx).ravel(),
                         integ_in_pixels(dxi, dyi, tmp_wyy).ravel(),
                         integ_in_pixels(dxi, dyi, tmp_wxy).ravel()])


        h_vector = np.zeros((6, len(self.xi)))
        grad_w = np.zeros((6, len(self.xi)))

        h_vector[:3] = dpar * flux 
        h_vector[3] = phi
        h_vector[4:] = dpos * flux
        
        grad_w[:3] = dpar * dweight
        grad_w[4:] = dpos * dweight
        # d_chi:
        B = (-2 * weight * res * h_vector + grad_w).sum(axis=1)
        # hessian_chi:
        A = (2 * weight * h_vector[:, None,:] * h_vector[None,:, :]).sum(axis=2)

        return A, B, chi2

    def fit_all_params(self, init_flux, init_w, max_iter=30):
        # Minimize equation 5 from the note:

        self.flux = init_flux
        self.wxx, self.wyy = init_w, init_w
        self.wxy = 0
        
        init_vars = np.array([self.wxx, self.wyy, self.wxy, self.flux, 
                              self.x, self.y, self.flux])
        vars = np.copy(init_vars)

        min_diff = [0.1, 0.01]

        for l in range(2): # 2 loops to fit flux, correct weight, refit flux

            oldchi = 1e30
            dchi2 = 1e30
            iter = 0

            # Gauss-Newton minimizer
            while (dchi2 > min_diff[l]) & (iter < max_iter):

                A, B, chi = self.computeABchi2(vars)

                factor = SL.cho_factor(A)
                solution = SL.cho_solve(factor, B)

                # get bracket for Brent linesearch:
                init_step = 3
                c = np.nan
                while np.isnan(c):
                    new_vars = np.copy(vars)
                    new_vars[:6] -= init_step * solution
                    try:
                        c = self.computeABchi2(new_vars, only_chi=True)
                    except:
                        c = np.nan

                    if np.isnan(c):
                        init_step *= 0.6

                # do Brent line search to get step along gradient:
                def brent_lnsrch(x):
                    new_vars = np.copy(vars)
                    new_vars[:6] -= x * solution
                    return self.computeABchi2(new_vars, only_chi=True)
                    
                try:
                    step_fit = SO.brent(brent_lnsrch, brack=(0, init_step), 
                                        tol=1e-4, full_output=1)
                    chi2 = step_fit[1]
                except:
                    chi2 = np.nan

                # For certain scipy versions Brent search goes outside bounds and finds NaN!
                # Check for this and, if chi2=nan, just use initial step
                if not np.isnan(chi2):
                    vars[:6] -= step_fit[0] * solution
                else:
                    vars[:6] -= 0.5 * init_step * solution
                    chi2 = self.computeABchi2(vars, only_chi=True)

                dchi2 = np.copy(abs(chi2 - oldchi))

                iter += 1
                oldchi = np.copy(chi2)

                
            vars[6] = np.copy(vars[3])

        self.chi2 = chi2    
        self.covariance = SL.inv(A)
        self.flux = vars[3]
        self.eflux = self.covariance[3,3]**0.5
        self.x = vars[4]
        self.y = vars[5]
        self.vxc = self.covariance[4,4]
        self.vyc = self.covariance[5,5]

        self.wxx, self.wyy, self.wxy = np.copy(vars[:3])
        self.psfparams = np.copy(vars[:3])
        w_covs = self.covariance[:3,:3]
        self.paramweight = SL.inv(w_covs)

    def computeABchi2_pos(self, vars, wxx, wyy, wxy, fluxforweight, res_fns, c,
                          only_chi=False, plot=False):

        flux, new_x, new_y = vars

        xc = new_x - np.round(new_x)
        yc = new_y - np.round(new_y)
        dx = self.xi - xc
        dy = self.yi - yc
        dxi = self.xi_int - xc
        dyi = self.yi_int - yc

        fact = (1 + wxx * dx**2 + wyy * dy**2 + 2 * wxy * dx * dy)
        norm = (wxx * wyy - wxy**2)**0.5 * (2.5 - 1)/np.pi
        def tmp(x, y):
            ft = (1 + wxx * x**2 + wyy * y**2 + 2 * wxy * x * y)
            return norm * ft**-2.5

        phi = integ_in_pixels(dxi, dyi, tmp).ravel()
        self.phi = phi
        orig_x = np.arange(-self.hsize, self.hsize+1)
        r_conv = np.array([fn(orig_x - yc, orig_x - xc).ravel() 
                           for fn in res_fns])

        pos_array = np.array([1, new_x, new_y])
        pos_array = np.array([1, self.ax * new_x + self.bx,
                              self.ay * new_y + self.by])
        resid = (r_conv * pos_array[:,None]).sum(axis=0)

        phi_part = phi * (1 - (c * pos_array).sum())
        
        psi = phi_part + resid

        self.data_slice, self.weight_slice = vignette(self.data, self.weight,
                                                      self.hsize, new_x, new_y)

        if plot:
            nd = int(len(dx)**0.5)
            d_s = self.data_slice.reshape(nd, nd)
            m_s = (flux * psi).reshape(nd, nd)
            r_s = (d_s - m_s)
            plt.imshow(np.hstack([d_s, m_s, r_s]), interpolation='nearest')
            plt.show()

        res = (self.data_slice - flux * psi)
        weight = (self.weight_slice**-1 + fluxforweight * psi /self.gain)**-1
        dweight = - weight**2 * res**2 * fluxforweight / gain

        chi2 = (weight * res**2).sum()

        if only_chi:
            return chi2

        def tmp_xpos(x, y):
            return self.phi_grad_pos(x, y, wxx, wyy, wxy)[0]
        def tmp_ypos(x, y):
            return self.phi_grad_pos(x, y, wxx, wyy, wxy)[1]

        dpos = np.array([integ_in_pixels(dxi, dyi, tmp_xpos).ravel(),
                         integ_in_pixels(dxi, dyi, tmp_ypos).ravel()])

        # 3 = 1 flux + 2 position params
        h_vector = np.zeros((3, len(self.xi)))
        grad_w = np.zeros((3, len(self.xi)))

        h_vector[0] = psi
        h_vector[1:] = dpos * flux
        
        grad_w[1:] = dpos * dweight
        # d_chi:
        B = (-2 * weight * res * h_vector + grad_w).sum(axis=1)
        # hessian_chi:
        A = (2 * weight * h_vector[:, None,:] * h_vector[None,:, :]).sum(axis=2)

        return A, B, chi2

    def fit_pos(self, init_flux, res_fns, c, use_global_w_params=True):
        
        #init_vars = np.array([init_flux, self.xc, self.yc])
        init_vars = np.array([init_flux, self.x, self.y])

        if use_global_w_params:
            wxx = self.g_wxx
            wyy = self.g_wyy
            wxy = self.g_wxy
        else:
            wxx = self.wxx
            wyy = self.wyy
            wxy = self.wxy

        vars = np.copy(init_vars)

        min_diff = [0.1, 0.01]
        max_iter=30
        
        fluxforweight = init_flux

        for l in range(2): # 2 loops to fit flux, correct weight, refit flux
            oldchi = 1e30
            dchi2 = 1e30
            iter = 0

            # Gauss-Newton minimizer
            while (dchi2 > min_diff[l]) & (iter < max_iter):

                A, B, chi = self.computeABchi2_pos(vars, wxx, wyy, wxy, 
                                                   fluxforweight, res_fns, c)
                try:
                    factor = SL.cho_factor(A)
                except:
                    print self.flux
                    print init_vars
                    print vars
                    print A
                    print B
                    raise ValueError("Problem with Hessian of chi2, look at A 
                                     because this should not happen")

                solution = SL.cho_solve(factor, B)

                # get bracket:
                init_step = 3
                tmp_chi = np.nan
                while np.isnan(tmp_chi):
                    new_vars = np.copy(vars)
                    new_vars -= init_step * solution
                    tmp_chi = self.computeABchi2_pos(new_vars, wxx, wyy, wxy, 
                                                     fluxforweight, res_fns,
                                                     c, only_chi=True)
                    if np.isnan(tmp_chi):
                        init_step *= 0.6

                # do Brent line search to get step along gradient:
                def brent_lnsrch(x):
                    new_vars = np.copy(vars)
                    new_vars -= x * solution
                    return self.computeABchi2_pos(new_vars, wxx, wyy,
                                                  wxy, fluxforweight,
                                                  res_fns, c, only_chi=True)

                try:
                    step_fit = SO.brent(brent_lnsrch, brack=(0, init_step), 
                                        tol=1e-4, full_output=1)
                    chi2 = step_fit[1]
                except:
                    chi2 = np.nan

                # Sometimes Brent search goes outside bounds and finds NaN!
                # Check for this and, if chi2=nan, just use initial step
                if not np.isnan(chi2):
                    vars -= step_fit[0] * solution
                else:
                    vars -= 0.5 * init_step * solution
                    chi2 = self.computeABchi2_pos(vars, wxx, wyy,
                                                  wxy, fluxforweight,
                                                  res_fns, c, only_chi=True)

                dchi2 = np.copy(abs(chi2 - oldchi))

                iter += 1
                oldchi = np.copy(chi2)

            fluxforweight = vars[0]
        # Reset phi, etc.:
        self.computeABchi2_pos(vars, wxx, wyy, wxy,
                               fluxforweight, res_fns, c, only_chi=True)

        self.chi2 = chi2    
        self.covariance = SL.inv(A)
        self.flux = vars[0]
        self.eflux = self.covariance[0,0]**0.5
        self.x = vars[1]
        self.y = vars[2]
        self.xc = self.x - np.round(self.x)
        self.yc = self.y - np.round(self.y)
        self.vxc = self.covariance[1,1]
        self.vyc = self.covariance[2,2]


class MoffatPSF(object):

    def __init__(self, hsize, xc, yc, full_data, full_weights, gain, fluxmax, degree=1):

        self.name = "MOFFAT25"

        self.xc = xc
        self.yc = yc

        self.n_stars = len(xc)
        # Set up indices for the PSF vignettes
        self.nd = hsize * 2 + 1
        self.xi = np.tile(np.arange(-hsize, hsize+1), (self.nd, 1))
        self.yi = np.tile(np.arange(-hsize, hsize+1)[:, None], (1, self.nd))
        
        xdiffs = self.xc - np.round(self.xc) 
        ydiffs = self.yc - np.round(self.yc) 

        self.x_inds = self.xi[None,:,:] - xdiffs[:,None,None]
        self.y_inds = self.yi[None,:,:] - ydiffs[:,None,None]

        # Load data:
        self.full_data = full_data
        self.full_weights = full_weights
        self.outliers = np.zeros(self.n_stars, dtype=bool)
        self.use_outliers = np.zeros(self.n_stars, dtype=bool)
        self.ymax, self.xmax = full_data.shape

        self.gain = gain
        self.fluxmax = fluxmax

        # Set up parameters to normalize the CCD-position dependent polynomials:
        self.ax = 2./self.xmax
        self.bx = 1 - self.ax * self.xmax
        self.ay = 2./self.ymax
        self.by = 1 - self.ay * self.ymax

        # Set up variables to help calculate the derivative of the chi2 function
        der = np.array([np.ones(self.n_stars), self.ax * self.xc + self.bx, 
                        self.ay * self.yc + self.by])
        deg_dict = {0: 1, 1: 3, 2: 6}
        self.degree = deg_dict[degree]
        self.wder = der[:self.degree].T

        # Calculate the number of variables:
        self.nvars = self.n_stars + self.degree * 3

    def initparamsfromseeing(self, seeing):

        self.wxx_vector = np.zeros(self.degree)
        self.wyy_vector = np.zeros(self.degree)
        self.wxy_vector = np.zeros(self.degree)

        init_w = (0.6547/seeing)**2
        self.wxx_vector[0] = init_w
        self.wyy_vector[0] = init_w


    def reset_pos(self):
        # Given changed self.xc and self.yc, reset helper variables

        xdiffs = self.xc - np.round(self.xc) 
        ydiffs = self.yc - np.round(self.yc) 

        self.x_inds = self.xi[None,:,:] - xdiffs[:,None,None]
        self.y_inds = self.yi[None,:,:] - ydiffs[:,None,None]

        der = np.array([np.ones(self.n_stars), self.ax * self.xc + self.bx, 
                        self.ay * self.yc + self.by])
        self.wder = der[:self.degree].T


    def set_moffat(self, vars, expn=-2.5):

        self.wxx_vector = vars[:self.degree]
        self.wyy_vector = vars[self.degree:-self.degree]
        self.wxy_vector = vars[-self.degree:]
        self.wxx = (self.wxx_vector[None,:] * self.wder).sum(axis=1)
        self.wyy = (self.wyy_vector[None,:] * self.wder).sum(axis=1)
        self.wxy = (self.wxy_vector[None,:] * self.wder).sum(axis=1)

        self.det = self.wxx * self.wyy - self.wxy**2
        self.norm = np.sqrt(np.fabs(self.det)) * (-expn - 1) / np.pi

    def moffat(self, x, y, i, expn=-2.5):
        
        fact = (1 + x**2 * self.wxx[i] + y**2 * self.wyy[i] + 
                2 * x * y * self.wxy[i])

        return self.norm[i] * np.power(fact, expn)
        
    def moffat_integ(self, x, y, i, quadrature_order=4, expn=-2.5):
        p,w = leggauss(quadrature_order)
        p *= 0.5 ; w *= 0.5
        gx,gy = np.meshgrid(p,p)
        wx,wy = np.meshgrid(w,w)
        ww = (wx*wy).ravel()
        xc = x.reshape(x.shape + (1,1,)) + gx
        yc = y.reshape(y.shape + (1,1,)) + gy
        v = self.moffat(xc,yc, i, expn=-2.5)
        v = v.reshape(v.shape[0], v.shape[1], -1)

        return np.dot(v, ww)

    def check_corners(self, corners):
        
        okparams = True
        for c in corners:
            c_array = np.hstack([1, (c[1]-1000)/1000, (c[0]-2000)/1000])
            wxx = (self.wxx_vector * c_array).sum()
            wyy = (self.wyy_vector * c_array).sum()
            wxy = (self.wxy_vector * c_array).sum()
            if (wxx * wyy - wxy**2) < 0:
                okparams = False
        return okparams

    def set_phi(self):
        self.phi = np.array([self.moffat_integ(xi, yi, i) for (xi, yi, i)
                             in zip(self.x_inds, self.y_inds,
                                    np.arange(self.n_stars))])


    def remove_outliers(self, all_chi2s, sigmacut=4):

        chi2s = all_chi2s[~self.outliers]
        chi2_median = np.median(chi2s)
        chi2_sigma =((chi2s**2).sum()/len(chi2s) - np.mean(chi2s)**2)**0.5
        max_chi2 = chi2_median + chi2_sigma * sigmacut

        new_non_outliers = (all_chi2s <= max_chi2) | self.outliers

        print 'Max chi2', max_chi2, len(chi2s)
        print all_chi2s[~new_non_outliers]
        self.tmp_outliers = np.zeros(self.n_stars, dtype=bool)
        self.tmp_outliers[~new_non_outliers] = True
        self.use_outliers = self.tmp_outliers | self.outliers
 

    def moffat_fit(self, init_seeing=5, init_fluxes=None, 
                   max_iter=20, chi_tol=1e-5):

        # Get initial estimate of the Moffat PSF from the Gaussian seeing:
        self.initparamsfromseeing(init_seeing)
        # The initial parameters:
        w_vars = np.hstack([self.wxx_vector, self.wyy_vector, self.wxy_vector])
    
        # Solve equation 5 from the note:
        # Fit an elliptical Moffat PSF and position for each star,
        # skipping failure cases in this first estimate:
        Stars = []
        self.fluxes = np.zeros(self.n_stars)
        init_chis = np.zeros(self.n_stars)
        init_params = np.zeros((self.n_stars,3))
        init_paramweights = np.zeros((self.n_stars, 3, 3))
        init_param_sig = np.zeros((self.n_stars, 3))
        inv_weights = self.full_weights**-1
        for i in range(self.n_stars):

            S = Star(self.full_data, self.full_weights, inv_weights, 
                     self.xc[i], self.yc[i],
                     self.gain, self.fluxmax[i], self.nd)
            Stars.append(S)
            try:
                S.fit_all_params(init_fluxes[i], self.wxx_vector[0])
                self.fluxes[i] = S.flux
                self.xc[i] = S.x
                self.yc[i] = S.y
                init_chis[i] = S.chi2
                init_params[i] = S.psfparams
                init_paramweights[i] = S.paramweight
                init_param_sig[i] = np.diagonal(SL.inv(S.paramweight))**0.5
            except:
                print i, "outlier at %s, %s" % (S.x, S.y)
                self.fluxes[i] = init_fluxes[i]
                S.x = self.xc[i]
                S.y = self.yc[i]
                S.flux = init_fluxes[i]
                S.data_slice, S.weight_slice = vignette(self.full_data, 
                                                        self.full_weights, 
                                                        S.hsize, S.x, S.y) 
                init_chis[i] = 1e10
        # Reset some helper variables based on the new positions:
        self.reset_pos()
        self.stars = Stars
        
        init_chi = init_chis.sum()
        init_nstars = np.copy(self.n_stars)
        #print u"Initial \u03C7\u00B2=%s (\u03C7\u00B2/dof=%s)" \
        print u"Initial chi2=%s (chi2/dof=%s)" \
            % (init_chi, init_chi.sum()/(self.n_stars*self.nd**2
                                         - self.n_stars * (1 + 3)))
        print "Average ", init_chis.mean()

        tmp = self.remove_outliers(init_chis)
        print 'Stars passing cuts:', (~self.tmp_outliers).sum()

        # Fit equation 8 from the note:
        # Fit a first-degree polynomial to the individual stars' PSF parameters
        # as a function of the CCD position. Iterate, removing outliers, until there
        # are no more outliers. (Fit is otherwise linear)
        for i in range(max_iter):

            # Solve Ax=b using Newton-Raphson (where here A=hess_chi, b=dchi, x = d(new_vars)):
            dchi = np.zeros(9)
            hess_chi = np.zeros((9,9))
            b1 = np.array([init_paramweights[ip].dot(init_params[ip])
                           for ip in range(self.n_stars)])
            b = np.array([(self.wder[ip,None,:]*b1[ip,:, None]).ravel()
                           for ip in range(self.n_stars)])
                        
            b_noout = b[~self.use_outliers]
            dchi = 2 * b_noout.sum(axis=0)

            h = np.tile(self.wder, 3)
            big_weight = np.array([np.repeat(np.repeat(ip,3, axis=0), 3, axis=1)
                                   for ip in init_paramweights])
            
            hess_chi = 2 * (big_weight * h[:, :, None] * 
                            h[:,None,:])
            hess_chi = hess_chi[~self.use_outliers].sum(axis=0)

            factor = SL.cho_factor(hess_chi)
            new_vars = SL.cho_solve(factor, dchi)

            # Reset the parameters
            var_array = new_vars.reshape(3,3)
            w_params = np.array([var_array.dot(self.wder[s]) for s in 
                                 range(self.n_stars)])
            d_params = (w_params - init_params)

            param_chis = np.array([d_params[j].dot(init_paramweights[j].dot(d_params[j])) for j in range(self.n_stars)])

            print 'Mean before cuts', param_chis[~self.outliers].mean()
            ndof = (~self.use_outliers).sum() * 3 -  9 
            print 'chi/dof (before)', param_chis[~self.outliers].sum(), \
                ndof, param_chis[~self.outliers].sum()/ndof

            # Remove outlier stars:
            self.remove_outliers(param_chis)
            print 'New cuts', np.flatnonzero(self.tmp_outliers), \
                self.xc[self.tmp_outliers], self.yc[self.tmp_outliers]

            for xo in np.flatnonzero(self.tmp_outliers):
                print 'Erasing star at', self.xc[xo], self.yc[xo], \
                    self.fluxes[xo], param_chis[xo]

            if self.tmp_outliers.sum() == 0:
                print 'No more outliers, stop here'
                break
            else:
                self.outliers |= self.tmp_outliers

        # Given the fit parameters, get the analytic psf for each star:
        self.set_moffat(new_vars)
        self.set_phi()
        for s in range(self.n_stars):
            self.stars[s].set_globalpsfparams(w_params[s])
        
        print 'Initial PSF parameters:'
        print w_vars
        print 'Final PSF parameters:'
        print new_vars
        covariance = SL.inv(hess_chi)
        print 'PSF error: ', np.diagonal(covariance)**0.5
        self.covariance = covariance


class PSFResiduals(object):

    def __init__(self, hsize, xc, yc, phi, stars, gain, xmax, ymax, 
                 degree=1, fx=True, fx_index=(0,0)):

        self.xc = xc
        self.yc = yc
        self.n_stars = len(xc)
    
        self.hsize = hsize
        self.nd = 2*self.hsize + 1
        self.n_r = self.nd**2

        # Allow for debug cases where we don't have a list of stars
        try:
            self.data = np.array([S.data_slice for S in stars])
            self.weights = np.array([S.weight_slice for S in stars])
            self.init_weights_inv = np.copy(self.weights**-1)
        except:
            self.data = np.zeros((self.n_stars, self.nd, self.nd))
            self.weights = np.zeros_like(self.data)
        self.stars = np.array(stars)

        self.bad_pix = np.zeros_like(self.weights, dtype=bool)
        self.gain = gain

        # Degree of polynomial of R function determines shape of arrays:
        deg_dict = {0: 1, 1: 3, 2: 6}
        self.degree = deg_dict[degree]
        
        self.all_vars = self.n_stars + self.n_r * self.degree + 2*self.degree

        self.phi = phi

        self.fx = fx
        if self.fx:
            self.set_fix(fx_index)

        self.xmax = xmax
        self.ymax = ymax
        self.ax = 2./ xmax
        self.bx = 1 - self.ax * xmax
        self.ay = 2./ ymax
        self.by = 1 - self.ay * ymax
        norm_x = self.ax * self.xc + self.bx
        norm_y = self.ay * self.yc + self.by

        cder = np.array([np.ones((self.n_stars,1)), norm_x[:,None],
                         norm_y[:,None], norm_x[:,None]**2,
                         norm_y[:,None]**2, 
                         (norm_x * norm_y)[:,None]])[:self.degree]

        # Equation N:
        self.cder = cder.reshape(-1, self.n_stars)[:self.degree]

        #Equation N repeated by the number of pixels to use for R:
        self.rder = np.repeat(self.cder[:self.degree], self.n_r, axis=0).T
        self.rtile = np.tile(self.cder[:self.degree], (self.n_r, 1, 1)).T

        

    #@profile
    def reset_pos(self):
        t0 =time.time()
        norm_x = self.ax * self.xc + self.bx
        norm_y = self.ay * self.yc + self.by
        cder = np.array([np.ones((self.n_stars,1)), norm_x[:,None],
                         norm_y[:,None], norm_x[:,None]**2,
                         norm_y[:,None]**2, 
                         (norm_x * norm_y)[:,None]])[:self.degree]
        # Equation N:
        self.cder = cder.reshape(-1, self.n_stars)[:self.degree]

        # Equation N repeated by the number of pixels to use for R:
        self.rder = np.repeat(self.cder[:self.degree], self.n_r, axis=0).T
        self.rtile = np.tile(self.cder[:self.degree], (self.n_r, 1, 1)).T
        t1 = time.time()
        # Calculate the convolution matrices (C_s in the note)
        self.dconvolve()
        t2 = time.time()

        print 'reset times', t1 - t0, t2 - t1 

    def set_fix(self, ind):

        x, y = ind
        xi = np.tile(np.arange(-self.hsize, self.hsize+1), (self.nd, 1))
        yi = np.tile(np.arange(-self.hsize, self.hsize+1)[:, None], (1, self.nd))

        # This sets where to fix the residual grid:
        fx = True
        self.fx_index = np.flatnonzero((xi.ravel() == 0) & (yi.ravel() == 0))[0]
        self.fx_value = 0
        
    def set_weights(self, vars, rconv):
        # The weights need to be corrected for the flux of the star
        fluxes = vars[:self.n_stars]
        correction = fluxes[:,None] * self.psi_model(vars, rconv) / self.gain
        self.weights = (self.init_weights_inv + correction)**-1
        self.weights[self.bad_pix] = 0

    def convolve(self, r, return_fn=False):
        # Function to convolve between model grid and pixel grid

        dx = self.xc - np.round(self.xc)
        dy = self.yc - np.round(self.yc)
        rr = r.reshape(self.nd, self.nd)
        orig_x = np.arange(-self.hsize, self.hsize+1)
        new_x = np.arange(-self.hsize-1, self.hsize + 2)
        r_tmp = np.zeros((self.nd+2, self.nd+2))
        r_tmp[1:-1, 1:-1] = rr
        rr = r_tmp

        spl = RBS(new_x, new_x, rr, kx=1, ky=1)
        if return_fn:
            return spl
        r_conv = np.zeros((self.n_stars, self.nd, self.nd))

        for i in range(self.n_stars):
            r_conv[i] = spl(orig_x - dy[i], orig_x - dx[i])

        return r_conv.reshape(self.n_stars, -1)

    #@profile
    def dconvolve(self):
        # Set self.s_list variable which gives the derivative of the
        # convolution between model grid and data grid
        t0d = time.time()

        dx = (self.xc - np.round(self.xc))
        dy = (self.yc - np.round(self.yc))

        orig = np.zeros((3,3))
        orig[1,1] = 1
        t1d = time.time()
        x = np.array([-1, 0, 1])
        y = np.array([0.5,1,0.5])
        tx1 = time.time()
        spl = RBS(x, x, orig, kx=1, ky=1)
        kernels = np.array([spl(x - dy[s], x - dx[s]) for s in range(len(dx))])

        deriv_s = [[] for s in range(self.n_stars)]
        r_range = np.tile(np.arange(self.n_r)[:,None], (1, self.n_r))
        for i in range(self.nd):
            for j in range(self.nd):
                tmp = np.zeros((len(dx), self.nd+2, self.nd+2))
                tmp[:, i:i+3, j:j+3] = kernels 

                ins = tmp[:,1:-1,1:-1].reshape(self.n_stars, -1)
                nonz = ins != 0
                for s in range(self.n_stars):
                    nz = nonz[s]
                    tmp_array = np.array([ins[s,nz], r_range[nz,0], 
                                          r_range[i*self.nd+j, nz]]).T
                    deriv_s[s].extend(tmp_array)

        self.s_list = [None for s in range(self.n_stars)]

        for s in range(self.n_stars):
            tmp_array = np.array(deriv_s[s])
            self.s_list[s] = SS.coo_matrix((tmp_array[:,0],
                                            (tmp_array[:,1], tmp_array[:,2])),
                                           shape=(self.n_r, self.n_r)).todia()


    def flux_estimate(self):

        fluxes = ((self.data * self.phi * self.weights).sum(axis=1)/
                  (self.phi**2 * self.weights).sum(axis=1))
        return fluxes

    def psi_model(self, vars, rconv):
        # Combine analytic and discrete parts of the model:

        cs = vars[self.n_stars + self.degree * self.nd**2:][:self.degree]
        c_poly = (cs[:,None] * self.cder).sum(axis=0)
        psi = ((1 - c_poly.reshape(-1,1))*self.phi + rconv)

        return psi
        
    def chi2s(self, vars, rconv):
        # Calculate the chi2 of the model

        flux = vars[:self.n_stars]
        psi = self.psi_model(vars, rconv)
        lam = vars[-self.degree:]
        chi = (self.weights * (self.data - flux.reshape(-1,1)*psi)**2)
        return chi.sum(axis=1)

    def remove_outliers(self, chi2s, vars, r_poly, sigmacut=4):
        # Find and remove outlier stars, could be cleaner

        chi2_median = np.median(chi2s)
        chi2_sigma =((chi2s**2).sum()/len(chi2s) - np.mean(chi2s)**2)**0.5
        max_chi2 = chi2_median + chi2_sigma * sigmacut

        non_outliers = chi2s <= max_chi2

        new_vars = np.copy(vars)
        if (~non_outliers).sum() > 0:
            for i in np.flatnonzero(~non_outliers)[::-1]:
                print 'removing outlier star at %s, %s, with chi2 = %s' \
                    % (self.xc[i], self.yc[i], chi2s[i])
                new_vars = np.delete(new_vars, i)

            self.n_stars = non_outliers.sum()
            self.stars = self.stars[non_outliers]
            self.data = self.data[non_outliers]
            self.weights = self.weights[non_outliers]
            self.init_weights_inv = self.init_weights_inv[non_outliers]
            self.bad_pix = self.bad_pix[non_outliers]
            self.xc = self.xc[non_outliers]
            self.yc = self.yc[non_outliers]
            self.phi = self.phi[non_outliers]
            self.cder = self.cder[:,non_outliers]
            self.rder = self.rder[non_outliers]
            self.rtile = self.rtile[non_outliers]
            self.all_vars = self.n_stars + self.n_r * self.degree + 2*self.degree
            self.s_list = [self.s_list[no] for no in np.flatnonzero(non_outliers)]

        r_poly = r_poly[non_outliers]
        return new_vars, r_poly

    def remove_outlier_pixels(self, vars, rconv, sigmacut=5):
        # Find outlier pixels and set their weights to zero

        flux = vars[:self.n_stars]
        psi = self.psi_model(vars, rconv)
        model = flux[:,None] * psi
        resids = self.data - model
        
        removed_pixels = 0
        self.bad_pix = np.zeros_like(self.weights, dtype=bool)
        for i in range(self.n_stars):
            bad_ind = (self.weights[i] * resids[i]**2) > sigmacut**2
            self.bad_pix[i, bad_ind] = 1
            self.weights[i, bad_ind] = 0
            removed_pixels += bad_ind.sum()

        return removed_pixels

    def lag_parts(self, vars, rconv):
        # Helper function for calculating the lagrangian:
        flux = vars[:self.n_stars]
        r = vars[self.n_stars:][:self.degree 
                                * self.nd**2].reshape(self.degree, self.n_r)
        cs = vars[self.n_stars + self.degree * self.nd**2:][:self.degree]

        psi = self.psi_model(vars, rconv)
        lam = vars[-self.degree:]
        chi = (self.weights * (self.data - flux[:,None]*psi)**2).sum()

        g = (lam*(r.sum(axis=(1)) - cs)).sum()

        return chi, g

    def lagrangian(self, vars, rconv, full_output=0):
        # The lagrangian, Equation A from the note
        chi, g = self.lag_parts(vars, rconv)
        if full_output:
            return chi - g, chi, g
        else:
            return chi - g

    def dlagdf_h(self,vars, rconv):

        psi = self.psi_model(vars, rconv)
        dldf = -psi

        return dldf


    def dchidr_h(self, vars, rconv):
        # Equation B
        flux = vars[:self.n_stars].reshape(-1,1)
        dldr_1 = - flux * self.rder

        return dldr_1

    def dgdr(self, vars):

        lam = vars[-self.degree:]        
        return np.repeat(-lam, self.n_r)

    def dchidc_h(self, vars):

        flux = vars[:self.n_stars]
        dldc = flux[None, :,None] * self.cder[:,:,None] * self.phi

        return dldc

    def dgdc(self, vars):

        lam = vars[-self.degree:]
        return lam

    def dgdl(self, vars):
        # Equation D
        r = vars[self.n_stars:][:self.degree *
                                self.nd**2].reshape(self.degree, self.n_r)
        cs = vars[-2*self.degree:-self.degree]
        return -r.sum(axis=(1)) + cs

    #@profile
    def computeAB(self, vars, rconv):
        # A is the Hessian of the Lagrangian, B is the derivative
        # Compute the equations 14 - 28 in the note:

        B = np.zeros(self.all_vars)
        A = np.zeros((self.all_vars, self.all_vars))

        flux = vars[:self.n_stars]
        psi = self.psi_model(vars, rconv)
        res = self.data - flux[:, None] * psi

        # Set up some variables that will be used multiple times below:
        h_df = self.dlagdf_h(vars, rconv)
        h_dr = self.dchidr_h(vars, rconv)
        h_dc = self.dchidc_h(vars)
        h_dl = np.zeros(self.degree)
        
        dg_dr = self.dgdr(vars)
        dg_dc = self.dgdc(vars)
        dg_dl = self.dgdl(vars)

        B_helper = self.weights * res
        h_dr_helper = np.array([((self.s_list[s].transpose().dot(B_helper[s].T)).T)
                                 for s in range(self.n_stars)])
        h_df_conv = np.array([(self.s_list[s].transpose().dot(h_df[s].T)).T
                                for s in range(self.n_stars)])
        w_conv = np.array([(self.s_list[s].transpose().dot(self.weights[s].T)).T
                             for s in range(self.n_stars)])  
        
        w_h_dc = self.weights * h_dc
        w_h_dc_conv = np.array([[(self.s_list[s].T.dot(w_h_dc[i, s])).T for
                                 s in range(self.n_stars)]
                                for i in range(self.degree)])

        t_help = [self.s_list[s].multiply(2 * self.weights[s][None,:]) for s 
                  in range(self.n_stars)]
        alt_t_matrix = np.array([t_help[s].dot(self.s_list[s].transpose())
                             for s in range(self.n_stars)])


        # Equation 15, df:
        B[:self.n_stars] = 2 * (B_helper * h_df).sum(axis=1)
        # Equation 19, dfdf:
        A[:self.n_stars, :self.n_stars] = np.diag(2 * (self.weights * h_df**2).sum(axis=1))
        t4 = time.time()
        for d in range(self.degree):
            t41 = time.time()
            sl = slice(d * self.n_r, (d+1) * self.n_r)
            # Equation 16, dr:
            B[self.n_stars:][sl] = 2 * (h_dr_helper * h_dr[:,sl]).sum(axis=0)
            B[self.n_stars:][sl] += dg_dr[sl]
            # Equation 20, dfdr:
            x = 2 * w_conv * h_df_conv * h_dr[:,sl]
            A[:self.n_stars,self.n_stars:][:,sl] = x
            A[self.n_stars:,:self.n_stars][sl,:] = x.T
            t42 = time.time()
            # Equation 23, dr(1)dr(2) (get the cross terms among the R polynomial):
            for dd in range(d+1):
                sl2 = slice(dd * self.n_r, (dd+1) * self.n_r)
                f_help = flux**2 * self.cder[d] * self.cder[dd]
                x_alt = np.zeros((self.n_r, self.n_r))
                for s in range(self.n_stars):
                    x_alt += alt_t_matrix[s].multiply(f_help[s])
                A[self.n_stars:, self.n_stars:][sl, sl2] = x_alt
                A[self.n_stars:, self.n_stars:][sl2, sl] = x_alt
            # Equation 24, drdc
            for poly_ind in range(self.degree):
                cc_ind = self.n_stars + self.degree * self.n_r + poly_ind
                dCdc = self.cder[poly_ind] 
                x = 2 * (w_h_dc_conv[poly_ind] * h_dr[:,sl]).sum(axis=0)

                A[self.n_stars:, cc_ind][sl] = x
                A[cc_ind, self.n_stars:][sl] = x
                
                ll_ind = self.n_stars + self.degree * self.n_r + self.degree + poly_ind
                # Equation 25, drdlambda
                if poly_ind == d:
                    A[self.n_stars:,ll_ind][sl] = -1
                    A[ll_ind, self.n_stars:][sl] = -1


        c_ind = self.n_stars + self.degree * self.n_r
        c_sl = slice(c_ind, c_ind + self.degree)
        # Equation 21, dfdc
        x = 2 * ((self.weights* h_df)[None,:,:] * h_dc).sum(axis=2)
        A[c_sl, :self.n_stars] = x
        A[:self.n_stars, c_sl] = x.T

        # Equation 26, dcdc
        x = self.weights[None, None,:, :] * h_dc[:,None, :, :] * h_dc[None, :,:, :]
        A[c_sl, c_sl] = 2 * x.sum(axis=(2,3))
        # Equation 27, dcdlambda
        l_sl = slice(c_ind + self.degree, c_ind + 2 * self.degree)
        A[c_sl, l_sl] = np.diag(np.ones(self.degree))
        A[l_sl, c_sl] = np.diag(np.ones(self.degree))

        # Equation 17, dc
        B[-2*self.degree:
          -self.degree] = 2 * (B_helper[None,:,:] * h_dc).sum(axis=(1,2))
        B[-2*self.degree:-self.degree] += dg_dc
        # Equation 18, dlambda
        B[-self.degree:] = dg_dl
         
        # If a pixel of the R matrix is fixed, i.e. the central pixel, which
        # controls for degeneracies with the analytic portion of the PSF,
        # remove it from the matrix of fit parameters. This seems simpler than
        # redoing all of the indices.
        fx_indices = np.array([0, self.n_r, 2*self.n_r]) \
                     + self.fx_index + self.n_stars
        if self.fx:
            B = np.delete(B, fx_indices)
            A = np.delete(A, fx_indices, axis=0)
            A = np.delete(A, fx_indices, axis=1)
        A_prime = A[:-2 * self.degree, :-2 * self.degree]

        A_nonzero = A.nonzero()
        A_pr_nonzero = A_prime.nonzero()

        newnumvars = self.all_vars - self.degree * self.fx

        # Now convert the A matrix to a scipy sparse matrix:
        A_sparse = SS.coo_matrix((A[A_nonzero], (A_nonzero[0], A_nonzero[1])),
                                 shape=(newnumvars, newnumvars))
        A_pr_sparse = SS.coo_matrix((A_prime[A_pr_nonzero], 
                                     (A_pr_nonzero[0], A_pr_nonzero[1])),
                                    shape=(newnumvars - 2*self.degree,
                                           newnumvars - 2*self.degree))

        a_matrix = A[-2*self.degree:, :-2*self.degree]
        k_matrix = A[-2*self.degree:, -2*self.degree:]
        # Return a couple of options for either a block solution, or cholesky with the 
        # simplicial method:
        return B, A_sparse, A_pr_sparse, a_matrix, k_matrix

    def get_psfmoments(self, wxx, wyy, wxy, xi, yi):
        
        x_center = int(self.xmax / 2.)
        y_center = int(self.ymax / 2.)

        # c and r are their zero-order values at the center
        center_c = self.c[0]
        center_r = self.r[0]


        def center_psf(x, y):
            fact = (1 + (x**2) * wxx + (y**2) * wyy + 2 * x * y * wxy)
            det = wxx * wyy - wxy**2
            norm = np.sqrt(np.fabs(det)) * (2.5 - 1) / np.pi
            return norm * np.power(fact, -2.5)
            
        center_phi = integ_in_pixels(xi, yi, center_psf).ravel()
        center_psf = (1 - center_c)*center_phi + center_r
        
        xi = xi.ravel()
        yi = yi.ravel()
        momentx1 = (xi * center_psf).sum()
        momenty1 = (yi * center_psf).sum()
        momentx2 = (xi**2 * center_psf).sum()
        momenty2 = (yi**2 * center_psf).sum()
        momentxy = (xi * yi * center_psf).sum()
        
        psf_sum_sq = center_psf.sum()**2
        psf_sq_sum = (center_psf**2).sum()
        noise_equiv_area = psf_sum_sq / psf_sq_sum

        return momentx1, momenty1, momentx2, momenty2, momentxy, noise_equiv_area

    #@profile
    def residuals_fit(self, init_fluxes=None, max_iter=30, lag_tol=1e-6):
        
        if init_fluxes==None:
            fluxes = self.flux_estimate()
        else:
            fluxes = init_fluxes

        model_orig = fluxes[:,None] * self.phi        
        resid_orig = (self.data - model_orig)
        
            
        # Initial parameters:
        # r function 
        r_orig = np.zeros((self.degree, self.n_r))
        #r_orig[0] = np.average(resid_orig/fluxes[:,None], axis=0, 
        #                       weights=self.weights*fluxes[:,None]**2)
        rconv = np.array([self.convolve(rr) for rr in r_orig])
        rconv = np.transpose(rconv, (1, 0, 2))
        # Equation 31:
        r_poly = (rconv * self.rtile).sum(axis=1)
        r_poly_orig = np.copy(r_poly)

        c_orig = np.zeros(self.degree)
        #c_orig = np.sum(r_orig, axis=1)
        #Equation 30:
        c_poly_orig = (c_orig[:,None] * self.cder[:self.degree]).sum(axis=0)

        lam_orig = np.ones(self.degree)
        fit_vars = np.hstack([fluxes, r_orig.ravel(), c_orig, lam_orig])

        self.set_weights(fit_vars, r_poly)
        refit_fluxes = self.flux_estimate()
        fit_vars[:self.n_stars] = refit_fluxes
        self.set_weights(fit_vars, r_poly)

        # Calculate the convolution matrices (C_s in the note)
        self.dconvolve()

        self.xi = np.tile(np.arange(-hsize, hsize+1), (self.nd, 1)).ravel()
        self.yi = np.tile(np.arange(-hsize, hsize+1)[:, None], 
                          (1, self.nd)).ravel()
        dxi = abs(self.xi[:, None] - self.xi[None,:])
        dyi = abs(self.yi[:, None] - self.yi[None,:])
        self.nonzero = ((dxi < 2) & (dyi < 2)).nonzero()
        self.nonzero1 = ((dxi < 2) & (dyi < 2))

        lagrangian_fid, chi, p = self.lagrangian(fit_vars, r_poly,
                                                 full_output=1)
        print 'Initial Lagrangian = %s, (chi=%s, penalty=%s)' \
            % (lagrangian_fid, chi, p)
        
        variables = []
        derivatives = []
        max_iter = 5
        # One loop with all pixels, one loop after removing outlier pixels:
        for loop in range(2):
            # Iterate solving Ax=b for x, where A=Hess(chi2), b = d(chi2)
            # to step towards minimum of chi2:
            for iter in range(max_iter):

                t00 = time.time()

                variables.append(fit_vars)

                chi2s = self.chi2s(fit_vars, r_poly)

                # Fill in matrix A and vector b for equation Ax=b
                fx_indices = np.array([0, self.n_r, 2*self.n_r]) \
                             + self.fx_index + self.n_stars
                tdl = time.time()
                dlagrange, matrix1, matrix2, a_mat, k_mat = self.computeAB(fit_vars, r_poly)
                print "Compute AB time", time.time() - tdl

                t1 = time.time()

                # We try a couple methods to solve the cholesky quickly, the first two use cholmod
                # and are very fast, so I do not bother removing first, which often fails.
                try:
                    lag_theta = dlagrange[:self.n_stars + (self.n_r-self.fx)*self.degree]
                    lag_lmbda = dlagrange[self.n_stars + (self.n_r-self.fx)*self.degree:]
                    f = cholesky(matrix2)
                    HA = f(a_mat.T)
                    KAHA = k_mat - a_mat.dot(HA)
                    S = SL.inv(KAHA)
                    Hlagth = f(lag_theta)
                    n1 = lag_lmbda - a_mat.dot(Hlagth)
                    lmbda = S.dot(n1)
                    n2 = lag_theta - a_mat.T.dot(lmbda)
                    theta = f(n2)
                    choldecomp = np.hstack([theta, lmbda])

                except sksparse.cholmod.CholmodNotPositiveDefiniteError:
                    print 'Normal cholesky failed'
                    try:
                        f = cholesky(matrix1, mode='simplicial')
                        choldecomp = f(dlagrange)
                        print '  Used simplicial method'
                    except:
                        print 'Simplicial failed'
                        choldecomp = SSLA.spsolve(matrix1, dlagrange)
                        print '  Used spsolve'
                print "Matrix solve: ", time.time() - t1

                # Insert fixed value back into the matrix:
                if self.fx:
                    for i in range(self.degree):
                        ind = self.n_stars + self.n_r * i + self.fx_index
                        choldecomp = np.insert(choldecomp, ind, self.fx_value)

                # Do line search along direction of cholesky solution:

                # First, set up convolution of dR to speed up line search:
                r_chol = choldecomp[self.n_stars:self.n_stars 
                                    + self.n_r * self.degree].reshape(self.degree, self.n_r)

                rchconv = np.array([self.convolve(rr) for rr in r_chol])
                rchconv = np.transpose(rchconv, (1, 0, 2))
                r_chol_poly = (rchconv * self.rtile).sum(axis=1)

                t2 = time.time()
                def lnsrc(x):
                    new_vars = np.copy(fit_vars)
                    new_vars -= x * choldecomp
                    r_new_poly = r_poly - x * r_chol_poly
                    self.set_weights(new_vars, r_new_poly)
                    l = self.lagrangian(new_vars, r_new_poly)
                    return l

                fopt = SO.fmin(lnsrc, 0, disp=0, full_output=1)
                step = fopt[0]
                print 'Step:', step

                print "Line search time", time.time() - t2

                # Actually take step towards minimization:
                new_vars = np.copy(fit_vars) 
                new_vars -= step * choldecomp

                # Get all the individual variables out of the new_vars vector:    
                fs = new_vars[:self.n_stars]
                r = new_vars[self.n_stars:self.n_stars 
                             + self.n_r * self.degree].reshape(self.degree,
                                                               self.n_r)

                r_fns = [self.convolve(rr, return_fn=True) for rr in r]
                c = new_vars[-self.degree*2:-self.degree]

                # Refit the flux and position of each star. This is necessary
                # to ensure that the fitting method is consistent when we use the
                # PSF to fit science targets.
                new_fluxes = np.zeros(self.n_stars)
                t_star = time.time()
                old_x = np.copy(self.xc)
                old_y = np.copy(self.yc)
                for s, S in enumerate(self.stars):
                    x_copy = np.copy(S.x)
                    y_copy = np.copy(S.y)
                    try:
                        S.fit_pos(fs[s], r_fns, c)
                    except:
                        print "Bad fit at %s, %s" % (x_copy, y_copy)
                        new_fluxes[s] = new_vars[s]
                        continue
                    self.xc[s] = S.x
                    self.yc[s] = S.y
                    new_fluxes[s] = S.flux
                    self.phi[s] = S.phi
                    self.data[s] = S.data_slice
                    self.init_weights_inv[s] = S.weight_slice**-1
                new_vars[:self.n_stars] = new_fluxes
                deltx = self.xc - old_x
                delty = self.yc - old_y
                print 'Position changes: <dx>=%.5f, std_dx= %.5f, <dy>=%.5f, std_dy= %.5f, ' \
                    % (deltx.mean(), deltx.std(), delty.mean(), delty.std())
                print 'Star refit time', time.time() - t_star
                self.reset_pos()
                
                # Reset parameters again now that positions have been refit:
                fit_vars = new_vars 
                rconv = np.array([self.convolve(rr) for rr in r])
                rconv = np.transpose(rconv, (1, 0, 2))
                r_poly = (rconv * self.rtile).sum(axis=1) # Equation 31

                c = fit_vars[-self.degree*2:-self.degree] # Equation 30
                c_poly = (c[:,None] * self.cder[:self.degree]).sum(axis=0)
                lam = fit_vars[-self.degree:]
                self.set_weights(fit_vars, r_poly)

                new_lagr, ch, p = self.lagrangian(fit_vars, r_poly,
                                                  full_output=1)
            
                print 'Iteration %s: Lagrangian = %s (chi=%s, penalty=%s)' \
                    % (iter, new_lagr, ch, p)
                dof = self.n_stars * self.n_r
                dof -= 3 * 3 # psf params
                dof -= (self.degree * self.n_r - self.degree*self.fx)
                print ' ---> chi/dof = %s / %s = %s' % (ch, dof, ch/dof)
            
                # Exit loop if line search returns 0 or change in 
                # lagrangian is below lag_tol:
                if (step == 0) or (abs(new_lagr - lagrangian_fid) < lag_tol):
                    print "Convergence reached"
                    break

                lagrangian_fid = new_lagr
                chi2s = self.chi2s(fit_vars, r_poly)

                fit_vars, r_poly = self.remove_outliers(chi2s, fit_vars, r_poly)

                print 'Iteration time:', time.time() - t00
            # If we are on the first outer loop, get rid of outlier pixels:
            if loop == 0:
                bad_pix = self.remove_outlier_pixels(fit_vars, r_poly)
                if bad_pix == 0:
                    print 'No outlier pixels, stopping here'
                    break
                else:
                    print 'Removing %s pixels and restarting fit' % bad_pix
                    max_iter = 3

        # Set the final parameters so they can be output:
        self.fluxes = fit_vars[:self.n_stars]
        self.r = r
        self.c = c
        self.r_poly = r_poly
    
        self.final_chi2s = chi2s

        final_psi = self.psi_model(fit_vars, r_poly)
        print final_psi.sum(axis=1)
        self.final_model = self.fluxes[:,None] * final_psi

        # Some final diagnostics. The R sums and C values should be about equal if 
        # the minimization has succeeded. If sum_R = C, the values of lambda will be
        # degenerate, but they usually converge to a small value.
        print 'R sums:', r.sum(axis=1)
        print 'C values:', c, 'orig:', c_orig
        print 'Lambda values:', lam, 'orig', lam_orig
        print 'Final lagrangian parts:', self.lag_parts(fit_vars, r_poly)

    def stars_fit(self, output_file):
         
        with open(output_file, 'r') as f:
            old_output = pickle.load(f)

        self.xc = old_output['xc']
        self.yc = old_output['yc']

        match_stars = [S for S in self.stars if ((((S.x - self.xc)**2 < 3) & ((S.y - self.yc)**2 < 3)).any()
                       and (hasattr(S, 'covariance') and hasattr(S, 'paramweight')))]
        try:     
            assert len(self.xc) == len(match_stars)
        except:
            print len(self.xc), len(match_stars), "something failed, skip that star"
        self.stars = match_stars

	self.r = old_output['r']
        self.c = old_output['c']
        r_fns = [self.convolve(rr, return_fn=True) for rr in self.r]

        new_fluxes = np.zeros(len(self.stars))
        for s, S in enumerate(self.stars):
            x_copy = np.copy(S.x)
            y_copy = np.copy(S.y)
            try:
                S.fit_pos(old_output['fluxes'][s], r_fns, self.c)
            except:
                print "Bad fit at %s, %s" % (x_copy, y_copy)
                continue
            self.xc[s] = S.x
            self.yc[s] = S.y
            new_fluxes[s] = S.flux
            self.phi[s] = S.phi
            self.data[s] = S.data_slice
            self.init_weights_inv[s] = S.weight_slice**-1
             
       

def poloka_emulate_output(M, R, seeing, path):
    """Write a file that emulates the psf.dat output of poloka
    """

    file = op.join(path, "psf.dat")
    f = open(file, 'w')
    f.write("ImagePSF_version 1\n")
    f.write("%s\n" % (M.name))
    f.write("%s %s\n" % (R.hsize, R.hsize))
    f.write("%s %s\n" % (seeing, R.gain))
    f.write("%s %s\n" % (M.xmax, M.ymax))
    f.write("%s\n" % M.degree)
    reorder = np.array([0, 2, 1])
    for w in [M.wxx_vector, M.wyy_vector, M.wxy_vector]:
        write_vector = w[reorder]
        f.write("Poly2_version 1\n")
        f.write("%s %s %s %s\n" % (M.ax, M.bx, M.ay, M.by))
        f.write("1\n")
        f.write("3\n")
        for wv in write_vector:
            f.write("%s\n" % wv)
    f.write("pixellized_residuals\n")
    f.write("Poly2Image_version 1\n")
    f.write("%s %s %s %s\n" % (M.ax, M.bx, M.ay, M.by))
    f.write("%s %s %s\n" % (1, R.hsize, R.hsize))
    f.write("3\n")
    f.write("%s %s\n" % (R.hsize, R.hsize))
    f.write(" ".join(R.r[0].astype(str)) + "\n")
    f.write("%s %s\n" % (R.hsize, R.hsize))
    f.write(" ".join(R.r[2].astype(str)) + "\n")
    f.write("%s %s\n" % (R.hsize, R.hsize))
    f.write(" ".join(R.r[1].astype(str)))
    f.close()
        

def psfstars_list(Stars, psfmoments, path):
    """Write a file that emulates the psfstars.list output file of poloka-psf
    """

    file = op.join(path, 'psfstars.list')

    header = "@MOMENTX1 %s\n@MOMENTY1 %s\n@MOMENTX2 %s\n@MOMENTY2 %s\n" \
             "@MOMENTXY %s\n@NOISE_EQUIVALENT_AREA %s\n" % psfmoments
    header += ("# x : x position (pixels)\n" \
              "# y : y position (pixels)\n" \
              "# sx : x position r.m.s\n" \
              "# sy : y position r.m.s\n" \
              "# rhoxy : xy correlation\n" \
              "# flux : flux in image ADUs\n" \
              "# eflux : Flux uncertainty\n" \
              "# fluxmax : Peak pixel value above background\n" \
              "# psfx :\n" \
              "# psfy :\n" \
              "# psfchi2 :\n" \
              "# npar : number of psf params\n" \
              "# p0 :\n" \
              "# p1 :\n" \
              "# p2 :\n" \
              "# wp00 :\n" \
              "# wp10 :\n" \
              "# wp11 :\n" \
              "# wp20 :\n" \
              "# wp21 :\n" \
              "# wp22 :\n" \
              "#covff :\n" \
              "#covxf :\n" \
              "#covxx :\n" \
              "#covyf :\n" \
              "#covyx :\n" \
              "#covyy :\n" \
              "# format  BaseStar 3  PSFStar 1\n" \
              "# end" )
    output = np.zeros((len(Stars), 27))
    for ss, s in enumerate(Stars):
        # Star cov variables are in order "flux, x, y"
        output[ss, 0] = s.x
        output[ss, 1] = s.y
        sx = s.covariance[1,1]**0.5
        sy = s.covariance[2,2]**0.5 
        rhoxy = s.covariance[1,2] / (sx * sy)
        output[ss, 2] = sx
        output[ss, 3] = sy
        output[ss, 4] = rhoxy
        output[ss, 5] = s.flux
        output[ss, 6] = s.covariance[0, 0]**0.5
        output[ss, 7] = s.fluxmax # This is just transfered form orig starfile
        output[ss, 8] = s.x # Again, for fun
        output[ss, 9] = s.y 
        output[ss, 10] = s.chi2
        output[ss, 11] = 3
        output[ss, 12] = s.wxx
        output[ss, 13] = s.wyy
        output[ss, 14] = s.wxy
        t = 0
        for i in range(3):
            for j in range(i+1):
                output[ss, 15+t] = s.paramweight[i, j]
                t += 1
        t = 0
        for i in range(3):
            for j in range(i+1):
                output[ss, 21+t] = s.covariance[i, j]
                t += 1
    np.savetxt(file, output, header=header, comments='')

def output_psf(M, R, path):
    
    dict = {}
    dict['fluxes'] = R.fluxes
    dict['dfluxes'] = R.dfluxes
    dict['xc'] = R.xc
    dict['yc'] = R.yc
    dict['psfchi2'] = R.final_chi2s
    dict['wxx'] = M.wxx_vector
    dict['wyy'] = M.wyy_vector
    dict['wxy'] = M.wxy_vector
    dict['w_cov'] = M.covariance
    dict['r'] = R.r
    dict['c'] = R.c
    dict['gain'] = R.gain

    # Temp test stuff:
    dict['r_poly'] = R.r_poly
    dict['cder'] = R.cder
    dict['phi'] = R.phi

    dict['models'] = R.final_model
    dict['data'] = R.data
    dict['init_weights_inv'] = R.init_weights_inv
    dict['weights'] = R.weights

    with open(op.join(path, 'test_out.pkl'), 'w') as f:
        pickle.dump(dict, f)

if __name__ == '__main__':
    print 'Starting psf.py'
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('prefix', type=str,
                        help='Path to ccd file with dbimfast output')
    parser.add_argument('-o', '--output_path', type=str, default='testout',
                        help='Path to directory for output')
    args = parser.parse_args()
    
    # Get seeing and initial positions and fluxes of isolated stars:
    seeing, xc, yc, init_f, fluxmax = get_sxy(args.prefix)
    # Set vignette size:
    hsize = int(5 * seeing) + 1
    
    print 'Initial Seeing = %s' % seeing
    print 'Hsize=%s, Vignette Side=%s' % (hsize, 2*hsize+1)
    print 'Starting fit with %s stars' % len(xc)
    data, weights, corners, gain, full_data, full_weights = get_data(args.prefix, xc, yc, hsize)

    # Fit the analytic portion of the PSF model:
    M = MoffatPSF(hsize, xc, yc, full_data, full_weights, gain, fluxmax, degree=1)
    M.moffat_fit(init_seeing=seeing, init_fluxes=init_f)
    if not M.check_corners(corners):
        raise ValueError('Could not model a reasonable parameters spatial'
                         ' variation of the analytic PSF. stopping here.')

    phi = M.phi.reshape(M.n_stars, -1)
    print phi.sum(axis=1)

    # Given the analytic portion of the PSF, fit any residual shape in the PSF:
    R = PSFResiduals(hsize, M.xc, M.yc, phi, M.stars, gain, M.xmax, M.ymax,
                     degree=1)
    # DEBUG option to load old output and recalculate results:
    if op.isfile(op.join(args.output_path, 'test_out.pkl')):
        R.stars_fit(op.join(args.output_path, 'test_out.pkl'))

    else:
        R.residuals_fit()
        poloka_emulate_output(M, R, seeing, args.output_path)
        output_psf(M, R, args.output_path)

    print R.phi.sum(axis=1)
    # Get moments of the final PSF:
    psfmoments = R.get_psfmoments(M.wxx_vector[0], M.wyy_vector[0],
                                  M.wxy_vector[0], M.xi, M.yi)

    # Make output files in format needed for the scene-modeling code:
    psfstars_list(R.stars, psfmoments, args.output_path)

    print 'Total Time:', time.time() - start_time
