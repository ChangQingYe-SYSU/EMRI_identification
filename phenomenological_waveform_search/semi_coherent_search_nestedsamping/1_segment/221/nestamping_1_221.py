import sys
import os
import numpy as np
import emcee
import scipy.interpolate as spi
# 去做探测
#%%
from copy import copy
import sys
import os
from typing_extensions import ParamSpec
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np

import pyfftw
from scipy.fftpack import fft,ifft
import random
import math
import h5py
import sys
import time
import emcee
from schwimmbad import MPIPool
import dynesty
import dynesty.pool as dypool

'''Michelson 响应所用到的公式'''




def get_F(t,qS,phiS,_psi):
    '''https://arxiv.org/pdf/1901.02159.pdf'''

    q_j0806 = 1.65
    phi_j0806 = 2.10
    f_sc = 1/(3.65 * 24 *3600) 
    k = 2*np.pi* f_sc * t

    cos2k = np.cos(2*k)
    sin2k = np.sin(2*k)
    cos2qS = np.cos(2*qS)
    sin2qS = np.sin(2*qS)
    cosqS = np.cos(qS)
    sinqS = np.sin(qS)
    
    cosqs_j0806 = np.cos(q_j0806)
    sinqs_j0806 = np.sin(q_j0806)
    cos2qs_j0806 = np.cos(2*q_j0806)
    sin2qs_j0806 = np.sin(2*q_j0806)
    
    sin2phiS_2phi_j0806 = np.sin(2*phiS - 2*phi_j0806)
    cos2phiS_2phi_j0806 = np.cos(2*phiS - 2*phi_j0806)

    sinphiS_phi_j0806 = np.sin(phiS - phi_j0806)
    cosphiS_phi_j0806 = np.cos(phiS - phi_j0806)
    

    D_pluss = (np.sqrt(3)/ 32)*(4*cos2k*((3+cos2qS)*cosqs_j0806*sin2phiS_2phi_j0806 + \
        2*sinphiS_phi_j0806*sin2qS*sinqs_j0806) \
            - sin2k*(3 + cos2phiS_2phi_j0806*(9+cos2qS*(3+cos2qs_j0806)) -6*cos2qs_j0806*sinphiS_phi_j0806**2 \
                - 6 * cos2qS*sinqs_j0806**2 + 4 * cosphiS_phi_j0806*sin2qS*sin2qs_j0806))


    D_cross = (np.sqrt(3)/8)*(-4*cos2k*(cos2phiS_2phi_j0806*cosqS*cosqs_j0806 + cosphiS_phi_j0806*sinqS*sinqs_j0806) \
        + sin2k *(-cosqS*(3+sin2qs_j0806)*sin2phiS_2phi_j0806 -2 *sinphiS_phi_j0806*sinqS*sin2qs_j0806))
    
    
    psi_ldc = _psi
    _F_pluss = D_pluss * np.cos(psi_ldc) - D_cross * np.sin(psi_ldc)
    _F_cross = D_pluss * np.sin(psi_ldc) + D_cross * np.cos(psi_ldc)


    return  _F_pluss,_F_cross

def get_doppler(t,omega,qS,phiS):
    
    AUsec = 499.004783836156412
    orbphs=2.*np.pi*t/tyr
    
    cosorbphs=np.cos(orbphs-phiS)

    sinqS=np.sin(qS)
    
    Doppler_phi = omega * AUsec * sinqS * cosorbphs
    
    return Doppler_phi



'''数据处理常用公式'''




'''TQ_PSD'''
def michelson_TQ(f):
    c = 2.99792458e8
    L_TQ = 3 ** 0.5 * 1e8
    f_star_TQ = c / (2 * np.pi * L_TQ)
    Sa_TQ = 1e-15 ** 2
    Sx_TQ = 1e-12 ** 2  
    return 1/L_TQ**2 * (2*(1+np.cos(f/f_star_TQ)**2)*Sa_TQ / (2*np.pi*f)**4 * (1+1e-4/f)+Sx_TQ)

#produce noise 

def frequency_noise_from_psd(psd, delta_f, seed=None):
    """ Create noise with a given psd.
    Return noise coloured with the given psd. The returned noise
    FrequencySeries has the same length and frequency step as the given psd.
    Note that if unique noise is desired a unique seed should be provided.
    Parameters
    ----------
    psd : FrequencySeries
        The noise weighting to color the noise.
    seed : {0, int} or None
        The seed to generate the noise. If None specified,
        the seed will not be reset.
    Returns
    --------
    noise : FrequencySeriesSeries
        A FrequencySeries containing gaussian noise colored by the given psd.
    """
    sigma = 0.5 * (psd / delta_f) ** (0.5)
    if seed is not None:
        np.random.seed(seed)

    not_zero = (sigma != 0)

    sigma_red = sigma[not_zero]
    noise_re = np.random.normal(0, sigma_red)
    noise_co = np.random.normal(0, sigma_red)
    noise_red = noise_re + 1j * noise_co

    noise = np.zeros(len(sigma),dtype=complex)
    noise[not_zero] = noise_red

    return noise



def TransformData(x,wis=None):
    #pyfftw.import_wisdom(ws)
    N = len(x)
    yt = pyfftw.empty_aligned(N, dtype='float64')
    yf = pyfftw.empty_aligned(int(N/2+1), dtype='complex128')
    fft_object = pyfftw.FFTW(yt, yf, flags=('FFTW_ESTIMATE',))
    #fft_object = pyfftw.FFTW(yt, yf)
    yt = np.copy(x)
    yf = np.copy(fft_object(yt*dt))
    # wis = pyfftw.export_wisdom()
    # print ("wis = ", wis)
    return (yf[1:])




def Computeinner(d, x, S, fr,df,fmin=None, fmax=None):
    imax = len(fr)-1
    if (fmax != None):
        imax = np.argwhere(fr >fmax)[0][0] +1
    imin = 1
    if (fmin != None):
        imin = np.argwhere(fr >fmin)[0][0] +1

    SNR2 = (4.0*df) * np.sum(np.real(d[imin:imax]*np.conjugate(x[imin:imax])/S[imin:imax]))

    return SNR2


def log_likelihood_form_data_and_template(_f_signal,_f_temple,Sa):

    inner_h_h = Computeinner(_f_temple,_f_temple, Sa, freq, df,fmin=1e-4)
    inner_D_h = Computeinner(_f_signal,_f_temple, Sa, freq, df,fmin=1e-4)
    # inner_D_D = Computeinner(_f_signal,_f_signal, Sa, freq, df,fmin=1e-4)
    
    log_likelihood_values = inner_D_h - 0.5 * inner_h_h
    
    return log_likelihood_values



########mcmc fuction

def log_likelihood(_fit_pa):
    '''从mcmc维度中获取到拟合参数和方位角度信息'''
    san_pa = _fit_pa[:4]
    theta,phi = _fit_pa[4:6]
    # theta,phi = 0,0
    _psi = _fit_pa[6]
    A_index = _fit_pa[7]

    t_0 = Time_observation[-1] * time_seg_i
    
    time_response = t_0 + Time_observation

    omega_t = _fit_pa[2] + _fit_pa[1] * Time_observation + _fit_pa[0]* Time_observation**2
    
    # Doppler = get_doppler(time_response,omega_t,theta,phi)
    
    F_pluss,F_cross = get_F(time_response,theta,phi,0)

    
    '''定义波形'''
    #定义振幅
    fit_A = (1 + 1j) * 10**(A_index)

    fuction_evolution = np.poly1d(san_pa)
    
    phi_fit = fuction_evolution(Time_observation)
    
    # l_x = []

    # for _psi in [0,np.pi/4]:    
    
    F_pluss,F_cross = get_F(time_response,theta,phi,_psi)

    h_temple = fit_A * np.exp(1j*phi_fit)

    h_response = h_temple.real * F_pluss + h_temple.imag *F_cross

    h_f = TransformData(h_response)

    snr_all = log_likelihood_form_data_and_template(signal_f,h_f,Sa)

    
    return snr_all


def log_prior(pop):
    for i in range(8):
        if pop[i] < lb[i] or pop[i] > ub[i]:
            return -np.inf
        
    return 0.0
    

def log_probability(pop):
    lp = log_prior(pop)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(pop)



def prior_transform(u):
    v = 1 * u
    for i in range(len(lb)):
        v[i] = (ub[i] - lb[i])  * u[i] + lb[i]
    
    return v





if __name__=="__main__":

    t1 = time.time()
    tyr = 24*3600*365
    dt = 15

    
    ##############################
    with h5py.File('../../../signal_1.hdf5','r') as f:
        Michaelson_freq_signal =  f['Michaelson_freq_domin'][:]
        # Michaelson_freq_noise = f['Michaelson_niose_freq'][:]
        hamonic_t = f['Michaelson_time_domin'][:]
        noise_f = f['Michaelson_niose_freq_0_point_1_yr'][:]
        f.close()


    Time_observation = np.arange(0,0.1*tyr,dt)
    
    df = 1 / (len(Time_observation) * dt )
    freq = df * np.arange(int(len(Time_observation)/2 + 1))

    Sa = michelson_TQ(freq[1:])
    
    # 频率随时间的演化的量，即f2,f3 仍然选择，在第一段的时间范围内，而f1定义在增加0.001 hz1.68759367e-02
    

    lb =[2e-18,1e-10,0.0204,0,0,0,0,-21.5]
    ub = [8e-18,2e-10,0.0215,2*np.pi,np.pi,2*np.pi,np.pi/2,-21]


    time_seg_i = 1

    N_0_1_tyr = int(0.1*tyr/dt)
    
    
    signal_f_not_noise = TransformData(hamonic_t[time_seg_i*N_0_1_tyr:(time_seg_i+1)*N_0_1_tyr,1])


    signal_f = signal_f_not_noise + noise_f
    
    
    ndim = len(lb)

    path = '/data1/yecq/para_estimation_Michaelson/source_1/fit_phi/2023_6_19_for_paper/segment_1/221/'
    
    with dypool.Pool(50, log_likelihood, prior_transform) as pool:
        # The important thing that we provide the loglikelihood/prior transform from 
        # the pool    
        psampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform, ndim, 
                                    nlive=10000, sample='rslice',pool=pool,
                                    )
        psampler.run_nested(dlogz=0.1,checkpoint_file=path + 'nested_run.sav')
        
    pres = psampler.results



