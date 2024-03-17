import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux, GenerateEMRIWaveform
from few.utils.utility import (get_overlap,
                               get_mismatch,
                               get_fundamental_frequencies,
                               get_separatrix,
                               get_mu_at_t,
                               get_p_at_t,
                               get_kerr_geo_constants_of_motion,
                               xI_to_Y,
                               Y_to_xI)

from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.waveform import SchwarzschildEccentricWaveformBase
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.summation.directmodesum import DirectModeSum
from few.utils.constants import *
from few.summation.aakwave import AAKSummation
from few.waveform import Pn5AAKWaveform, AAKWaveformBase
import time
from multiprocessing import Pool
import multiprocessing



def physcis_to_fit(M_log,mu,e0,p0):
    
    traj = EMRIInspiral(func="SchwarzEccFlux")
    M = 10**M_log
    fitting_para = np.zeros(40) 
    fitting_para[:4] = M_log,mu,e0,p0
    
    try:
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M,mu, 0.0,p0, e0, 1.0,new_t=new_t, upsample=True, fix_t=True)
    except:
        return fitting_para
     
    h_all = {}
    
    
    if len(t) / len_segment < 3:
        comput_N_seg = int(len(t) / len_segment)
    else:
        comput_N_seg = int(len(t) / len_segment)
    
    
    for seg in range(comput_N_seg):
        for n in range(3):
            Phi_phi_seg = Phi_phi[seg*len_segment:(seg+1)*len_segment] - np.int0(Phi_phi[seg*len_segment]/(2*np.pi)) * 2*np.pi
            Phi_r_seg = Phi_r[seg*len_segment:(seg+1)*len_segment] - np.int0(Phi_r[seg*len_segment]/(2*np.pi)) * 2*np.pi
            y3 = np.polyfit(Time_observation,2*Phi_phi_seg + n*Phi_r_seg,3)
            
            fitting_para[4+ 12*seg + 4*n:4+ 12*seg + 4*n +4] = y3
            
            
    return fitting_para
            
            
            

if __name__=="__main__":
    
    tyr = 24*3600*365
    dt = 15  
    Time_observation = np.arange(0,0.1*tyr,dt)
  
    
    new_t = np.arange(0,0.3*tyr,dt)
    N_segment = 3
    
    len_segment = len(Time_observation)
    
    len_M = 100
    len_e = 10
    len_mu = 10
    len_p = 100
    
    
    re = np.zeros([len_M*len_mu*len_e*len_p,40])

    i = 0
    j = 0
    t0 = time.time()
    # pool = Pool(processes=150)
    array_log_M = np.linspace(5.85, 6.17,len_M)
    num_tasks = 50
    
    for i_m in range(len_M):
        np.save('/data1/yecq/para_estimation_Michaelson/source_1/fit_to_physic/gaid.npy',re)
        t1 = time.time()
        print(f'当前进度 {j}%,花费时间{t1 -t0}')
        j = j + 1
        pool_list = []
        pool = multiprocessing.Pool(processes=num_tasks)
        for mu in np.linspace(5, 15,len_mu):
            for e in np.linspace(0.0, 0.7,len_e):
                for p in  np.linspace(7.68, 9.48,len_p):     
                    pool_list.append(pool.apply_async(physcis_to_fit, (array_log_M[j],mu,e,p)))
                    
        # 所有任务提交给池后，关闭池，不允许添加新任务
        pool.close()

        # 等待所有任务完成
        pool.join()
                    
        for pool_i in range(len_mu*len_e*len_p):
            re[i,:] = pool_list[pool_i].get()
            i =i +1 

