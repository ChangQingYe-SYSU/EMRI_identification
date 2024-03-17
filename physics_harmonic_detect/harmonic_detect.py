#%%
from copy import copy
import sys
import os
from typing_extensions import ParamSpec
import multiprocessing as mp
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
from few.waveform import FastSchwarzschildEccentricFlux
from few.trajectory.inspiral import EMRIInspiral
import multiprocessing as mp
from few.utils.utility import (get_fundamental_frequencies, get_p_at_t)
import pyfftw
from scipy.fftpack import fft,ifft
import random
import math
import multiprocessing as mp
import h5py
import sys
import time
from multiprocessing import Pool




def getparameters(produce_siganal,seed = None):
    #设置初始参数
    if seed is not None:
            np.random.seed(seed)
    M = float(pow(10,random.uniform(4,7)))
    mu = float(random.uniform(5,15))
    e0 = float(random.uniform(0.1,0.5))
    if produce_siganal:
        t_out =  0.48  #定义模板
    else:
        t_out = 1.24256850e+07 / tyr    # 定义信号时所用
        
    a = 0.0
    x0 = 1.0
    T = 0.5
    dt = 15.0
    qK = float(random.uniform(0,2*math.pi))
    phiK = float(random.uniform(0,math.pi))
    phiS = float(random.uniform(0,math.pi))
    qS = float(random.uniform(0,2*math.pi))
    dist = 1  # distance
    Phi_phi0 = float(random.uniform(0,2*math.pi))
    Phi_theta0 = float(random.uniform(0,2*math.pi))
    Phi_r0 = float(random.uniform(0,2*math.pi))

    #计算t0时刻的p0,在t_out取(0.2,0.5)年的情况下
    
    if M/mu < 1e4:
        return None

    traj_args = [M, mu, a, e0, x0]
    # print (traj_args)

    traj_module = EMRIInspiral(func="SchwarzEccFlux",Phi_phi0 =Phi_phi0,Phi_theta0 =Phi_theta0,Phi_r0 =Phi_r0)

    p_new = get_p_at_t(
        traj_module,
        t_out,
        traj_args,
        index_of_p=3,
        index_of_a=2,
        index_of_e=4,
        index_of_x=5,
        traj_kwargs={},
        # kerr_separatrix=False,
        xtol=2e-17,
        rtol=8.881784197001252e-16,
        bounds=None,
    )


    # print('p0 = {} will create a waveform that is {} years long, given the other input parameters.'.format(p_new, t_out))

    d_Parameters = {}
    d_Parameters['M'] = M;d_Parameters['mu'] = mu;d_Parameters['p0'] = p_new
    d_Parameters['e0'] = e0;d_Parameters['a'] = a;d_Parameters['x0'] = x0
    d_Parameters['t_out'] = t_out;d_Parameters['T'] = T;d_Parameters['dt'] = dt
    d_Parameters['qK'] = qK;d_Parameters['phiK'] = phiK
    d_Parameters['qS'] = qS;d_Parameters['phiS'] = phiS
    d_Parameters['Phi_phi0'] = Phi_phi0;d_Parameters['Phi_theta0'] = Phi_theta0;d_Parameters['Phi_r0'] = Phi_r0
    d_Parameters['dist'] = dist
    return(d_Parameters)

'''这里是用于计算Michaelson 通道下的时间域的波形'''




def get_F(t,qS,phiS,qK,phiK):
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

    
    """get the psi"""


    cqS = np.cos(qS)
    sqS = np.sin(qS)

    cphiS = np.cos(phiS)
    sphiS = np.sin(phiS)

    cqK = np.cos(qK)
    sqK = np.sin(qK)

    cphiK = np.cos(phiK)
    sphiK = np.sin(phiK)

    # get polarization angle

    up_ldc = cqS * sqK * np.cos(phiS - phiK) - cqK * sqS
    dw_ldc = sqK * np.sin(phiS - phiK)

    if dw_ldc != 0.0:
        psi_ldc = -np.arctan2(up_ldc, dw_ldc)

    else:
        psi_ldc = 0.5 * np.pi

    _F_pluss = D_pluss * np.cos(psi_ldc) - D_cross * np.sin(psi_ldc)
    _F_cross = D_pluss * np.sin(psi_ldc) + D_cross * np.cos(psi_ldc)

    return _F_pluss,_F_cross



def fast_f(pa,specific_modes,check_phi2=True):
    num_threads = 1
    use_gpu = False
    # keyword arguments for summation generator (InterpolatedModeSum)
    sum_kwargs = {
        "use_gpu": False,  # GPU is availabel for this type of summation
        "pad_output": False,
        'qS': pa['qS'],
        "phiS":pa["phiS"]
    }
    # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
    inspiral_kwargs={
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
        }

    # keyword arguments for inspiral generator (RomanAmplitude)
    amplitude_kwargs = {
        "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
        "use_gpu": use_gpu  # GPU is available in this class
    }
    # keyword arguments for Ylm generator (GetYlms)
    Ylm_kwargs = {
        "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
    }


    few = FastSchwarzschildEccentricFlux(
        inspiral_kwargs=inspiral_kwargs,
        amplitude_kwargs=amplitude_kwargs,
        Ylm_kwargs=Ylm_kwargs,
        sum_kwargs=sum_kwargs,
        use_gpu=use_gpu,
        num_threads=num_threads,  # 2nd way for specific classes
    )
    if specific_modes == None:
        wave = few(pa['M'],pa['mu'],pa['p0'],pa['e0'],pa['qS'],pa['phiS'],\
                Phi_phi0=pa['Phi_phi0'],Phi_r0=pa['Phi_r0'],dt=pa['dt'],T=pa['T'],dist=pa['dist'],\
                eps=1e-2)
    else:
        wave = few(pa['M'],pa['mu'],pa['p0'],pa['e0'],pa['qS'],pa['phiS'],\
                Phi_phi0=pa['Phi_phi0'],Phi_r0=pa['Phi_r0'],dt=pa['dt'],T=pa['T'],dist=pa['dist'],\
                mode_selection=specific_modes,include_minus_m=True)

    t_wave = np.arange(len(wave)) *15

    F_pluss,F_cross = get_F(t_wave,pa['qS'],pa['phiS'],pa['qK'],pa['phiK'])

    if check_phi2:
        h = wave.real * F_pluss + wave.imag *F_cross

        plunge = len(wave) * pa['dt']

        if len(wave) < int(0.5 * tyr / dt):
            temple_wave = np.zeros((int(0.5 * tyr /dt),2))
            temple_wave[:,0] = np.arange(0,0.5*tyr,15)
            temple_wave[:len(wave),1] = h

        return (temple_wave,plunge)
    else:
        t_wave_0 = wave
        t_wave_5 = wave * np.exp(1j * np.pi/2)
        
        h_0 =  t_wave_0.real * F_pluss + t_wave_0.imag *F_cross
        h_5 =  t_wave_5.real * F_pluss + t_wave_5.imag *F_cross
        
        plunge = len(wave) * pa['dt']

        if len(wave) < int(0.5 * tyr / dt):
            temple_wave_0 = np.zeros((int(0.5 * tyr /dt),2))
            temple_wave_0[:,0] = np.arange(0,0.5*tyr,15)
            temple_wave_0[:len(wave),1] = h_0
            temple_wave_5 = np.zeros((int(0.5 * tyr /dt),2))
            temple_wave_5[:,0] = np.arange(0,0.5*tyr,15)
            temple_wave_5[:len(wave),1] = h_5
            return temple_wave_0,temple_wave_5






'''TQ_PSD'''
def michelson_TQ(f):
    c = 2.99792458e8
    L_TQ = 3 ** 0.5 * 1e8
    f_star_TQ = c / (2 * np.pi * L_TQ)
    Sa_TQ = 1e-15 ** 2
    Sx_TQ = 1e-12 ** 2  
    return 1/L_TQ**2 * (2*(1+np.cos(f/f_star_TQ)**2)*Sa_TQ / (2*np.pi*f)**4 * (1+1e-4/f)+Sx_TQ)


'''Comput SNR'''
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
    #{{{
    #fr = freq[1:]
    imax = len(fr)-1
    if (fmax != None):
        imax = np.argwhere(fr >fmax)[0][0] +1
    imin = 1
    if (fmin != None):
        imin = np.argwhere(fr >fmin)[0][0] +1

    SNR2 = (4.0*df) * np.sum(np.real(d[imin:imax]*np.conjugate(x[imin:imax])/S[imin:imax]))

    return(SNR2)


'''用于计算模板的并合时间co-Correlation'''

def AE_to_plungetime_to_ifft(signal_f,temple_f):

    normal_temple_A = np.sqrt(Computeinner(temple_f,temple_f,Sa, freq, df,fmin=1e-4))

    SNR_no_sum_AE= (4.0*df) * signal_f * np.conjugate(temple_f/normal_temple_A)/Sa

    tm_ifft_A = ifft(SNR_no_sum_AE) * len(freq)

    t_A = np.argmax(np.abs((np.real(tm_ifft_A)))) * 15 *2

    re = [t_A,np.max(np.abs((np.real(tm_ifft_A))))]
    #pr


    pl = False
    if (pl):
        plt.figure(figsize=(8, 6))
        N = len(tm_ifft_A)
        time = np.arange(N)*15*2 /(3600*24*365)
        plt.plot(time,np.real(tm_ifft_A),label= 'signal_A')
        plt.xlabel('time[tyr]',fontsize=20)
        plt.ylabel('np.real(ifft)',fontsize=20)
        plt.legend(loc='upper left')
        plt.savefig(f'./fig{i}.jpg')
        plt.close()

    return(re) 

'''参数倒推到并合时间'''
def evolution_para_time(l_t_params,l_re,l_det,write_f):
    t_params_new = {}
    traj_module = EMRIInspiral(func="SchwarzEccFlux")
    t_params_new=l_t_params.copy()
    #
    new_t = np.arange(0, 0.5*tyr, 30) 
    inputs = [t_params_new['M'],t_params_new['mu'], t_params_new['a'], t_params_new['p0'],t_params_new['e0'],t_params_new['x0']]
    traj_kwargs = {'T': 0.98}
    Phi_phi0 = t_params_new['Phi_phi0']
    Phi_theta0 = t_params_new['Phi_theta0']
    Phi_r0 = t_params_new['Phi_r0']

    out = traj_module(*inputs,Phi_phi0 =Phi_phi0,Phi_theta0 =Phi_theta0,**traj_kwargs,new_t=new_t, upsample=True, fix_t=True)
    # Phi_phi0=t_params_new['Phi_phi0'],Phi_theta0=t_params_new['Phi_theta0'],Phi_r0=t_params_new['Phi_r0']

    '''把参数演化保存'''
    write_f['evlution'] = out[:2][:]

    plunge = l_re[0]/tyr - l_det
    #print ("plunge time in ",plunge)

    step = int ((out[0][-1] - plunge *  tyr) /30)
    t,p,e,x,Phi_phi,Phi_theta, Phi_r = out[0][step],out[1][step],out[2][step],out[3][step],out[4][step],out[5][step],out[6][step]

    t_params_new['e0']= e
    t_params_new['p0']= p
    t_params_new['Phi_phi0']=Phi_phi
    t_params_new['Phi_theta0']=Phi_theta
    t_params_new['Phi_r0']=Phi_r

    return (t_params_new)


'''求解初始相位'''

def constans_aj(_f_signal,_f_temple,Sa,ratio = True):
    normal_temple_A = Computeinner(_f_temple, _f_temple, Sa, freq, df,fmin=1e-4)
    SNRda = Computeinner(_f_signal, _f_temple/np.sqrt(normal_temple_A), Sa, freq, df,fmin=1e-4)

    # print(SNRda,normal_temple_A)
            
    if (ratio):
        return (SNRda)
    else:
        return (SNRda,normal_temple_A)




def tanpanduan(theta):
    if theta < 0 :
        theta = theta + 2 * math.pi
    if theta > 2 * math.pi:
        theta = theta % (2 * math.pi)
    return (theta)

def computer_initial_phi(AE_f_signal,AE_0,AE_5,Sa,aplm_phi = True):
    a_0 = constans_aj(AE_f_signal,AE_0,Sa,True)
    a_1 = constans_aj(AE_f_signal,AE_5,Sa,True)
    if (a_0 ==0 ):
        initial_large_phi_0 = 0
    else:
        initial_large_phi_0 = np.arctan2(a_1,a_0)
        initial_large_phi_0 = tanpanduan(initial_large_phi_0)

    if (aplm_phi):
        return (initial_large_phi_0)
    else:
        return (np.sqrt(a_0**2 + a_1**2))


def get_detect(i):
        
    h5f = h5py.File(f'/data1/yecq/para_estimation_Michaelson/source_1/harmonic/template/temple_{i}.hdf5','w')
    
    t_params = None
    
    while t_params == None:
        try:
            mode_section = None
            t_params = getparameters(True) #产生模板在plunge time 在0.48年的随机参数
            temple_wave_obervation_time,temple_plunge = fast_f(t_params,mode_section) # 产生 [2,2,0] 的谐频模板，并考虑TDI响应(l,m,n)
        except:
            t_params = None


    temple_f = TransformData(temple_wave_obervation_time[:,1])

    '''保存文件'''
    h5_pa = h5f.create_group('params')
    for key in t_params.keys():
        h5_pa[key] = t_params[key]
    
    h5f['Michaelson_freq_domin'] = temple_f



    '''计算并合时间'''

    re = AE_to_plungetime_to_ifft(signal_f,temple_f)


    det = 0.5 - temple_plunge/tyr

    temple_plunge_pa = evolution_para_time(t_params,re,det,h5f)
    
    h5f.close()
        
    ############计算完plunge time，计算相位

    hamoic_pa = {}

    hamoic_pa = temple_plunge_pa.copy()

    hamoic_pa['Phi_phi0'] = 0
    hamoic_pa['Phi_r0'] = 0

    d_SNR_all = {}
    l_SNR_all = []


    for n in [-2,-1,0,1,2]:
        for l in [2,3]:
            for m in [1,2]:
                
                temple_wave_0_t,temple_wave_5_t = fast_f(hamoic_pa,[(l,m,n)],False)
                temple_wave_0_f = TransformData(temple_wave_0_t[:,1])
                temple_wave_5_f = TransformData(temple_wave_5_t[:,1])
                snr_homic = computer_initial_phi(signal_f,temple_wave_0_f,temple_wave_5_f,Sa,aplm_phi = False)
                d_SNR_all[(l,m,n)] = snr_homic
                l_SNR_all.append(snr_homic)
                snr_homic = 0

    SNR_relsts = np.sum(np.array(l_SNR_all))

    if np.max(np.array(l_SNR_all)) > 3:
        re_write('hamonic_pa',str(hamoic_pa))
        re_write('0_5_para',str(t_params))
        re_write('phi_snr',str(d_SNR_all))
        re_write('snr',str(np.max(np.array(l_SNR_all))))



def pell(i):
    try:
        get_detect(i)
    except:
        pass

    return 0



def re_write(name,asci):
    w = open(f'/data1/yecq/para_estimation_Michaelson/source_1/harmonic/{name}_1.txt',"a")
    w.write(str(asci)+ '\n')
    w.close()



if __name__=="__main__":
    t1 = time.time()
    tyr = 24*3600*365
    dt = 15

    ##############################
    with h5py.File('/data1/yecq/para_estimation_Michaelson/source_1/signal/signal_1.hdf5','r') as f:
        Michaelson_freq_signal =  f['Michaelson_freq_domin'][:]
        Michaelson_freq_noise = f['Michaelson_niose_freq'][:]
        f.close()

    df = 1 / (int(0.5 * tyr /dt) * dt )
    freq = np.arange(int(int(0.5 * tyr /dt) /2 +1)) *df

    Sa = michelson_TQ(freq[1:])

    signal_f = Michaelson_freq_signal + Michaelson_freq_noise



    snr2 = Computeinner(signal_f, Michaelson_freq_signal,Sa,freq,df,10**-4)

    print ('the signal snr is {}'.format(np.sqrt(snr2)))


    pool = Pool(50)
    for i in range(100):
        re = pool.apply_async(func=pell,args=(i,))

    pool.close()
    pool.join() 

