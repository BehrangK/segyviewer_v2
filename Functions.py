# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 08:41:06 2020

@author: koushb
"""

from skimage.filters import sobel
import numpy as np
from scipy.signal import  lfilter
import cmath
from scipy.signal.signaltools import hilbert
from scipy.signal import butter


def do_filter(x,file_name):
    #file_name=filters.value
    
    data = np.load(file_name, fix_imports=True, encoding='bytes') # array containing dict, dtype 'object'    
    b,a=data['ba']
    if x.ndim > 1:
        for i in range(x.shape[-1]):    
            x[...,i]= lfilter(b, a, x[...,i])
    else:
        x=lfilter(b, a, x)
    return(x)


def low_high_pass(x,a,b):
    if x.ndim > 1:
        for i in range(x.shape[-1]):    
            x[...,i]= low_high_pass(x[...,i],a,b)
    else:
        x=lfilter(b, a, x)
    return(x)
    
def lowpass(x,a,b):    
    if x.ndim > 1:
        for i in range(x.shape[-1]):    
            x[...,i]= lowpass(x[...,i],d,order)
    else:
        b, a = butter(order, d, 'lowpass', fs=1000)
        x=lfilter(b, a, x)
    return(x)

def highpass(x,d,order=6):    
    if x.ndim > 1:
        for i in range(x.shape[-1]):    
            x[...,i]= highpass(x[...,i],d,order)
    else:
        
        x=lfilter(b, a, x)
    return(x)


# from https://github.com/geopyteam/cognitivegeo/blob/master/cognitivegeo/src/seismic/attribute.py

def calcEdge_detection(x):
    
    x=np.squeeze(x)
    if x.ndim > 2:
        #print(x.shape)
        for i in range(x.shape[-1]):    
            x[...,i]= calcEdge_detection(x[...,i])    
    else:
        x=sobel(x)
    return x
    
    


    return(x)

def calcCumulativeSum(x):
    """
    Calculate cusum attribute
    Args:
        x: seismic data in 3D matrix [Z/XL/IL]
    Return:
        cusum attribute as 3D matrix
    """
    return np.cumsum(x, axis=0)


def calcPhaseShift_trace(x,d):
    #from https://stackoverflow.com/questions/52179919/amplitude-and-phase-spectrum-shifting-the-phase-leaving-amplitude-untouched
    
    phase=d*(np.pi/180)
    signalFFT = np.fft.rfft(x)
    

    ## Get Power Spectral Density
    signalPSD = np.abs(signalFFT) ** 2
    signalPSD /= len(signalFFT)**2

    ## Get Phase
    signalPhase = np.angle(signalFFT)

    ## Phase Shift the signal +90 degrees
    newSignalFFT = signalFFT * cmath.rect( 1., phase )

    ## Reverse Fourier transform
    newSignal = np.fft.irfft(newSignalFFT,x.shape[0])
    return newSignal


def calcPhaseShift(x,d):
    x=np.squeeze(x)
    if x.ndim > 1:
        #print(x.shape)
        for i in range(x.shape[-1]):    
            x[...,i]= calcPhaseShift_trace(x[...,i],d)    
    else:
        x=calcPhaseShift(x,y)
    return x

   


def calcFirstDerivative_trace(y,x): #x:time sample y: trace
    #x=TimeSamples
    
    dy = np.zeros(y.shape,np.float)
    #dy = np.empty_like(x)
  
    dy[0:-1] = np.diff(y)/np.diff(x)
    dy[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])

    return(dy)



def calcFirstDerivative(x,y):
    """
    Calculate first derivative attribute
    Args:
        x: seismic data in 3D matrix [Z/XL/IL]
    Return:
        first-derivative attribute as 3D matrix
    """
    x=np.squeeze(x)
    if x.ndim > 1:
        #print(x.shape)
        for i in range(x.shape[-1]):    
            x[...,i]= calcFirstDerivative(x[...,i],y)    
    else:
        x=calcFirstDerivative_trace(x,y)
    return x




def calcInstantEnvelope(x,d):
    """
    Calculate instantaneous envelop attribute
    Args:
        x: seismic data in 3D matrix [Z/XL/IL]
    Return:
        Instantaneous envelop attribute as 3D matrix
    """
    #https://www.gaussianwaves.com/2017/04/extracting-instantaneous-amplitude-phase-frequency-hilbert-transform/
    
    Z=hilbert(x, axis=0,N=d)
    attrib = np.abs(Z)

    return attrib


def calcInstantPhase(x,d):
    """
    Calculate instantaneous phase attribute
    Args:
        x: seismic data in 3D matrix [Z/XL/IL]
    Return:
        Instantaneous phase attribute as 3D matrix
    """
    Z=hilbert(x, axis=0,N=d)
    inst_phase = np.unwrap(np.angle(Z))#inst phase
    #attrib = np.diff(inst_phase)/(2*np.pi)*fs #inst frequency

    #attrib = np.angle(hilbert(x, axis=0,N=d))
    attrib = inst_phase * 180.0 / np.pi
    #
    return attrib

def calcInstantFrequency(x,d):
    """
    Calculate instantaneous frequency attribute
    Args:
        x: seismic data in 3D matrix [Z/XL/IL]
    Return:
        Instantaneous frequency attribute as 3D matrix
    """
    Z=hilbert(x, axis=0,N=d)
    inst_phase = np.unwrap(np.angle(Z))#inst phase
    attrib = np.diff(inst_phase)/(2*np.pi)*1000 #inst frequency

    #instphase = np.unwrap(np.angle(hilbert(x, axis=0,,N=d)), axis=0) * 0.5 / np.pi
    #attrib = np.zeros(np.shape(x))
    #attrib[1:-1, ...] = 0.5 * (instphase[2:,...] - instphase[0:-2,...])
    return attrib

def calcInstantCosPhase(x,d):
    """
    Calculate instantaneous cosine of phase attribute
    Args:
        x: seismic data in 3D matrix [Z/XL/IL]
    Return:
        Cosine of phase attribute as 3D matrix
    """
    Z=hilbert(x, axis=0,N=d)
    attrib = np.angle(Z)
    attrib = np.cos(attrib)
    #
    return attrib

def calcInstantQuadrature(x,d):
    """
    Calculate instantaneous quadrature attribute
    Args:
        x: seismic data in 3D matrix [Z/XL/IL]
    Return:
        Instantaneous quadrature attribute as 3D matrix
    """
    Z=hilbert(x, axis=0,N=d)
    attrib = np.imag(Z)
    return attrib


def calc_att(x,y,att='Phase Shift',d=0):
    #att=attribute.value
    #print(x.shape,y.shape)
    r=x.copy()
    if att=='Phase Shift':
        #print('Phase')
        r=calcPhaseShift(x,d)
    elif att=='Edge Detection':
        r=calcEdge_detection(x)
    elif att=='CumulativeSum':
        r=calcCumulativeSum(x)
    elif att=='FirstDerivative':
        r=calcFirstDerivative(x,y)
    elif att=='InstantEnvelope':
        r=calcInstantEnvelope(x,d)
    elif att=='InstantQuadrature':
        r=calcInstantQuadrature(x,d)
    elif att=='InstantPhase':
        r=calcInstantPhase(x,d)
    elif att=='InstantFrequency':
        r=calcInstantFrequency(x,d)
    elif att=='InstantCosPhase':
        r=calcInstantCosPhase(x,d)
    elif att=='Low Pass':
        order=6
        b, a = butter(order, d, 'lowpass', fs=1000)
        r=low_high_pass(x,a,b)
    elif att=='High Pass':
        order=6
        b, a = butter(order, d, 'highpass', fs=1000)
        r=low_high_pass(x,a,b)
    else:
        raise NameError('Could not find the Attribute name:'+att)
    return(r)