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
#from scipy import ndimage
from scipy.fftpack.helper import next_fast_len

def hilbert3(x,axis=0):
    m=np.ma.size(x,axis=axis)
    n=next_fast_len(m)
    r=hilbert(x,N=n,axis=axis)
    r=r.take(indices=range(m), axis=axis)
    return(r)

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
    #signalPhase = np.angle(signalFFT)

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
            x[...,i]= calcPhaseShift(x[...,i],d)    
    else:
        x=calcPhaseShift_trace(x,d)
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




def calcInstantEnvelope(x):
    """
    Calculate instantaneous envelop attribute
    Args:
        x: seismic data in 3D matrix [Z/XL/IL]
    Return:
        Instantaneous envelop attribute as 3D matrix
    """
    #https://www.gaussianwaves.com/2017/04/extracting-instantaneous-amplitude-phase-frequency-hilbert-transform/
    

    Z=hilbert3(x, axis=0)
    attrib = np.abs(Z)

    return attrib

def calcInstantPhase(x):
    """
    Calculate instantaneous phase attribute
    Args:
        x: seismic data in 3D matrix [Z/XL/IL]
    Return:
        Instantaneous phase attribute as 3D matrix
    """
    Z=hilbert3(x, axis=0)
    #inst_phase = np.unwrap(np.angle(Z))#inst phase
    #attrib = np.diff(inst_phase)/(2*np.pi)*fs #inst frequency

    inst_phase = np.angle(Z)
    attrib = inst_phase * 180.0 / np.pi
    #
    return attrib


def calcInstantFrequency(x,y):
    """
    Calculate instantaneous frequency attribute
    Args:
        x: seismic data in 3D matrix [Z/XL/IL]
    Return:
        Instantaneous frequency attribute as 3D matrix
    """
    #Z=hilbert3(x, axis=0)
    #inst_phase = np.unwrap(np.angle(Z))#inst phase
    #attrib = np.diff(inst_phase)/(2*np.pi)*1000 #inst frequency

    #instphase = np.unwrap(np.angle(hilbert(x, axis=0,,N=d)), axis=0) * 0.5 / np.pi
    #attrib = np.zeros(np.shape(x))
    #attrib[1:-1, ...] = 0.5 * (instphase[2:,...] - instphase[0:-2,...])
    instphase =calcInstantPhase(x)
    attrib=calcFirstDerivative(instphase,y)
    return attrib

def calcInstantCosPhase(x):
    """
    Calculate instantaneous cosine of phase attribute
    Args:
        x: seismic data in 3D matrix [Z/XL/IL]
    Return:
        Cosine of phase attribute as 3D matrix
    """
    Z=hilbert3(x, axis=0)
    attrib = np.angle(Z)
    attrib = np.cos(attrib)
    #
    return attrib

def calcInstantQuadrature(x):
    """
    Calculate instantaneous quadrature attribute
    Args:
        x: seismic data in 3D matrix [Z/XL/IL]
    Return:
        Instantaneous quadrature attribute as 3D matrix
    """
    Z=hilbert3(x, axis=0)
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
        r=calcInstantEnvelope(x)
    elif att=='InstantQuadrature':
        r=calcInstantQuadrature(x)
    elif att=='InstantPhase':
        r=calcInstantPhase(x)
    elif att=='InstantFrequency':
        r=calcInstantFrequency(x,y)
    elif att=='InstantCosPhase':
        r=calcInstantCosPhase(x)
    elif att=='Low Pass':
        order=6
        b, a = butter(order, d, 'lowpass', fs=1000)
        r=low_high_pass(x,a,b)
    elif att=='High Pass':
        order=6
        b, a = butter(order, d, 'highpass', fs=1000)
        r=low_high_pass(x,a,b)
    elif att=='Relative Acoustic Impedance':
        order=6
        b, a = butter(order, d, 'highpass', fs=1000)
        r=calcCumulativeSum(x)
        r=low_high_pass(r,a,b)
    else:
        raise NameError('Could not find the Attribute name:'+att)
    return(r)


def array_for_sliding_window(x, wshape):
    """Build a sliding-window representation of x.
    The last dimension(s) of the output array contain the data of
    the specific window.  The number of dimensions in the output is 
    twice that of the input.
    Parameters
    ----------
    x : ndarray_like
       An array for which is desired a representation to which sliding-windows 
       computations can be easily applied.
    wshape : int or tuple
       If an integer, then it is converted into a tuple of size given by the 
       number of dimensions of x with every element set to that integer.
       If a tuple, then it should be the shape of the desired window-function
    Returns
    -------
    out : ndarray
        Return a zero-copy view of the data in x so that operations can be 
        performed over the last dimensions of this new array and be equivalent 
        to a sliding window calculation.  The shape of out is 2*x.ndim with 
        the shape of the last nd dimensions equal to wshape while the shape 
        of the first n dimensions is found by subtracting the window shape
        from the input shape and adding one in each dimension.  This is 
        the number of "complete" blocks of shape wshape in x.
    Raises
    ------
    ValueError
        If the size of wshape is not x.ndim (unless wshape is an integer).
        If one of the dimensions of wshape exceeds the input array. 
    Examples
    --------
    >>> x = np.linspace(1,5,5)
    >>> x
    array([ 1.,  2.,  3.,  4.,  5.])
    >>> array_for_rolling_window(x, 3)
    array([[ 1.,  2.,  3.],
           [ 2.,  3.,  4.],
           [ 3.,  4.,  5.]])
           
    >>> x = np.arange(1,17).reshape(4,4)
    >>> x
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12],
           [13, 14, 15, 16]])
    >>> array_for_rolling_window(x, 3)
    array([[[[ 1,  2,  3],
             [ 5,  6,  7],
             [ 9, 10, 11]],
            [[ 2,  3,  4],
             [ 6,  7,  8],
             [10, 11, 12]]],
           [[[ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15]],
            [[ 6,  7,  8],
             [10, 11, 12],
             [14, 15, 16]]]])
    """
    x = np.asarray(x)

    try:
        nd = len(wshape)
    except TypeError:
        wshape = tuple(wshape for i in x.shape)
        nd = len(wshape)
    if nd != x.ndim:
        raise ValueError("wshape has length {0} instead of "
                         "x.ndim which is {1}".format(len(wshape), x.ndim)) 
    
    out_shape = tuple(xi-wi+1 for xi, wi in zip(x.shape, wshape)) + wshape
    if not all(i>0 for i in out_shape):
        raise ValueError("wshape is bigger than input array along at "
                         "least one dimension")

    out_strides = x.strides*2
    
    return np.lib.stride_tricks.as_strided(x, out_shape, out_strides)

    
def marfurt_semblance(region):
    # We'll need an ntraces by nsamples array
    # This stacks all traces within the x-y "footprint" into one
    # two-dimensional array.
    region = region.reshape(-1, region.shape[-1])
    ntraces, nsamples = region.shape
   

    square_of_sums = np.sum(region, axis=0)**2
    sum_of_squares = np.sum(region**2, axis=0)
    
    sembl = square_of_sums.sum() / sum_of_squares.sum()
    
    r=sembl/ ntraces
    #print(r)
    return r


def marfurt_semblance2(region):
    region = region.reshape(-1, region.shape[-1])
    ntraces, nsamples = region.shape

    cov = region.dot(region.T)
    sembl = cov.sum() / cov.diagonal().sum()
    return sembl / ntraces

def gersztenkorn_eigenstructure(region):
    # Once again, stack all of the traces into one 2D array.
    region = region.reshape(-1, region.shape[-1])

    cov = region.dot(region.T)
    vals = np.linalg.eigvalsh(cov)
    return vals.max() / vals.sum()


def complex_eigenstructure(region):
    region = region.reshape(-1, region.shape[-1])

    region = hilbert(region, axis=-1)
    region = np.hstack([region.real, region.imag])

    cov = region.dot(region.T)
    vals = np.linalg.eigvals(cov)
    return np.abs(vals.max() / vals.sum())