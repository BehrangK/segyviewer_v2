# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:54:09 2020

@author: koushb
"""
#https://github.com/seg/tutorials-2015/blob/master/1512_Semblance_coherence_and_discontinuity/Discontinuity_tutorial.ipynb
import numpy as np
import segyio
import matplotlib.pyplot as plt

from scipy import ndimage,signal


segyfile = r"C:\Users\koushb\OneDrive - Husky Energy\ML\ML_SI_Data\Tucker\PP.segy"
f = segyio.open(segyfile)
il, xl, t = f.ilines, f.xlines, -f.samples

dt = t[0] - t[1]



SI_cube = np.flip(segyio.cube(f),axis=2)


ij=SI_cube.std(axis=2)
ij[ij>0]=1
plt.imshow(ij)
ii,jj=np.where(ij==0)
SI_cube[ii,jj,:]=np.nan
d=SI_cube[:,180,:].T


m=(t.max()-t.min())/SI_cube.shape[2]
yticks=np.array([0,50,100,150,200,250,300,333])


fig, ax = plt.subplots(1,1)
img = ax.imshow(d, cmap='RdYlBu',vmin=0.3*d.min(), vmax=0.3*d.max())
ax.set_xticks(il[::60])
ax.set_xticklabels(il[::50])
ax.set_yticks(yticks)
ax.set_yticklabels(np.round(yticks*m+t.min()))
ax.invert_yaxis()
#fig.colorbar(img)
plt.show()

def plot(data, title='',j0=-1,k0=-1):
    # We'll take slices half-way through the volume (// is integer division)
    if j0==-1:
        j0 = data.shape[1] // 2
    if k0==-1:
        k0 = data.shape[2] // 2
    # Setup subplots where one is 3x the height of the other
    # Our data has a fairly narrow time range, so we'll make a cross section 
    # that's 1/3 the height of the time slice
    gs = plt.GridSpec(4, 1)
    fig = plt.figure(figsize=(8, 9))
    ax1 = fig.add_subplot(gs[:-1], anchor='S')
    ax2 = fig.add_subplot(gs[-1], anchor='N')
    
    # Plot the sections
    ax1.imshow(data[:,:,k0].T, cmap='RdYlBu')
    ax2.imshow(data[:,j0,:].T, cmap='RdYlBu')
    
    # Mark the cross section locations...
    for ax, loc in zip([ax1, ax2], [j0, k0]):
        ax.axhline(loc, color='red')
        ax.set(xticks=[], yticks=[])
    
    ax1.set(title=title)
    plt.show()

plot(SI_cube, 'Input Dataset',180,100)



def bahorich_coherence(data, zwin):
    ni, nj, nk = data.shape
    out = np.zeros_like(data)
    
    # Pad the input to make indexing simpler. We're not concerned about memory usage.
    # We'll handle the boundaries by "reflecting" the data at the edge.
    padded = np.pad(data, ((0, 1), (0, 1), (zwin//2, zwin//2)), mode='reflect')

    for i, j, k in np.ndindex(ni, nj, nk):
        # Extract the "full" center trace
        center_trace = data[i,j,:]
        
        # Use a "moving window" portion of the adjacent traces
        x_trace = padded[i+1, j, k:k+zwin]
        y_trace = padded[i, j+1, k:k+zwin]

        # Cross correlate. `xcor` & `ycor` will be 1d arrays of length
        # `center_trace.size - x_trace.size + 1`
        xcor = np.correlate(center_trace, x_trace)
        ycor = np.correlate(center_trace, y_trace)
        
        # The result is the maximum normalized cross correlation value
        center_std = center_trace.std()
        px = xcor.max() / (xcor.size * center_std * x_trace.std())
        py = ycor.max() / (ycor.size * center_std * y_trace.std())
        out[i,j,k] = np.sqrt(px * py)

    return out

bahorich = bahorich_coherence(SI_cube, 5)
plot(bahorich, 'Bahorich & Farmer (1995)')

def moving_window(data, window, func):
    # `generic_filter` will give the function 1D input. We'll reshape it for convinence
    wrapped = lambda region: func(region.reshape(window))
    
    # Instead of an explicit for loop, we'll use a scipy function to do the same thing
    # The boundaries will be handled by "reflecting" the data, by default
    return ndimage.generic_filter(data, wrapped, window)

def marfurt_semblance(region):
    # We'll need an ntraces by nsamples array
    # This stacks all traces within the x-y "footprint" into one
    # two-dimensional array.
    region = region.reshape(-1, region.shape[-1])
    ntraces, nsamples = region.shape

    square_of_sums = np.sum(region, axis=0)**2
    sum_of_squares = np.sum(region**2, axis=0)
    sembl = square_of_sums.sum() / sum_of_squares.sum()
    return sembl / ntraces



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

    region = signal.hilbert(region, axis=-1)
    region = np.hstack([region.real, region.imag])

    cov = region.dot(region.T)
    vals = np.linalg.eigvals(cov)
    return np.abs(vals.max() / vals.sum())

marfurt = moving_window(SI_cube, (3, 3, 9), marfurt_semblance)
plot(marfurt, 'Marfurt et al (1998)')

gersztenkorn = moving_window(SI_cube, (3, 3, 9), gersztenkorn_eigenstructure)
plot(gersztenkorn, 'Gersztenkorn & Marfurt (1999)')


complex_gersztenkorn = moving_window(SI_cube, (3,3,9), complex_eigenstructure)
plot(complex_gersztenkorn, 'Including Analytic Trace\nGersztenkorn & Marfurt (1999)')