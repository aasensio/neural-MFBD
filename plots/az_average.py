import numpy as np
from convolvend import convolvend as convolve


def power_spectrum(*args,**kwargs):
    """
    Thin wrapper of PSD2.  Returns the 1D power spectrum in stead of the 2D Power Spectral Density
    """
    kwargs['oned']=True
    return PSD2(*args,**kwargs)

def PSD2(image, image2=None, oned=False, 
        fft_pad=False, real=False, imag=False,
        binsize=1.0, radbins=1, azbins=1, radial=False, hanning=False, 
        wavnum_scale=False, twopi_scale=False, **kwargs):
    """
    Two-dimensional Power Spectral Density.
    NAN values are treated as zero.

    image2 - can specify a second image if you want to see the cross-power-spectrum instead of the 
        power spectrum.
    oned - return radial profile of 2D PSD (i.e. mean power as a function of spatial frequency)
           freq,zz = PSD2(image); plot(freq,zz) is a power spectrum
    fft_pad - Add zeros to the edge of the image before FFTing for a speed
        boost?  (the edge padding will be removed afterwards)
    real - Only compute the real part of the PSD (Default is absolute value)
    imag - Only compute the complex part of the PSD (Default is absolute value)
    hanning - Multiply the image to be PSD'd by a 2D Hanning window before performing the FTs.  
        Reduces edge effects.  This idea courtesy Paul Ricchiazzia (May 1993), author of the
        IDL astrolib psd.pro
    wavnum_scale - multiply the FFT^2 by the wavenumber when computing the PSD?
    twopi_scale - multiply the FFT^2 by 2pi?
    azbins - Number of azimuthal (angular) bins to include.  Default is 1, or
        all 360 degrees.  If azbins>1, the data will be split into [azbins]
        equally sized pie pieces.  Azbins can also be a numpy array.  See
        AG_image_tools.azimuthalAverageBins for details
        
    
    radial - An option to return the *azimuthal* power spectrum (i.e., the spectral power as a function 
        of angle).  Not commonly used.
    radbins - number of radial bins (you can compute the azimuthal power spectrum in different annuli)
    """
    
    # prevent modification of input image (i.e., the next two lines of active code)
    image = image.copy()

    # remove NANs (but not inf's)
    image[image!=image] = 0

    if hanning:
        image = hanning2d(*image.shape) * image

    if image2 is None:
        image2 = image
    else:
        image2 = image2.copy()
        image2[image2!=image2] = 0
        if hanning:
            image2 = hanning2d(*image2.shape) * image2

    if real:
        psd2 = np.real( correlate2d(image,image2,return_fft=True,fft_pad=fft_pad) ) 
    elif imag:
        psd2 = np.imag( correlate2d(image,image2,return_fft=True,fft_pad=fft_pad) ) 
    else: # default is absolute value
        psd2 = np.abs( correlate2d(image,image2,return_fft=True,fft_pad=fft_pad) ) 
    # normalization is approximately (np.abs(image).sum()*np.abs(image2).sum())

    if wavnum_scale:
        wx = np.concatenate([ np.arange(image.shape[0]/2,dtype='float') , image.shape[0]/2 - np.arange(image.shape[0]/2,dtype='float') -1 ]) / (image.shape[0]/2.)
        wy = np.concatenate([ np.arange(image.shape[1]/2,dtype='float') , image.shape[1]/2 - np.arange(image.shape[1]/2,dtype='float') -1 ]) / (image.shape[1]/2.)
        wx/=wx.max()
        wy/=wy.max()
        wavnum = np.sqrt( np.outer(wx,np.ones(wx.shape))**2 + np.outer(np.ones(wy.shape),wx)**2 )
        psd2 *= wavnum

    if twopi_scale:
        psd2 *= np.pi * 2

    if radial:
        azbins,az,zz = radialAverageBins(psd2,radbins=radbins, interpnan=True, binsize=binsize, **kwargs)
        if len(zz) == 1:
            return az,zz[0]
        else:
            return az,zz

    if oned:
        return pspec(psd2, azbins=azbins, binsize=binsize, **kwargs)

    # else...
    return psd2

def pspec(psd2, return_index=True, wavenumber=False, return_stddev=False, azbins=1, binsize=1.0, view=False, **kwargs):
    """
    Create a Power Spectrum (radial profile of a PSD) from a Power Spectral Density image

    return_index - if true, the first return item will be the indexes
    wavenumber - if one dimensional and return_index set, will return a normalized wavenumber instead
    view - Plot the PSD (in logspace)?
    """
    #freq = 1 + np.arange( np.floor( np.sqrt((image.shape[0]/2)**2+(image.shape[1]/2)**2) ) ) 

    azbins,(freq,zz) = azimuthalAverageBins(psd2,azbins=azbins,interpnan=True, binsize=binsize, **kwargs)
    if len(zz) == 1: zz=zz[0]
    # the "Frequency" is the spatial frequency f = 1/x for the standard numpy fft, which follows the convention
    # A_k =  \sum_{m=0}^{n-1} a_m \exp\left\{-2\pi i{mk \over n}\right\}
    # or 
    # F_f = Sum( a_m e^(-2 pi i f x_m)  over the range m,m_max where a_m are the values of the pixels, x_m are the
    # indices of the pixels, and f is the spatial frequency
    freq = freq.astype('float')  # there was a +1.0 here before, presumably to deal with div-by-0, but that shouldn't happen and shouldn't have been "accounted for" anyway

    if return_index:
        if wavenumber:
            fftwavenum = (np.fft.fftfreq(zz.size*2)[:zz.size])
            return_vals = list((fftwavenum,zz))
            #return_vals = list((len(freq)/freq,zz))
        else:
            return_vals = list((freq/len(freq),zz))
    else:
        return_vals = list(zz)
    if return_stddev:
        azbinsS,(freqstd,zzstd) = azimuthalAverageBins(psd2,azbins=azbins,stddev=True,interpnan=True, binsize=binsize, **kwargs)
        return_vals.append(zzstd)
    
    if view and pyplotOK:
        pyplot.loglog(freq,zz)
        pyplot.xlabel("Spatial Frequency")
        pyplot.ylabel("Spectral Power")

    return return_vals


def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False, 
        binsize=0.5, weights=None, steps=False, interpnan=False, left=None, right=None,
        mask=None ):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels).
    stddev - if specified, return the azimuthal standard deviation instead of the average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and radial
        profile so you can plot a step-form radial profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values
    mask - can supply a mask (boolean array same size as image with True for OK and False for not)
        to average over only select data.

    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    if mask is None:
        mask = np.ones(image.shape,dtype='bool')
    # obsolete elif len(mask.shape) > 1:
    # obsolete     mask = mask.ravel()

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)  
    nbins = int(np.round(r.max() / binsize)+1)
    maxbin = nbins * binsize
    bins = np.linspace(0,maxbin,nbins+1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:]+bins[:-1])/2.0

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    #nr = np.bincount(whichbin)[1:]
    nr = np.histogram(r,bins)[0]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or xrange(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape
    if stddev:
        # Find out which radial bin each point in the map belongs to
        whichbin = np.digitize(r.flat,bins)
        # This method is still very slow; is there a trick to do this with histograms? 
        radial_prof = np.array([image.flat[mask.flat*(whichbin==b)].std() for b in xrange(1,nbins+1)])
    else: 
        radial_prof = np.histogram(r, bins, weights=(image*weights*mask))[0] / np.histogram(r, bins, weights=(mask*weights))[0]

    if interpnan:
        radial_prof = np.interp(bin_centers,bin_centers[radial_prof==radial_prof],radial_prof[radial_prof==radial_prof],left=left,right=right)

    if steps:
        xarr = np.array(zip(bins[:-1],bins[1:])).ravel() 
        yarr = np.array(zip(radial_prof,radial_prof)).ravel() 
        return xarr,yarr
    elif returnradii: 
        return bin_centers,radial_prof
    elif return_nr:
        return nr,bin_centers,radial_prof
    else:
        return radial_prof

def azimuthalAverageBins(image,azbins,symmetric=None, center=None, **kwargs):
    """ Compute the azimuthal average over a limited range of angles 
    kwargs are passed to azimuthalAverage """
    y, x = np.indices(image.shape)
    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    r = np.hypot(x - center[0], y - center[1])
    theta = np.arctan2(x - center[0], y - center[1])
    theta[theta < 0] += 2*np.pi
    theta_deg = theta*180.0/np.pi

    if isinstance(azbins,np.ndarray):
        pass
    elif isinstance(azbins,int):
        if symmetric == 2:
            azbins = np.linspace(0,90,azbins)
            theta_deg = theta_deg % 90
        elif symmetric == 1:
            azbins = np.linspace(0,180,azbins)
            theta_deg = theta_deg % 180
        elif azbins == 1:
            return azbins,azimuthalAverage(image,center=center,returnradii=True,**kwargs)
        else:
            azbins = np.linspace(0,359.9999999999999,azbins)
    else:
        raise ValueError("azbins must be an ndarray or an integer")

    azavlist = []
    for blow,bhigh in zip(azbins[:-1],azbins[1:]):
        mask = (theta_deg > (blow % 360)) * (theta_deg < (bhigh % 360))
        rr,zz = azimuthalAverage(image,center=center,mask=mask,returnradii=True,**kwargs)
        azavlist.append(zz)

    return azbins,rr,azavlist

def radialAverage(image, center=None, stddev=False, returnAz=False, return_naz=False, 
        binsize=1.0, weights=None, steps=False, interpnan=False, left=None, right=None,
        mask=None, symmetric=None ):
    """
    Calculate the radially averaged azimuthal profile.
    (this code has not been optimized; it could be speed boosted by ~20x)

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels).
    stddev - if specified, return the radial standard deviation instead of the average
    returnAz - if specified, return (azimuthArray,azimuthal_profile)
    return_naz   - if specified, return number of pixels per azimuth *and* azimuth
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and azimuthal
        profile so you can plot a step-form azimuthal profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values
    mask - can supply a mask (boolean array same size as image with True for OK and False for not)
        to average over only select data.

    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])
    theta = np.arctan2(x - center[0], y - center[1])
    theta[theta < 0] += 2*np.pi
    theta_deg = theta*180.0/np.pi
    maxangle = 360

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    if mask is None:
        # mask is only used in a flat context
        mask = np.ones(image.shape,dtype='bool').ravel()
    elif len(mask.shape) > 1:
        mask = mask.ravel()

    # allow for symmetries
    if symmetric == 2:
        theta_deg = theta_deg % 90
        maxangle = 90
    elif symmetric == 1:
        theta_deg = theta_deg % 180
        maxangle = 180

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)  
    nbins = int(np.round(maxangle / binsize))
    maxbin = nbins * binsize
    bins = np.linspace(0,maxbin,nbins+1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:]+bins[:-1])/2.0

    # Find out which azimuthal bin each point in the map belongs to
    whichbin = np.digitize(theta_deg.flat,bins)

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    nr = np.bincount(whichbin)[1:]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or xrange(1,nbins+1) )
    # azimuthal_prof.shape = bin_centers.shape
    if stddev:
        azimuthal_prof = np.array([image.flat[mask*(whichbin==b)].std() for b in xrange(1,nbins+1)])
    else:
        azimuthal_prof = np.array([(image*weights).flat[mask*(whichbin==b)].sum() / weights.flat[mask*(whichbin==b)].sum() for b in xrange(1,nbins+1)])

    #import pdb; pdb.set_trace()

    if interpnan:
        azimuthal_prof = np.interp(bin_centers,
            bin_centers[azimuthal_prof==azimuthal_prof],
            azimuthal_prof[azimuthal_prof==azimuthal_prof],
            left=left,right=right)

    if steps:
        xarr = np.array(zip(bins[:-1],bins[1:])).ravel() 
        yarr = np.array(zip(azimuthal_prof,azimuthal_prof)).ravel() 
        return xarr,yarr
    elif returnAz: 
        return bin_centers,azimuthal_prof
    elif return_naz:
        return nr,bin_centers,azimuthal_prof
    else:
        return azimuthal_prof

def radialAverageBins(image,radbins, corners=True, center=None, **kwargs):
    """ Compute the radial average over a limited range of radii """
    y, x = np.indices(image.shape)
    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    r = np.hypot(x - center[0], y - center[1])

    if isinstance(radbins,np.ndarray):
        pass
    elif isinstance(radbins,int):
        if radbins == 1:
            return radbins,radialAverage(image,center=center,returnAz=True,**kwargs)
        elif corners:
            radbins = np.linspace(0,r.max(),radbins)
        else:
            radbins = np.linspace(0,np.max(np.abs(np.array([x-center[0],y-center[1]]))),radbins)
    else:
        raise ValueError("radbins must be an ndarray or an integer")

    radavlist = []
    for blow,bhigh in zip(radbins[:-1],radbins[1:]):
        mask = (r<bhigh)*(r>blow)
        az,zz = radialAverage(image,center=center,mask=mask,returnAz=True,**kwargs)
        radavlist.append(zz)

    return radbins,az,radavlist



def correlate2d(im1,im2, boundary='wrap', **kwargs):
    """
    Cross-correlation of two images of arbitrary size.  Returns an image
    cropped to the largest of each dimension of the input images
    Options
    -------
    return_fft - if true, return fft(im1)*fft(im2[::-1,::-1]), which is the power
        spectral density
    fftshift - if true, return the shifted psd so that the DC component is in
        the center of the image
    pad - Default on.  Zero-pad image to the nearest 2^n
    crop - Default on.  Return an image of the size of the largest input image.
        If the images are asymmetric in opposite directions, will return the largest 
        image in both directions.
    boundary: str, optional
        A flag indicating how to handle boundaries:
            * 'fill' : set values outside the array boundary to fill_value
                       (default)
            * 'wrap' : periodic boundary
    WARNING: Normalization may be arbitrary if you use the PSD
    """    
    return convolve(np.conjugate(im1), im2[::-1, ::-1], normalize_kernel=False,
            boundary=boundary, ignore_edge_zeros=False, **kwargs)
