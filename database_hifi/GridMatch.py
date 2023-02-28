"""
This is a full port of the libgridmatch library (by D. Shine, LMSAL) to python.
The routines were originally written as a C/DLM module for IDL.

In this implementation, some of them have been partly modified
in order to allow multiple threads in the processing.
The calculations have been accelerated using Numba.

Coded by J. de la Cruz Rodriguez (ISP-SU 2022)
"""

import numpy as np
import numba

# ****************************************************************

fastmath = True

# ****************************************************************

@numba.jit(nopython=True,fastmath=True)
def _CS(val, limit = 1.e-100):
    sign = 1.0
    if(val < 0.0):
        sign = -1.0

    if(abs(val) < limit ):
        val = limit*sign

    return val
    
    
# ****************************************************************

@numba.jit(nopython=True,fastmath=fastmath, parallel=True)
def bilint_fast2D(d, yy, xx, nthreads=4):
    """
    Simple fast bilinear interpolation
    Input:
        d: 2D numpy array with the input data with [ny, nx] elements
       yy: 2D array with the y-locations of the interpolated values. 
           Must be scaled to [ny]
       xx: 2D array with the x-locations of the interpolated values. 
           Must be scaled to [nx]

    nthreads: Number of threads to use in the interpolation (default 4)

    Dependencies: Numba, Numpy
    
    Coded by J. de la Cruz Rodriguez (ISP-SU, 2022)
    """
    if(d.ndim != 2):
        print("[error] bilint_fast2D: image array must be 2D, returning")
        return np.zeros((2,2))

    # Set numbra threads
    numba.set_num_threads(max(int(nthreads),1))

    ny, nx   = d.shape
    ny1, nx1 = xx.shape

    d1  = d.reshape(d.size)
    xx1 = xx.reshape(xx.size)
    yy1 = yy.reshape(yy.size)

    ntot = nx1*ny1
    res  = np.zeros(ntot, dtype='float64')

    nxm1 = float(nx-1)
    nym1 = float(ny-1)
    
    for jj in numba.prange(ntot):

        # Check limits
        y = min(max(0,yy1[jj]),nym1)
        x = min(max(0,xx1[jj]),nxm1)

        # bracket indexes in the data array
        ix = min(int(x), nx-2)
        iy = min(int(y), ny-2)

        # Interpolation weights
        dx  = x - ix
        dx1 = 1.0 - dx 
        dy  = y - iy
        dy1 = 1.0 - dy

        # Weighted sum
        res[jj] = d1[iy*nx+ix]*dx1*dy1 + d1[iy*nx+ix+1]*dx*dy1 + \
                  d1[(iy+1)*nx+ix]*dx1*dy + d1[(iy+1)*nx+ix+1]*dx*dy

    
    return res.reshape((ny1,nx1))

# ****************************************************************

@numba.jit(nopython=True,fastmath=fastmath)
def Congrid(img, ny1, nx1, nthreads=4):
    """
    Congrid interpolates a 2D numpy array with dimensions [ny, nx]
    to dimensions [ny1,nx1] using bilinear interpolation

    Coded by J. de la Cruz Rodriguez (ISP-SU 2022)
    """
    ny, nx = img.shape
    xx = np.outer(np.ones(ny1), (np.arange(nx1)/(nx1-1.0) * (nx-1.0)))
    yy = np.outer((np.arange(ny1)/(ny1-1.0)) * (ny-1.0), np.ones(nx1))

    return bilint_fast2D(img, yy, xx, nthreads=nthreads)
    

# ****************************************************************

@numba.jit(nopython=True,fastmath=fastmath, parallel=True)
def Stretch(img, grid, nthreads=4):
    """
    Stretch applies a distortion grid to a 2D image
    Input:
        img: 2D numpy array with dimensions [ny,nx]
       grid: 3D numpy array with dimensions [ny1,nx1,2] containing the 
             distorsion grid in the x/y axes.

    Optional:
         nthreads: Integer, number of threads
    
    
    Ported to Python by J. de la Cruz Rodriguez (ISP-SU, 2022), 
    originally written by D. Shine (LMSAL, 1994)
    """
    if(img.ndim != 2):
        print("[error] Stretch: img must be a 2D array, exiting")
        return img

    if(grid.ndim != 3):
        print("[error] Stretch: grid must be a 3D array, exiting")
        return img

    # Set numbra threads
    numba.set_num_threads(max(int(nthreads),1))
    
    ny, nx      = img.shape
    nyg,nxg,dum = grid.shape
    
    res = np.zeros((ny, nx), dtype='float64')
    n = nx*1
    m = ny*1

    nxgm = int(nxg - 1)
    nygm = int(nyg - 1)
    nx1 = nx-1
    ny1 = ny-1

    # get steps in x-axis
    xd =  float(n)/nxg
    xinc = 1.0/xd
    xs = xinc + (xd - 1.0)/(2.0*xd)
    
    # get steps in y-axis
    yd = float(m) / nyg
    yinc = 1.0/yd
    y0 = yinc + (yd - 1.0)/(2.0*yd)

    for iy in numba.prange(m):

        y = y0 + iy * yinc
        x = xs*1
        jy = int(y*1)

        dy = float(y) - jy
        dy1 = 1.0 - dy

        if(jy < 1):
            j1 = 0
            j2 = 0
        elif(jy >= nyg):
            j1 = nygm*1
            j2 = nygm*1
        else:
            j1 = jy-1
            j2 = j1+1

        for ix in range(n):
            jx = int(x*1)
            dx = float(x) - jx
            dx1 = 1.0 - dx
            
            if(jx < 1):
                i1 = 0
                i2 = 0
            elif(jx>=nxg):
                i1 = nxgm*1
                i2 = nxgm*1
            else:
                i1 = jx - 1
                i2 = i1 + 1

            # Bilinear interpolation of the coordinates for a pixel in the image
            w1 = dy1*dx1
            w2 = dy1*dx
            w3 = dy*dx1
            w4 = dy*dx
            
            xl = w1*grid[j1,i1,0] + w2*grid[j1,i2,0] + w3*grid[j2,i1,0] + w4*grid[j2,i2,0] + ix
            yl = w1*grid[j1,i1,1] + w2*grid[j1,i2,1] + w3*grid[j2,i1,1] + w4*grid[j2,i2,1] + iy
            
            xl = min(max(xl,0.0), float(nx1))
            yl = min(max(yl,0.0), float(ny1))
            
            # Now bilinear interpolation of the actual image
            # bracket indexes
            i3 = int(min(max(int(xl),0),nx1-1))
            j3 = int(min(max(int(yl),0),ny1-1))
            
            dxp  = xl - i3
            dxp1 = 1.0 - dxp 
            dyp  = yl - j3
            dyp1 = 1.0  - dyp

            res[iy,ix] = img[j3,i3]*(dxp1*dyp1) + img[j3,i3+1]*(dxp*dyp1) + \
                         img[j3+1,i3]*(dxp1*dyp) + img[j3+1,i3+1]*(dxp*dyp)


            x += xinc
    return res

# ****************************************************************

@numba.jit(nopython=True,fastmath=fastmath, parallel=True)
def StretchMatrix(ny, nx, grid, nthreads=4, bare=False):
    """
    StretchMatrix calculates the fullsize distortion grid from the compact form
    Input:
        ny,nx: Matrix dimensions [ny,nx]
       grid: 3D numpy array with dimensions [ny1,nx1,2] containing the 
             distorsion grid in the x/y axes.

    Optional:
         nthreads: Integer, number of threads
    
    
    Ported to Python by J. de la Cruz Rodriguez (ISP-SU, 2022), 
    originally written by D. Shine (LMSAL, 1994)
    """
    if(grid.ndim != 3):
        print("[error] Stretch: grid must be a 3D array, exiting")
        return img
    
    # Set numbra threads
    numba.set_num_threads(max(int(nthreads),1))

    res = np.zeros((2,ny,nx), dtype='float64')
    nyg,nxg,dum = grid.shape

    n = nx*1
    m = ny*1

    nxgm = int(nxg - 1)
    nygm = int(nyg - 1)
    nx1 = nx-1
    ny1 = ny-1

    # get steps in x-axis
    xd =  float(n)/nxg
    xinc = 1.0/xd
    xs = xinc + (xd - 1.0)/(2.0*xd)
    
    # get steps in y-axis
    yd = float(m) / nyg
    yinc = 1.0/yd
    y0 = yinc + (yd - 1.0)/(2.0*yd)

    for iy in numba.prange(m):

        y = y0 + iy * yinc
        x = xs*1
        jy = int(y*1)

        dy = float(y) - jy
        dy1 = 1.0 - dy

        if(jy < 1):
            j1 = 0
            j2 = 0
        elif(jy >= nyg):
            j1 = nygm*1
            j2 = nygm*1
        else:
            j1 = jy-1
            j2 = j1+1

        for ix in range(n):
            jx = int(x*1)
            dx = float(x) - jx
            dx1 = 1.0 - dx
            
            if(jx < 1):
                i1 = 0
                i2 = 0
            elif(jx>=nxg):
                i1 = nxgm*1
                i2 = nxgm*1
            else:
                i1 = jx - 1
                i2 = i1 + 1

            # Bilinear interpolation of the coordinates for a pixel in the image
            w1 = dy1*dx1
            w2 = dy1*dx
            w3 = dy*dx1
            w4 = dy*dx
            
            
            xl = (w1*grid[j1,i1,0] + w2*grid[j1,i2,0] + w3*grid[j2,i1,0] + w4*grid[j2,i2,0]) + ix
            yl = (w1*grid[j1,i1,1] + w2*grid[j1,i2,1] + w3*grid[j2,i1,1] + w4*grid[j2,i2,1]) + iy
            
            xl = min(max(xl,0.0), float(nx1))
            yl = min(max(yl,0.0), float(ny1))

            if(bare):
                xl -= ix
                yl -= iy
            
            res[0,iy,ix] = xl
            res[1,iy,ix] = yl
            
            x += xinc

    return res

# ****************************************************************

@numba.jit(nopython=True,fastmath=fastmath)
def _GaussianKernels(ny, nx, gwid, nxa, nxb, nya, nyb):
    
    gwx = np.zeros(nx, dtype='float64')
    gwy = np.zeros(ny, dtype='float64')
    wid = float(gwid)*0.6005612
    
    if (wid > 0.0):
        xcen = float((nxa + nxb)//2)
        ycen = float((nya + nyb)//2)

        for i in range(nxa,nxb+1):
            xq = float((i - xcen) / wid)
            gwx[i] = np.exp(-(xq*xq))

        for i in range(nya,nyb+1):
            xq = float((i - ycen) / wid)
            gwy[i] = np.exp(-(xq*xq))

    else:
        gwx[nxa:nxb+1] = 1.0
        gwy[nya:nyb+1] = 1.0

    return gwx, gwy

# ****************************************************************

@numba.jit(nopython=True,fastmath=fastmath)
def _Averag(p1, nxa, nxb, nya, nyb, nxs, nys, idx, idy, gx, gy):
    nxc = 0; nyc = 0; nxd = 0; nyd = 0

    p = p1.reshape(p1.size)
    #gx = gx1.reshape(gx1.size)
    #gy = gy1.reshape(gy1.size)
    
    if((nxa+idx)<0):
        nxc = -int(idx)
    else:
        nxc = int(nxa)
        
    if((nya+idy)<0):
        nyc = -int(idy)
    else:
        nyc = int(nya)

    if((nxb+idx) > nxs):
        nxd = int(nxs - idx)
    else:
        nxd = int(nxb)

    if((nyb+idy) > nys):
        nyd = int(nys - idy)
    else:
        nyd = int(nyb)

    sumgx = 0.0
    for ii in range(nxc,nxd):
        sumgx += gx[ii]


        
    suma=0.0
    sumg=0.0
    for j in range(nyc,nyd):
        sumx = 0.0
        jj = int(idx+nxs*(j+idy))
        for i in range(nxc,nxd):
            sumx += gx[i] * p[i+jj]
        suma += gy[j]*sumx
        sumg += gy[j]*sumgx

    suma /= _CS(sumg)
    
    return suma
        
# ****************************************************************

@numba.jit(nopython=True,fastmath=fastmath)
def _Unbias(m1, m2, nxa, nxb, nya, nyb, nxs, nys, gy, gx, idely, idelx):
    
    av1 = _Averag(m1, nxa, nxb, nya, nyb, nxs, nys,     0,     0, gx, gy)
    t0  = _Averag(m2, nxa, nxb, nya, nyb, nxs, nys, idelx, idely, gx, gy)
    t1  = _Averag(m2, nxa, nxb, nya, nyb, nxs, nys, idelx + 1, idely, gx, gy)
    t2  = _Averag(m2, nxa, nxb, nya, nyb, nxs, nys, idelx - 1, idely, gx, gy)
    t3  = _Averag(m2, nxa, nxb, nya, nyb, nxs, nys, idelx, idely + 1, gx, gy)
    t4  = _Averag(m2, nxa, nxb, nya, nyb, nxs, nys, idelx, idely - 1, gx, gy)
    t5  = _Averag(m2, nxa, nxb, nya, nyb, nxs, nys, idelx + 1, idely + 1, gx, gy)
    
    av2 = t0*1
    cx  = 0.5*(t1 - t2)
    cy  = 0.5*(t3 - t4)
    cxx = 0.5*(t1 - 2*t0 + t2)
    cyy = 0.5*(t3 - 2*t0 + t4)
    cxy = t5 + t0 - t1 - t3
    
    return av1, av2, cx, cy, cxx, cxy, cyy

# ****************************************************************

@numba.jit(nopython=True,fastmath=fastmath)
def _GetMin(p):
    f11 = p[0]; f21 = p[1]; f31 = p[2]
    f12 = p[3]; f22 = p[4]; f32 = p[5]
    f13 = p[6]; f23 = p[7]; f33 = p[8]
    
    fx = 0.5 * ( f32 - f12 )
    fy = 0.5 * ( f23 - f21 )
    t  = 2.0 * ( f22 )
    fxx =  f32 + f12 - t
    fyy = f23 + f21 - t
     
    if (f33 < f11):
        if (f33 < f31):
            if (f33 < f13):
                fxy = f33+f22-f32-f23
            else:
                fxy = f23+f12-f22-f13
        else:
            if (f31 < f13):
                fxy = f32+f21-f31-f22
            else:
                fxy = f23+f12-f22-f13
    else: 
        if (f11 < f31):
            if (f11 < f13):
                fxy = f22+f11-f21-f12
            else:
                fxy = f23+f12-f22-f13
        else :
            if (f31 < f13):
                fxy = f32+f21-f31-f22
            else:
                fxy = f23+f12-f22-f13

    
    t = -1./_CS(fxx *fyy - fxy *fxy)
    x0 = t * (fx * fyy - fy * fxy)
    y0 = t * (fy * fxx - fx * fxy)

    if ((abs(x0) >= 0.75) or (abs(y0) >= 0.75)):
        x0 = -fx/_CS(fxx)
        y0 = -fy/_CS(fyy)
    
    return x0, y0
    
# ****************************************************************

@numba.jit(nopython=True,fastmath=fastmath)
def _Resid(m1i, m2i, idx, idy, nxa, nxb, nya, nyb, nxs, nys, ndmx, gx, gy, bs):

    m1 = m1i.reshape(m1i.size)
    m2 = m2i.reshape(m2i.size)
    
    nxc = int(nxa)*1
    if ((nxc + idx) < 0):
        nxc = int(-idx)
    nyc = int(nya)*1
    if ((nyc + idy) < 0):
        nyc = int(- idy)
    nxd = int(nxb)*1
    if ((nxd + idx) >= nxs):
        nxd = int(nxs - idx - 1)
    nyd = int(nyb)*1
    if ((nyd + idy) >= nys):
        nyd = int(nys - idy - 1)

    suma = 0.0
    sumg = 0.0
    
    nx = int(nxd - nxc + 1)
    p2 = nyc*1
    ps = nxc*1

    j = int(nyd -nyc + 1)
    if ((j <= 0) or ((nxd - nxc + 1) <= 0)):
        return -1
    while(j):
        i  = nx*1
        p1 = ps*1
        gyp2 = gy[p2]*1
        while (i):
            sumg += gx[p1] * gyp2
            p1 += 1
            i -= 1
        p2 += 1
        j -= 1

    o1 = int(nyc*nxs + nxc)
    o2 = int((nyc + idy)*nxs + nxc + idx)
    ny = int(nxs - nx)
    p2 = nyc*1
    j = int(nyd - nyc + 1)
    ndmx2 = float(ndmx)**2

    while(j):
        i = nx*1
        p1 = ps*1
        sumx = 0.0
        while(i):
            t = float(m1[o1]) - float(m2[o2])
            o1 += 1
            o2 += 1
            t = min((t + bs)**2, ndmx2)
            sumx += gx[p1]*t
            p1+=1
            i-=1
        suma+= gy[p2]*sumx
        p2 +=1
        o1 += ny
        o2 += ny
        j-=1


    return suma / _CS(sumg)

# ****************************************************************

@numba.jit(nopython=True,fastmath=fastmath)
def _Match_1(p1, p2, nxa, nxb, nya, nyb, ny, nx, gwy, gwx, stretch_clip):
    itmax = int(40)
    ndmx  = int(1000)
    
    done = np.zeros(9, dtype='int32')
    res = np.zeros(9, dtype='float64')
    buf = np.zeros(9, dtype='float64')

    idelx = 0 #round(xoffset)*1
    idely = 0 #round(yoffset)*1
    
    av1,av2,cx,cy,cxx,cxy,cyy = _Unbias(p1, p2, nxa, nxb, nya, nyb, nx, ny, gwy, gwx, idely, idelx)

    badflag = 0; badmatch = 0

    it = itmax*1
    while(it):
        it -= 1
        for k in range(9):
            if(done[k] == 0):
                i = int(idelx + (k % 3) - 1)
                j = int(idely + (k // 3) - 1)
                avdif = av2 +  i*cx + j*cy + i*i*cxx + i*j*cxy + j*j*cyy - av1

                res[k] = _Resid(p1, p2, i, j, nxa, nxb, nya, nyb, \
                                nx, ny, ndmx, gwx, gwy, avdif)
        i = 0
        t = res[0]*1

        for k in range(1,9):
            if(res[k] < t):
                t = res[k]*1
                i = k*1
                
        if(t < 0):
            badflag = 1
            break
        
        idelx += int((i % 3) - 1)
        idely += int((i // 3) - 1)

        if((abs(idelx) > stretch_clip) or (abs(idely) > stretch_clip)):
            badflag += 1
            break

        if(i == 4):
            break; # Done, it is in the center
        
        di = int((i%3) - 1)
        dj = int((i//3) - 1)
        dd = int(dj * 3 + di)
        
        for k in range(9):
            In = int(k%3 + di)
            jn = int(k//3 + dj)
            if((In >= 0) and (jn >= 0) and (In < 3)and(jn < 3)):
                done[k] = 1
                buf[k] = res[k+dd]
            else:
                done[k] = 0
        
        res[:] = buf # put back in res array

    if(it <= 0):
        badflag += 1

    if(badflag):
        badmatch += 1
        return 0.0, 0.0

    
    t1,t2 = _GetMin(res)

    return idelx+t1, idely+t2
    
# ****************************************************************

@numba.jit(nopython=True,fastmath=fastmath)
def  _GridMatchOne(ii,gwid,m1,m2,stretch_clip,dy2,dx2,nyg,nxg,ny,nx,gy,gx):
    
    i1 = max(0,int(gx[ii]-dx2))
    i2 = min(nx,int(gx[ii]+dx2)) - 1
    j1 = max(0,int(gy[ii]-dy2))
    j2 = min(ny,int(gy[ii]+dy2)) - 1
  
    # Get Gaussian kernels
    gwx, gwy = _GaussianKernels(ny,nx,gwid,i1,i2,j1,j2)
    
    # Match patch
    return _Match_1(m1, m2, i1, i2, j1, j2, ny, nx, gwy, gwx, stretch_clip)
    
    

# ****************************************************************

@numba.jit(nopython=True,fastmath=fastmath,parallel=True)
def GridMatch(m1, m2, gy, gx, dy, dx, gwid, stretch_clip, nthreads=4): 

    # Set numbra threads
    numba.set_num_threads(max(int(nthreads),1))
    
    if(stretch_clip < 2):
        stretch_clip = 2

        
    stretch_clip -= 1

    ny, nx = m1.shape
    nyg, nxg = gx.shape
    
    nc = nxg*nyg
    dx2 = dx//2
    dy2 = dy//2
    
    res = np.zeros((nyg,nxg,2), dtype='float64')
    res1 = res.reshape(res.size)
    gx1 = gx.reshape(gx.size)
    gy1 = gy.reshape(gy.size)
    
    for ii in numba.prange(nc):
        res1[2*ii], res1[2*ii+1] = _GridMatchOne(ii,gwid,m1,m2,stretch_clip,dy2,dx2,nyg,nxg,ny,nx,gy1,gx1)
        
        
    return res
        
# ****************************************************************

#@numba.jit(nopython=True,fastmath=fastmath)
def DSGridNest(m1, m2, tiles, clips, nthreads = 4):
    
    # Set numbra threads
    numba.set_num_threads(max(int(nthreads),1))

    vg = np.int32(tiles)
    clip = np.int32(clips)
    
    ny, nx = m1.shape
    nest = len(vg)

    displ = np.zeros((2,2,2), dtype='float64')
    
    if(len(clips) != nest):
        print("[error] DSGridNest: the tiles and clips arrays must have the same number of elements, exiting!")
        return displ

    nprev = -1
    
    for k in range(nest):
        n = vg[k]*1

        stretch_clip = clip[k]*1
        ngw = int((2.0*nx)/n)
        nw  = int(1.25*ngw)
        
        if(nx > ny):
            nxg = n
            nyg = round(float(n)*ny/nx)
        else:
            nyg = n
            nxg = round(float(n)*nx/ny)
   
        wx = float(nx)/nxg
        wy = float(ny)/nyg

        if(k == 0):
            displprev = np.zeros((nyg,nxg,2))
         
        # Grid of cells to perform the cross-corralation
        gx = np.outer(np.ones(nyg, dtype='float64'), np.arange(nxg))
        gx = (gx*wx+wx*0.5-1.0).astype(np.int32)
        
        gy = np.outer(np.arange(nyg), np.ones(nxg, dtype='float64'))
        gy = (gy*wy+wy*0.5-1.0).astype(np.int32)

        dx = nw*1
        dy = nw*1
        gwid = float(ngw*1)

        # Get initial correction in the coarser grid
        if(k == 0):
            displ = GridMatch(m1, m2, gy, gx, dy, dx, gwid, stretch_clip)
        else:
            # Interpolate the old grid to the new size
            if(n != nprev):
                disx = np.ascontiguousarray(displprev[:,:,0])
                disy = np.ascontiguousarray(displprev[:,:,1])
                prev = np.zeros((nyg, nxg, 2), dtype='float64')

                prev[:,:,0] = Congrid(disx, nyg, nxg, nthreads=nthreads)
                prev[:,:,1] = Congrid(disy, nyg, nxg, nthreads=nthreads)
                
            else:
                prev = np.copy(displprev)

            # Remove the effect of the previous grids and search for
            # a correction in the finer one
            m3 = Stretch(m2, prev, nthreads=nthreads)
            displnew = GridMatch(m1, m3, gy, gx, dy, dx, gwid, stretch_clip)
            displ = prev+displnew
            
            
        nprev = n*1
        displprev = np.copy(displ)

    return displ

# ****************************************************************

def DSGridNestBurst(cub, tiles, clips, nthreads=4, apply_correction = False):
    """
    DSGridNestBurst takes a burst of images where the observed object is the same,
    and removes the distorsion using the provided tile and clip values.
    
    This routine returns the correction grid with dimensions [nt,nyg,nxg,2].

    Input:
           cub: a 3D numpy array with dimensions [nt, ny, nx]
         tiles: a tuple/list/1D array with the number of tiles to be used in each nested search 
                e.g., tiles=[12,16,32,48,64]. The nested search is done from less tiles to more tiles.

         clips: a tuple/list/1D array with the maximum shift allowed per tile for each level of the nested
                nested search. e.g, clips=[10,6,4,3,2]. Tiles and clips should have the same number of elements.
    Optional:
      nthreads: Number of threads to be used in some of the multithreaded parts of the code (default=4).
      apply_correction: if True, it will copy the input data, correct it and also return the corrected cube.
                        if False: return cor
                        else:     return cor, cub_corrected
    
    Coded by J. de la Cruz Rodriguez (ISP-SU 2022)
    """
    
    nt, ny, nx = cub.shape

    # Get RMS contrast
    rms = np.zeros((nt), dtype='float32')
    for ii in range(nt):
        rms = np.nanstd(cub[ii]) / np.nanmean(cub[ii])

    # Choose the image with the highest RMS contrast as reference
    refidx = np.argmax(rms)
    ref = np.float32(cub[refidx])

    # Destretch all images to reference
    defined = False
    
    for ii in range(nt):
        
        if(ii == refidx): # skip refence image
            continue
        else:
            # Calculate the correction for each image, relative to the reference
            cor = DSGridNest(ref, np.float32(cub[ii]), tiles, clips, nthreads=nthreads)

            # if not defined, prepare the output array
            if(not defined):
                nyg, nxg, dum = cor.shape
                res = np.zeros((nt,nyg,nxg,2), dtype='float64')
                defined = True

            # copy correction to the output array
            res[ii] = cor
    
    # Now subtract the mean, the assumption is that random seeing motions will
    # average to zero, which is not true for a small sample.
    cormean = res.mean(axis=0)
    for ii in range(nt):
        res[ii] -= cormean


    # If requested, apply distorsions to a copy of the data
    if(apply_correction):
        cub1 = np.zeros(cub.shape, dtype=cub.dtype)
        for ii in range(nt):
            cub1[ii] = Stretch(cub[ii], res[ii], nthreads=int(nthreads))
        
        return res, cub1
    else:
        return res
    
# ****************************************************************

@numba.jit(nopython=True,fastmath=fastmath)
def _Smooth(var, win):
    win = int(win)
    win2 = win // 2

    N = var.size

    res = np.zeros(N, dtype='float64')

    for ii in range(N):
        j0 = max(0, ii-win2)
        j1 = min(ii+win2,N)

        suma = 0.0
        for jj in range(j0,j1):
            suma += var[jj]

        res[ii] = suma / max((j1-j0), 1)

    return res
    
    
# ****************************************************************

def _GridPrep(delta, tstep):
    
    nt, ngy, ngx, dum = delta.shape
    
    delx = delta[:,:,:,0]
    dely = delta[:,:,:,1]
    
    xvec = np.arange(nt)

    for j in range(ngy):
        for i in range(ngx):
            xq = np.cumsum(delx[:,j,i])
            xq1 = xq[1]*1
            cf = np.polyfit(xvec, xq, 1)
            yfit = xvec*cf[1] + cf[0]
            xq -= yfit
            xq -= _Smooth(xq,int(tstep))
            xq +=  xq1 - xq[1]
            delx[:,j,i] = xq

            yq = np.cumsum(dely[:,j,i])
            yq1 = yq[1]*1
            cf = np.polyfit(xvec, yq, 1)
            yfit = xvec*cf[1] + cf[0]
            yq -= yfit
            yq -= _Smooth(yq,int(tstep))
            yq +=  yq1 - yq[1]
            dely[:,j,i] = yq

    deltap = np.zeros(delta.shape, dtype='float64')
    deltap[:,:,:,0] = delx
    deltap[:,:,:,1] = dely
    
    deltap[0] = 0.0

    return deltap

# ****************************************************************

def destretch_tseries(cub, platescale, tiles, clips, tstep, nthreads=4, apply_correction = False):
    """
    destretch_tseries computes the distorsion matrices of a time series of images.
    The goal is to remove residual rubber-sheet motions. Compared to DSGridNestBurs, the target is allowed to slowly evolve from one image to the next.

    Input:
                    cub: 3D numpy array containing a time series of images [nt,ny,nx]
             platescale: pixels per arcsec, used to restrict the maximum shift.
                  tiles: a tuple/list/1D array with the number of tiles to be used in each nested search 
                         e.g., tiles=[12,16,32,48,64]. The nested search is done from less tiles to more tiles.
                  clips: a tuple/list/1D array with the maximum shift allowed per tile for each level of the nested
                         nested search. e.g, clips=[10,6,4,3,2]. Tiles and clips should have the same number of elements.
                  tstep: number of frames to perform the average. Typical values are 3 minutes (aka, the number of
                         frames that add up to 3 minutes). Just make sure you make it larger than 1, so there is
                         some form of average.
    Optional:
               nthreads: Number of threads to be used in some of the multithreaded parts of the code (default=4).
       apply_correction: if True, it will copy the input data, correct it and also return the corrected cube.
                         if False: return cor
                         else:     return cor, cub_corrected

    Ported from IDL (destretch_tseries.pro)
    """
    
    nt, ny, nx = cub.shape
    tile  = np.int32(tiles)
    clip  = np.int32(clips)

    maxg = tile.max()
    maxstr = 2 * platescale       # 2" maximum allowed displacement for any one grid cell.
    refim = np.copy(cub[0])
    
    if(nx > ny):
        maxgx = maxg*1
        maxgy = round(ny/nx*maxgx)
    else:
        maxgy = maxg*1
        maxgx = round(nx/ny*maxgy)

    delta = np.zeros((nt,maxgy,maxgx,2), dtype='float64')

    for ii in range(1,nt):

        # Calculate distorsion grid
        im = np.copy(cub[ii])
        dq = DSGridNest(refim, im, tile, clip, nthreads=nthreads)

        # Remove extreme shifts
        badg = np.where(dq > maxstr)
        dq[badg] = 0.0

        delta[ii] = dq
        refim = np.copy(im)

    idx = np.where(np.isinf(delta))
    delta[idx] = 0.0


    # Detrend and unsharp mask the displacements
    #delta = _GridPrep(delta, tstep)

    # correct cube?
    if(apply_correction):
        res = np.zeros(cub.shape, dtype=cub.dtype)
        for ii in range(nt):
            res[ii] = Stretch(cub[ii], delta[ii], nthreads=nthreads)
        return delta, res
    
    return delta
        
