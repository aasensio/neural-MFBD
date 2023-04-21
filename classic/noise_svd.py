import numpy as np

def optimalSVHT(matrix):
    """
    Returns the optimal threshold of the singular values of a matrix, i.e., it computes the optimal number
    of singular values to keep or the optimal rank of the matrix.
    Given a matrix Y=X+sigma*Z, with Y the observed matrix, X the matrix that we want to find, Z an matrix whose elements are iid Gaussian
    random numbers with zero mean and unit variance and sigma is the standard deviation of the noise, the matrix X can be recovered in the
    least squares sense using a truncated SVD (compute the SVD of the matrix, keep only the relevant singular values and reconstruct the
    matrix). The problem is estimate the number of such singular values to keep. This function uses the theory of Gavish & Donoho (2014) 
    that is valid for matrices with sizes larger than its rank, i.e., for low-rank matrices
    
    Args:
        matrix: matrix to estimate the rank
    
    Returns:
        thrKnownNoise: in case sigma is known, multiply this value with sigma and this gives the threshold
        thrUnknownNoise: returns the threshold directly
        noiseEstimation: returns the estimation of the noise
        wSVD: returns the singular values of the matrix
    """
    
    m, n = matrix.shape
    beta = 1.0 * m / n
    
    w = (8.0 * beta) / (beta + 1 + np.sqrt(beta**2 + 14 * beta +1))
    lambdaStar = np.sqrt(2.0 * (beta + 1) + w)
    
    omega = 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43
    uSVD, wSVD, vSVD = np.linalg.svd(matrix)
    medianSV = np.median(wSVD)
    
    thrKnownNoise = lambdaStar * np.sqrt(n)
    thrUnknownNoise = omega * medianSV
    
    muSqrt = lambdaStar / omega
    noiseEstimation = medianSV / (np.sqrt(n) * muSqrt)      
            
    return thrKnownNoise, thrUnknownNoise, noiseEstimation, wSVD

def optimalSVHT_cov(matrix):
    """
    Returns the optimal threshold of the singular values of a matrix, i.e., it computes the optimal number
    of singular values to keep or the optimal rank of the matrix.
    Given a matrix Y=X+sigma*Z, with Y the observed matrix, X the matrix that we want to find, Z an matrix whose elements are iid Gaussian
    random numbers with zero mean and unit variance and sigma is the standard deviation of the noise, the matrix X can be recovered in the
    least squares sense using a truncated SVD (compute the SVD of the matrix, keep only the relevant singular values and reconstruct the
    matrix). The problem is estimate the number of such singular values to keep. This function uses the theory of Gavish & Donoho (2014) 
    that is valid for matrices with sizes larger than its rank, i.e., for low-rank matrices
    
    Args:
        matrix: matrix to estimate the rank
    
    Returns:
        thrKnownNoise: in case sigma is known, multiply this value with sigma and this gives the threshold
        thrUnknownNoise: returns the threshold directly
        noiseEstimation: returns the estimation of the noise
        wSVD: returns the singular values of the matrix
    """
    
    m, n = matrix.shape
    beta = 1.0 * m / n
    
    w = (8.0 * beta) / (beta + 1 + np.sqrt(beta**2 + 14 * beta +1))
    lambdaStar = np.sqrt(2.0 * (beta + 1) + w)
    
    omega = 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43
    if (m > n):
        cov = matrix.T @ matrix
    else:
        cov = matrix @ matrix.T
    uSVD, wSVD, vSVD = np.linalg.svd(cov)
    medianSV = np.median(np.sqrt(wSVD))
    
    thrKnownNoise = lambdaStar * np.sqrt(n)
    thrUnknownNoise = omega * medianSV
    
    muSqrt = lambdaStar / omega
    noiseEstimation = medianSV / (np.sqrt(n) * muSqrt)      
            
    return thrKnownNoise, thrUnknownNoise, noiseEstimation, wSVD

def noise_estimation(x):
    """
    Returns the optimal threshold of the singular values of a matrix, i.e., it computes the optimal number
    of singular values to keep or the optimal rank of the matrix.
    Given a matrix Y=X+sigma*Z, with Y the observed matrix, X the matrix that we want to find, Z an matrix whose elements are iid Gaussian
    random numbers with zero mean and unit variance and sigma is the standard deviation of the noise, the matrix X can be recovered in the
    least squares sense using a truncated SVD (compute the SVD of the matrix, keep only the relevant singular values and reconstruct the
    matrix). The problem is estimate the number of such singular values to keep. This function uses the theory of Gavish & Donoho (2014) 
    that is valid for matrices with sizes larger than its rank, i.e., for low-rank matrices
    
    Args:
        matrix: matrix to estimate the rank
    
    Returns:
        thrKnownNoise: in case sigma is known, multiply this value with sigma and this gives the threshold
        thrUnknownNoise: returns the threshold directly
        noiseEstimation: returns the estimation of the noise
        wSVD: returns the singular values of the matrix
    """
    
    m, n = x.shape[-2:]
    beta = 1.0 * m / n
    
    w = (8.0 * beta) / (beta + 1 + np.sqrt(beta**2 + 14 * beta +1))
    lambdaStar = np.sqrt(2.0 * (beta + 1) + w)
    
    omega = 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43
    if (m > n):
        cov = np.swapaxes(x, -2, -1) @ x
    else:
        cov = x @ np.swapaxes(x, -2, -1)

    uSVD, wSVD, vSVD = np.linalg.svd(cov)
    
    medianSV = np.median(np.sqrt(wSVD), axis=-1)
        
    muSqrt = lambdaStar / omega
    noiseEstimation = medianSV / (np.sqrt(n) * muSqrt)      
            
    return noiseEstimation

if (__name__ == '__main__'):
    from skimage import data
    im = data.hubble_deep_field()[200:400, 200:400, 0]
    noise = 20.0 * np.random.randn(50, 200, 200)
    im = im[None, :, :] + noise

    thrKnownNoise, thrUnknownNoise, noiseEstimation, wSVD = optimalSVHT_cov(im.reshape((50, 200*200)))
    
    x = np.random.randn(10, 20, 20)
    x[0, :, :] *= 3.0
    noiseEstimation = noise_estimation(x)