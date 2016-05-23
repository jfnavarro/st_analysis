'''
Created on May 23, 2016

@author: josefernandeznavarro
'''
import numpy as np

def computeSizeFactors(counts, function=np.median):
    ''' Compute size factors to normalize gene counts
    as in DESeq 1 and 2
    This is just a code snipped from the original implementation
    in R :               
     locfunc = stats::median
     loggeomeans <- rowMeans(log(counts))
     if (all(is.infinite(loggeomeans))) {
        stop("every gene contains at least one zero, cannot compute log geometric means")
     }
     sf <- apply(counts, 2, function(cnts) {
               exp(locfunc((log(cnts) - loggeomeans)[is.finite(loggeomeans) & cnts > 0]))
           })
           
    @param counts a data frame with the counts to normalize (genes as rows)
    @param function the distance function to apply to compute the factors
    @return the size factors as an array
    '''
    # Geometric means of rows
    loggeomans = np.log(counts).mean(axis=1)
    if np.all(np.isinf(loggeomans)):
        raise RuntimeError("every gene contains at least one zero, cannot compute log geometric means")
    # Apply to columns
    return counts.apply(lambda x: np.exp(function((np.log(x) - loggeomans)[np.isfinite(loggeomans)])), axis=0)
