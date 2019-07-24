""" Different functions for analysis of ST datasets
"""
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors as mpcolors
from collections import Counter
import numpy as np
from rpy2.robjects import pandas2ri, r, numpy2ri, globalenv
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.packages as rpackages
base = rpackages.importr("base")

def RimportLibrary(lib_name):
    """ Helper function to import R libraries
    using the rpy2 binder
    """
    if not rpackages.isinstalled(lib_name):
        base.source("http://www.bioconductor.org/biocLite.R")
        biocinstaller = rpackages.importr("BiocInstaller")
        biocinstaller.biocLite(lib_name)
    return rpackages.importr(lib_name)

def deaDESeq2(counts, conds, comparisons, alpha):
    """Makes a call to DESeq2 to perform D.E.A. in the given
    counts matrix with the given conditions and comparisons.
    Returns a list of DESeq2 results for each comparison
    """
    pandas2ri.activate()
    results = list()
    try:
        deseq2 = RimportLibrary("DESeq2")
        # Create the R conditions and counts data
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_counts = ro.conversion.py2rpy(counts)
        cond = robjects.DataFrame({"conditions": robjects.StrVector(conds)})
        design = r('formula(~ conditions)')
        dds = r.DESeqDataSetFromMatrix(countData=r_counts, colData=cond, design=design)
        dds = r.DESeq(dds, parallel=False, useT=True, 
                      minmu=1e-6, minReplicatesForReplace=np.inf)
        # Perform the comparisons and store results in list
        for A,B in comparisons:
            result = r.results(dds, 
                               contrast=r.c("conditions", A, B), 
                               alpha=alpha, 
                               parallel=False)
            result = r['as.data.frame'](result)
            genes = r['rownames'](result)
            with localconverter(ro.default_converter + pandas2ri.converter):
                result = ro.conversion.rpy2py(result)
            # There seems to be a problem parsing the rownames from R to pandas
            # so we do it manually
            result.index = genes
            results.append(result)
    except Exception as e:
        raise e
    finally:
        pandas2ri.deactivate()
    return results

def linear_conv(old, min, max, new_min, new_max):
    """ A simple linear conversion of one value for one scale to another
    """
    return ((old - min) / (max - min)) * ((new_max - new_min) + new_min)
   
def weighted_color(colors, probs, n_bins=100):
    """Compute a weighted 0-1 value given
    a list of colours, probabalities and number of bins"""
    n_classes = float(len(colors)-1)
    l = 1.0 / n_bins
    h = 1-l
    p = 0.0
    for i,prob in enumerate(probs):
        wi = linear_conv(float(i),0.0,n_classes,h,l)
        p += abs(prob * wi)
    return p
 
def composite_colors(colors, probs):
    """Merge the set of colors
    given using a set of probabilities"""
    merged_color = [0.0,0.0,0.0,1.0]
    for prob,color in zip(probs,colors):
        new_color = mpcolors.colorConverter.to_rgba(color)
        merged_color[0] = (new_color[0] - merged_color[0]) * prob + merged_color[0]
        merged_color[1] = (new_color[1] - merged_color[1]) * prob + merged_color[1]
        merged_color[2] = (new_color[2] - merged_color[2]) * prob + merged_color[2]
    return merged_color