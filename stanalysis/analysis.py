""" Different functions for
analysis of ST datasets
"""
from stanalysis.normalization import RimportLibrary
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors as mpcolors
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, r, numpy2ri, globalenv
robjects.conversion.py2ri = numpy2ri
base = rpackages.importr("base")

def computeNClusters(counts, min_size=20):
    """Computes the number of clusters
    from the data using Scran::quickCluster"""
    pandas2ri.activate()
    r_counts = pandas2ri.py2ri(counts.transpose())
    scran = RimportLibrary("scran")    
    as_matrix = r["as.matrix"]
    clusters = scran.quickCluster(as_matrix(r_counts), min_size)
    n_clust = len(set(clusters))
    pandas2ri.deactivate()
    return n_clust

def dea(counts, conds, comparisons, size_factors=None):
    """Makes a call to DESeq2 to
    perform D.E.A. in the given
    counts matrix with the given conditions and comparisons
    """
    results = list()
    try:
        pandas2ri.activate()
        deseq2 = RimportLibrary("DESeq2")
        r("suppressMessages(library(DESeq2))")
        # Create the R conditions and counts data
        r_counts = pandas2ri.py2ri(counts)
        cond = robjects.DataFrame({"conditions": robjects.StrVector(conds)})
        design = r('formula(~ conditions)')
        dds = r.DESeqDataSetFromMatrix(countData=r_counts, colData=cond, design=design)
        if size_factors is None:
            dds = r.DESeq(dds)
        else:
            assign_sf = r["sizeFactors<-"]
            dds = assign_sf(object=dds, value=robjects.FloatVector(size_factors))
            dds = r.estimateDispersions(dds)
            dds = r.nbinomWaldTest(dds)
        # Perform the comparisons and store results in list
        for A,B in comparisons:
            result = r.results(dds, contrast=r.c("conditions", A, B))
            result = pandas2ri.ri2py_dataframe(r['as.data.frame'](result))
            results.append(result)
        pandas2ri.deactivate()
    except Exception as e:
        raise e
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

def Rtsne(counts, dimensions, theta=0.5, dims=50, perplexity=30, max_iter=1000):
    """Performs dimensionality reduction
    using the R package Rtsne"""
    pandas2ri.activate()
    r_counts = pandas2ri.py2ri(counts)
    tsne = RimportLibrary("Rtsne")    
    as_matrix = r["as.matrix"]
    tsne_out = tsne.Rtsne(as_matrix(counts), 
                          dims=dimensions, 
                          theta=theta, 
                          check_duplicates=False, 
                          pca=True, 
                          initial_dims=dims, 
                          perplexity=perplexity, 
                          max_iter=max_iter, 
                          verbose=False)
    pandas_tsne_out = pandas2ri.ri2py(tsne_out.rx2('Y'))
    pandas2ri.deactivate()
    return pandas_tsne_out