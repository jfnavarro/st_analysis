""" 
Normalization functions for the st analysis package
"""
import numpy as np
import pandas as pd
from collections import Counter
import multiprocessing
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri, r, numpy2ri
import rpy2.robjects as ro
ro.conversion.py2ri = numpy2ri
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

def computeTMMFactors(counts):
    """ Compute normalization size factors
    using the TMM method described in EdgeR and returns then as a vector.
    :param counts: a matrix of counts (genes as rows)
    :return returns the normalization factors a vector
    """
    pandas2ri.activate()
    r_counts = pandas2ri.py2ri(counts)
    edger = RimportLibrary("edgeR")
    multicore = RimportLibrary("BiocParallel")
    multicore.register(multicore.MulticoreParam(multiprocessing.cpu_count()-1))
    as_matrix = r["as.matrix"]
    dds = edger.calcNormFactors(as_matrix(r_counts), method="TMM")
    pandas_sf = pandas2ri.ri2py(dds)
    pandas_cm = pandas2ri.ri2py(r.colSums(counts))
    pandas2ri.deactivate()
    return pandas_sf * pandas_cm

def computeRLEFactors(counts):
    """ Compute normalization size factors
    using the RLE method described in EdgeR and returns then as a vector.
    :param counts: a matrix of counts (genes as rows)
    :return returns the normalization factors a vector
    """
    pandas2ri.activate()
    r_counts = pandas2ri.py2ri(counts)
    edger = RimportLibrary("edgeR")
    multicore = RimportLibrary("BiocParallel")
    multicore.register(multicore.MulticoreParam(multiprocessing.cpu_count()-1))
    as_matrix = r["as.matrix"]
    dds = edger.calcNormFactors(as_matrix(r_counts), method="RLE")
    pandas_sf = pandas2ri.ri2py(dds)
    pandas_cm = pandas2ri.ri2py(r.colSums(counts))
    pandas2ri.deactivate()
    return pandas_sf * pandas_cm

def computeMnnBatchCorrection(counts):
    """Computes batch correction to a list of batches (data frames)
    where each data frame represents a batch (animal for instance).
    The batch correction is computed using Scrans::mnnCorrect()
    from Marioni et al.
    :param counts: a list of matrices of counts
    :return returns a list of batch corrected matrices of counts
    """
    pandas2ri.activate()
    as_matrix = r["as.matrix"]
    r_counts = [as_matrix(pandas2ri.py2ri(x)) for x in counts]
    scran = RimportLibrary("scran")
    multicore = RimportLibrary("BiocParallel")
    r_call = """
        function(counts){
           norm_counts = do.call(mnnCorrect, c(counts, k=20, cos.norm.out=FALSE));
           return(lapply(norm_counts$corrected, as.data.frame))
        }
    """
    r_func = r(r_call)
    r_norm_counts = r_func(r_counts)
    norm_counts = [pandas2ri.ri2py(x) for x in r_norm_counts]
    pandas2ri.deactivate()
    return norm_counts

def computeSumFactors(counts, scran_clusters=True):
    """ Compute normalization factors
    using the deconvolution method
    described in Merioni et al.
    Returns the computed size factors as a vector.
    :param counts: a matrix of counts (genes as rows)
    :return returns the normalization factors a vector
    """
    n_cells = len(counts.columns)
    pandas2ri.activate()
    r_counts = pandas2ri.py2ri(counts)
    scran = RimportLibrary("scran")
    multicore = RimportLibrary("BiocParallel")
    multicore.register(multicore.MulticoreParam(multiprocessing.cpu_count()-1))
    as_matrix = r["as.matrix"]
    if scran_clusters and n_cells >= 50:
        r_clusters = scran.quickCluster(as_matrix(r_counts), 
                                        min(n_cells/10, 10), 
                                        method="igraph")
        min_cluster_size = min(Counter(r_clusters).values())
        sizes = list(range(min(int(min_cluster_size/4), 10), 
                           min(int(min_cluster_size/2), 50), 5))
        dds = scran.computeSumFactors(as_matrix(r_counts), 
                                      clusters=r_clusters, sizes=sizes)
    else:
        sizes = list(range(min(n_cells/4, 10), min(n_cells/2, 50), 5))
        dds = scran.computeSumFactors(as_matrix(r_counts), sizes=sizes)        
    pandas_sf = pandas2ri.ri2py(dds)
    pandas2ri.deactivate()
    return pandas_sf

def logCountsWithFactors(counts, size_factors):
    """ Uses the R package scater to log a matrix of counts (genes as rows)
    and a vector of size factor using the method normalize().
    :param counts: a matrix of counts (genes as rows)
    :param size_factors: a vector of size factors
    :return the normalized log counts (genes as rows)
    """
    columns = counts.columns
    indexes = counts.index
    pandas2ri.activate()
    r_counts = pandas2ri.py2ri(counts)
    scater = RimportLibrary("scran")
    r_call = """
        function(counts, size_factors){
          sce = SingleCellExperiment(assays=list(counts=as.matrix(counts)))
          sizeFactors(sce) = size_factors
          sce = normalize(sce)
          norm_counts = logcounts(sce)
          return(as.data.frame(norm_counts))
        }
    """
    r_func = r(r_call)
    r_norm_counts = r_func(r_counts, size_factors)
    pandas_norm_counts = pandas2ri.ri2py(r_norm_counts)
    pandas_norm_counts.index = indexes
    pandas_norm_counts.columns = columns
    pandas2ri.deactivate()
    return pandas_norm_counts

def computeSizeFactors(counts):
    """ Computes size factors using DESeq
    for the counts matrix given as input (Genes as rows
    and spots as columns).
    Returns the computed size factors as a vector.
    :param counts: a matrix of counts (genes as rows)
    :return returns the normalization factors a vector
    """
    pandas2ri.activate()
    r_counts = pandas2ri.py2ri(counts)
    deseq2 = RimportLibrary("DESeq2")
    multicore = RimportLibrary("BiocParallel")
    multicore.register(multicore.MulticoreParam(multiprocessing.cpu_count()-1))
    dds = deseq2.estimateSizeFactorsForMatrix(r_counts)
    pandas_sf = pandas2ri.ri2py(dds)
    pandas2ri.deactivate()
    return pandas_sf

def computeSizeFactorsSizeAdjusted(counts):
    """ Computes size factors using DESeq
    for the counts matrix given as input (Genes as rows
    and spots as columns) the counts are library size adjusted. 
    Returns the computed size factors as a vector.
    :param counts: a matrix of counts (genes as rows)
    :return returns the normalization factors a vector
    """
    lib_size = counts.sum(axis=0)
    counts = counts + (lib_size / np.mean(lib_size))
    return computeSizeFactors(counts)

def computeSizeFactorsLinear(counts):
    """ Computes size factors using DESeq2 iterative size factors
    for the counts matrix given as input (Genes as rows
    and spots as columns). 
    Returns the computed size factors as a vector.
    :param counts: a matrix of counts (genes as rows)
    :return returns the normalization factors a vector
    """
    pandas2ri.activate()
    r_counts = pandas2ri.py2ri(counts)
    deseq2 = RimportLibrary("DESeq2")
    multicore = RimportLibrary("BiocParallel")
    multicore.register(multicore.MulticoreParam(multiprocessing.cpu_count()-1))
    vec = rpackages.importr('S4Vectors')
    bio_generics = rpackages.importr("BiocGenerics")
    cond = vec.DataFrame(condition=base.factor(base.c(base.colnames(r_counts))))
    design = r('formula(~ condition)')
    dds = deseq2.DESeqDataSetFromMatrix(countData=r_counts, colData=cond, design=design)
    dds = bio_generics.estimateSizeFactors(dds, type="iterate")
    pandas_sf = pandas2ri.ri2py(bio_generics.sizeFactors(dds))
    pandas2ri.deactivate()
    return pandas_sf