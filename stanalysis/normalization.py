""" 
Normalization functions for the st analysis package
"""
import numpy as np
import pandas as pd
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
    as_matrix = r["as.matrix"]
    dds = edger.calcNormFactors(as_matrix(r_counts), method="TMM") * r.colSums(counts)
    pandas_sf = pandas2ri.ri2py(dds)
    pandas2ri.deactivate()
    return pandas_sf

def computeRLEFactors(counts):
    """ Compute normalization size factors
    using the RLE method described in EdgeR and returns then as a vector.
    :param counts: a matrix of counts (genes as rows)
    :return returns the normalization factors a vector
    """
    pandas2ri.activate()
    r_counts = pandas2ri.py2ri(counts)
    edger = RimportLibrary("edgeR")
    as_matrix = r["as.matrix"]
    dds = edger.calcNormFactors(as_matrix(r_counts), method="RLE") * r.colSums(counts)
    pandas_sf = pandas2ri.ri2py(dds)
    pandas2ri.deactivate()
    return pandas_sf

def computeSumFactors(counts):
    """ Compute normalization factors
    using the deconvolution method
    described in Merioni et al.
    Returns the computed size factors as a vector.
    :param counts: a matrix of counts (genes as rows)
    :return returns the normalization factors a vector
    """
    pandas2ri.activate()
    r_counts = pandas2ri.py2ri(counts)
    scran = RimportLibrary("scran")
    as_matrix = r["as.matrix"]
    r_clusters = scran.quickCluster(as_matrix(r_counts), 20)
    dds = scran.computeSumFactors(as_matrix(r_counts), clusters=r_clusters,
                                  sizes=r.c(10,20,30,40), positive=True)
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
    scater = RimportLibrary("scater")
    r_call = """
        function(counts, size_factors){
          sce = newSCESet(countData=counts)
          sce@phenoData$size_factor = size_factors
          sce = normalize(sce, recompute_cpm=FALSE)
          norm_counts = sce@assayData$norm_exprs
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
    deseq = RimportLibrary("DESeq")
    dds = deseq.estimateSizeFactorsForMatrix(r_counts)
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
    vec = rpackages.importr('S4Vectors')
    bio_generics = rpackages.importr("BiocGenerics")
    cond = vec.DataFrame(condition=base.factor(base.c(base.colnames(r_counts))))
    design = r('formula(~ condition)')
    dds = deseq2.DESeqDataSetFromMatrix(countData=r_counts, colData=cond, design=design)
    dds = bio_generics.estimateSizeFactors(dds, type="iterate")
    pandas_sf = pandas2ri.ri2py(bio_generics.sizeFactors(dds))
    pandas2ri.deactivate()
    return pandas_sf