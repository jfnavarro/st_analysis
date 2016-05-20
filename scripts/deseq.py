"""
DESeq implementation in Python obtained
from :
https://github.com/nborkowska/biopython

Tool for differental expression analysis of sequence count data, based on DESeq:
http://www-huber.embl.de/users/anders/DESeq/
Requirements:
    pandas version 0.8.1
    available at http://pandas.pydata.org/getpandas.html
    statsmodels version 0.5.0 available at http://statsmodels.sourceforge.net/devel/install.html
    numpy, scipy, matplotlib
Tested on python 2.7, numpy 1.6.1, scipy 0.9.0

"""

import numpy as np
import pylab as pl

from pandas import *
from scipy import stats as st
import statsmodels.api as sm

def dnbinom(x, size, mean):
    """
    probability mass function 
    with alternative parametrization
    """
    prob = size/(size+mean)
    return np.exp(st.nbinom.logpmf(x, size, prob))


class DSet(object):
    """
    Container for a data
    
    Parameters: 
    
    data: pandas DataFrame object, ex:
    
    >>> data = read_table('./your_data.tab', index_col=0)

    conds: array-like, describing experimental conditions, ex:

    >>> conds = ['A','A','B','B']

    You may need to add row names to your data:
    
    >>> data.index = ['gene_%d' % x for x in xrange(len(data.values))]
    >>> print data
            untreated1     untreated2     treated1     treated2
    gene_0          56             44           11           23
    gene_1         345            424          560          675
    gene_2          12             45           32           17

    Or create them by mixing some columns:

    >>> print data
    chrom     start     stop    10847_2    10847_3    10847_4
    chr 1    713615   714507         38         75        390
    chr 1    742153   742162         58         11         34
    
    >>> data.index = data.pop('chrom')+':'+data.pop('start').map(str) \ 
    ...     +'-'+data.pop('stop').map(str)
    >>> print data
                           10847_2    10847_3    10847_4
    chr 1:713615-714507         38         75        390
    chr 1:742153-742162         58         11         34 
    """

    DISP_MODES = (
            'max',
            'fit-only',
            'gene-only')

    DISP_METHODS = (
            'pooled',
            'per-condition',
            'blind')
    
    def __init__(self, data, conds=None, sizeFactors=None):
        
        if isinstance(data, DataFrame):
            if conds is not None:
                """ set experimental conditions as hierarchical index """
                index = MultiIndex.from_tuples(zip(conds, data.columns), \
                                               names=['conditions','replicates'])
                new = data.reindex(columns = index)
                for idx in index:
                    new[idx] = data[idx[1]]
                self.data = new
            else:
                self.data = data
            self.conds = conds
            self.sizeFactors = sizeFactors
            self.disps = None
        else:
            raise TypeError("Data must be a pandas DataFrame object!")
 
        
        
        """ params: 
            function - use specific function when estimating the factors,
                       median is the default """   
        counts = self.data
        # Geometric means of rows
        loggeomans = np.log(counts).mean(axis=0)
        if np.all(np.inf(loggeomans)):
            print "every gene contains at least one zero, cannot compute log geometric means"
            return
        # Apply to columns 
        def lamb(x):
            np.exp(function( (np.log(x) - loggeomans).where(np.isfinite(loggeomans) & x > 0) )
        self.sizeFactors = Series(counts.apply(lamb, axis=1, index=self.data.columns))
   
        #array = self.data.values
        #geometricMean = st.gmean(array, axis=1)
        #divided = np.divide(np.delete(array, np.where(geometricMean == 0),
        #    axis=0).T, [x for x in geometricMean if x != 0])
        #self.sizeFactors = Series(function(divided, axis=1), index=self.data.columns)
         
    @staticmethod
    def getNormalizedCounts(dataframe, factors):
        """ factors: array-like or Series """
        return dataframe.div(factors)
    
    @staticmethod
    def getBaseMeansAndVariances(dataframe):
        """ dataframe - DataFrame with normalized data """
        return DataFrame({
            'bMean': np.mean(dataframe.values, axis=1),
            'bVar': np.var(dataframe.values, axis=1, ddof=1)
            }, index=dataframe.index)

    def selectReplicated(self, normalized):
        return normalized.select(lambda x:self.conds.count(x[0]) > 1,
                axis=1).groupby(axis=1, level=0)

    def _estimateAndFitDispersions(self, mav, sizeFactors, xim):
        dispsAll = Series(
                (mav.bVar - xim * mav.bMean)/(mav.bMean)**2,
                index = mav.index)
        toDel = np.where(mav.bMean.values <= 0)
        dataframe = DataFrame({
            'means': np.log(np.delete(mav.bMean.values, toDel)),
            'variances':np.delete(mav.bVar.values, toDel)
            })
        fit = sm.GLM.from_formula(
                formula='variances ~ means',
                df=dataframe,
                family = sm.families.Gamma(link=sm.families.links.log)).fit()
        return dispsAll, fit
    
    def _calculateDispersions(self, mav, sizeFactors, testing, mode):
        xim = np.mean(1/sizeFactors)
        estDisp, fit = self._estimateAndFitDispersions(
                mav, sizeFactors, xim)
        tframe = DataFrame({'means':np.log(testing)})
        fittedDisp= np.clip(
                (fit.predict(tframe)-xim*testing)/testing**2,
                1e-8, float("Inf"))
        if mode == 'max':
            disp = np.maximum(estDisp, fittedDisp)
        elif mode == 'fit-only':
            disp = fittedDisp
        else:
            disp = estDisp
        return Series(np.maximum(disp, 1e-8), index=mav.index)

    def setDispersions(self, method='per-condition', mode='gene-only'):
        """ Get dispersion estimates """
        
        if mode not in self.DISP_MODES:
            raise ValueError("Invalid mode. Choose from %s, %s, %s." \
                    % self.DISP_MODES)
        if method not in self.DISP_METHODS:
            raise ValueError("Invalid method. Choose from %s, %s, %s." \
                    % self.DISP_METHODS)
        if self.sizeFactors is None:
            raise ValueError("No size factors available. \
                    Call 'setSizeFactors' first.")

        normalized = DSet.getNormalizedCounts(
                self.data, self.sizeFactors)
        overallBMeans = np.mean(normalized.values, axis=1)
        dfr = {}

        if method == 'pooled':
            """ select all conditions with replicates and estimate a
            single pooled empirical dispersion value """
        
            replicated = self.selectReplicated(normalized)
            groupv = replicated.agg(lambda x: sum((x - np.mean(x))**2))
            bVar = groupv.sum(axis=1) / len(replicated.groups)
            meansAndVars = DataFrame({'bMean':overallBMeans,'bVar':bVar},
                    index=self.data.index) 
            dispersions = self._calculateDispersions(
                    meansAndVars, self.sizeFactors,
                    overallBMeans, mode)
            for name, df in self.data.groupby(axis=1, level=0):
                dfr[name] = dispersions
        
        elif method == 'per-condition':
            replicated = self.selectReplicated(normalized)
            if not replicated.groups:
                raise Exception("None of your conditions is replicated."
                        + " Use method='blind' to estimate across conditions")
            for name, df in replicated:
                sizeFactors = self.sizeFactors[name].values
                meansAndVars = DSet.getBaseMeansAndVariances(
                        df)
                dispersions = self._calculateDispersions(
                        meansAndVars, sizeFactors,
                        overallBMeans, mode)
                dfr[name] = dispersions
            maxDisps = DataFrame(dfr).max(axis=1)
            for name, df in normalized:
                if not dfr.has_key(name):
                    dfr[name] = maxDisps
        else:
            meansAndVars = DSet.getBaseMeansAndVariances(
                    self.data)
            dispersions = self._calculateDispersions(
                    meansAndVars, self.sizeFactors,
                    overallBMeans, mode)
            for name, df in self.data.groupby(axis=1, level=0):
                dfr[name] = dispersions

        dfr = DataFrame(dfr)
        self.disps = dfr.fillna(1e-8)
    
    def _getpValues(self, counts, sizeFactors, disps):
        kss = counts.sum(axis=1, level=0).dropna(axis=1,how='all')
        mus = DSet.getNormalizedCounts(counts, sizeFactors).mean(axis=1)
        sumDisps, pvals = {}, []
        for name, col in counts.groupby(level=0, axis=1):
            n = mus*sizeFactors[name].sum()
            fullVars = np.maximum(
                    n + disps[name]*np.power(mus,2)*np.sum(
                        np.power(sizeFactors[name].values, 2)),
                    n*(1+1e-8)
                    )
            sumDisps[name] = (fullVars - n) / np.power(n, 2)
       
        sumDisps = DataFrame(sumDisps)
        sfSum = sizeFactors.sum(level=0).dropna()
        for index, row in kss.iterrows():
            if all(v == 0 for v in row.values):
                pval=np.nan
            else:
                ks = range(int(row.sum())+1)
                """ probability of all possible counts sums with 
                the same total count """
                
                ps = dnbinom(
                        ks, 1/sumDisps.ix[index, 0],
                        mus[index]*sfSum[0]
                        )*dnbinom(row.sum()-ks, 1/sumDisps.ix[index,1],
                                mus[index]*sfSum[1])

                """ probability of observed count sums """
                pobs = dnbinom(
                        kss.ix[index, 0], 1/sumDisps.ix[index, 0],
                        mus[index]*sfSum[0]
                        )*dnbinom(kss.ix[index, 1], 1/sumDisps.ix[index,1],
                                mus[index]*sfSum[1])
                
                if kss.ix[index,0]*sfSum[1] < kss.ix[index, 1]*sfSum[0]:
                    number = ps[:int(kss.ix[index,0]+1)]
                else:
                    number = ps[int(kss.ix[index,0]):]
                pval = np.nanmin([1, 2*np.nansum(number)/np.nansum(ps)]) 
            pvals.append(pval)

        return Series(pvals,index=counts.index)
    
    @staticmethod
    def _BenjaminiHochberg(pvals):
        pvals =  pvals.order(na_last=True, ascending=False).dropna()
        l = len(pvals)
        try:
            previous = pvals[0]
            for i,j in enumerate(pvals.iteritems()):
                corrected = min(j[1]*l/(l-i),previous)
                previous = pvals[i] = corrected
        except:
            pass
        return pvals

    def nbinomTest(self, condA, condB):
        if self.disps is None:
            raise ValueError("No dispersion values available."
                    + " Call 'setDispersions' first.")
        if any(cond not in set(self.conds) for cond in [condA, condB]):
            raise ValueError("No such conditions!")
        
        func = lambda x: x[0] in [condA, condB]
        testingConds = self.data.select(func, axis=1)
        sizeFactors = self.sizeFactors.select(func)
        normalizedConds = DSet.getNormalizedCounts(testingConds, sizeFactors)
        meansAndVars = DSet.getBaseMeansAndVariances(normalizedConds)
        dispersions = self.disps.select(lambda x: x in [condA, condB], axis=1)
        p_vals = self._getpValues(testingConds, sizeFactors, dispersions) 
        adjustedPVals = DSet._BenjaminiHochberg(p_vals)
        bmvA = DSet.getBaseMeansAndVariances(normalizedConds[condA])
        bmvB = DSet.getBaseMeansAndVariances(normalizedConds[condB])
        return DataFrame({
            'baseMean': meansAndVars.bMean,
            'baseVar': meansAndVars.bVar,
            'baseMeanA': bmvA.bMean,
            'baseVarA': bmvA.bVar,
            'baseMeanB': bmvB.bMean,
            'baseVarB': bmvB.bVar,
            'pval': p_vals,
            'pvalAdj': adjustedPVals,
            'foldChange': bmvB.bMean / bmvA.bMean,
            'log2FoldChange': np.log2( bmvB.bMean / bmvA.bMean)
            }, index=self.data.index)
    
    @staticmethod
    def plotResults(log2foldchange, pvals):
        pl.scatter(log2foldchange, pvals, alpha=0.2)
        pl.yscale('log')
        pl.ylim(1,1e-50)
        pl.show()
        
