# Spatial Transcriptomics Analysis 

Different tools for visualization, processing and analysis (supervised and un-supervised learning,
differential expression analysis, etc..) of Spatial Transcriptomics datasets (can also be used for single cell data).

The package is compatible with the output format of the data generated with the
ST Pipeline (https://github.com/SpatialTranscriptomicsResearch/st_pipeline) and give full
support to plot the data onto the tissue images but it is compatible with any single cell dataset
where the data is stored as a matrix of counts (genes as columns and spot/cells as rows).

This package makes use of the following R packages:

t-SNE
https://github.com/lvdmaaten/bhtsne

Scran
https://github.com/MarioniLab/Deconvolution2016

DESeq2
http://bioconductor.org/packages/devel/bioc/html/DESeq2.html

EdgeR
https://bioconductor.org/packages/release/bioc/html/edgeR.html

### License
MIT License, see LICENSE file.

### Authors
See AUTHORS file.

### Contact
For bugs, feedback or help you can contact Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>

### Input Format
The referred matrix format is the ST data format, a matrix of counts where spot coordinates are row names
and the genes are column names. This matrix format (.TSV) is generated with the
[ST Pipeline](https://github.com/SpatialTranscriptomicsResearch/st_pipeline)

The scripts that allows you to pass the tissue HE image can optionally take a 3x3 alignment file.
If the images are cropped to the exact array boundaries the alignment file is not needed
unless you want to plot the image in the original image size. If the image is un-cropped
then you need the alignment file to convert from spot coordinates to pixel coordinates.

The alignment file should look like :

a11 a12 a13 a21 a22 a23 a31 a32 a33

Where each a correspondonds to a cell of the affine transformation matrix.

### Installation

We recommend that you install the latest R version 3.4 Once you have installed R you can open
a R terminal or Rstudio and type the following:

    source("https://bioconductor.org/biocLite.R")
    biocLite("BiocParallel")
    biocLite("scran")
    biocLite("DESeq2")
    biocLite("Rtsne")
    biocLite("edgeR")
    
Before you install the ST Analysis package we recommend that you create a Python 3 virtual
environment. We recommend [Anaconda](https://anaconda.org/anaconda/python).
The latest versions of rpy2 (R binder for Python) are only compatible with Python 3 and R 3.4 but
older versions of rpy2 are compatible with Python 2 and older R versions.

#### OSX
The following instructions are for installing the ST Analysis package with Python 3.4 and Anaconda
(should be the same for Python 3.6)
Note: we advice to update Xcode to the latest version.

    conda create -n python3.4 python=3.4
    source activate python3.4
    brew install freetype
    brew install gcc
    export CC=/usr/local/Cellar/gcc/7.2.0/bin/gcc-7
    pip install rpy2
    export CC=/usr/bin/clang
    conda install matplotlib
    conda install pandas
    conda install scikit-learn
    git clone https://github.com/SpatialTranscriptomicsResearch/st_analysis.git
    cd st_analysis
    python setup.py install

#### Linux
The following instructions are for installing the ST Analysis package with Python 3.4 and Anaconda
(should be the same for Python 3.6)
Note: we advice to install and update the developer tools packages

    conda create -n python3.4 python=3.4
    source activate python3.4
    pip install rpy2
    pip install tzlocal
    conda install matplotlib
    conda install pandas
    conda install scikit-learn
    conda install readline
    git clone https://github.com/SpatialTranscriptomicsResearch/st_analysis.git
    cd st_analysis
    python setup.py install

A bunch of scripts (described behind) will then be available in your system.
Note that you can always type script_name.py --help to get more information
about how the script works. 
The ST Analysis package is compatible with Python 2 and 3 and we recomend to use
a virtual environment to make the installation of the dependencies easier. 

* NOTE that you will need to activate your Python environment before using any of the tools

        source activate python3.4

## Analysis tools

### To do un-supervised learning
To see how spots cluster together based on their expression profiles you can run:

    unsupervised.py --counts-table-files matrix_counts.tsv --normalization DESeq2 --num-clusters 5 --clustering KMeans --dimensionality tSNE --image-files tissue_image.JPG --use-log-scale 
    
The script can be given one or serveral datasets (matrices with counts). It will perform dimesionality reduction
and then cluster the spots together based on the dimesionality reduced coordinates.
It generates a scatter plot of the clusters. It also generates an image for
each dataset of the predicted classes on top of the tissue image (tissue image for each dataset must be given and optionally
an alignment file to convert to pixel coordiantes).
It also generate a file with the predicted classes for each spot that can be used in other analysis.
To know more about the parameters you can type --help

### To do supervised learning
You can train a classifier with the expression profiles of a set of spots
where you know the class (spot type) and then predict on a new dataset
of the same tissue. For that you can use the following script:

    supervised.py --train-data data_matrix.tsv --test-data data_matrix.tsv --train-casses train_classes.txt --test-classes test_classes.txt --image tissue_image.jpg
    
This will generate some statistics, a file with the predicted classes for each spot and a plot of
the predicted spots on top of the tissue image (if the image and the alignment matrix are given).
The script can take several datasets for the training set and it allows to normalize the training and testing data.
The test/train classes file shoud look like:

    XxY 1
    XxY 1
    XxY 2

Where X is the spot X coordinate and Y is the spot Y coordinate and 1,1 and 2 are
spot classes (regions).
To know more about the parameters you can type --help

### To visualize ST data (output from the ST Pipeline) 
Use the script st_data_plotter.py to plot ST data, it can use
filters (counts) and it can plot only selected genes using regular expressions. 
You can also normalize the data for visualization.
You need one or many matrices with the gene counts and spots and optionally
a tissue image and an alignment matrix for each dataset. A example run would be:

    st_data_plotter.py --cutoff 2 --show-genes Actb* --image-files tissue_image.jpg --counts-table-files data_matrix.tsv
    
This will generate a scatter plot of the expression of the spots that contain a gene Actb and
with higher expression than 2 and it will use the tissue image as background.
More info if you type --help
  
### To slice a matrix of counts based of regions of interest
You can slice a dataset based on regions of interests (spots) obtained
manually or with unsupervised.py. You need a file defining classes for each spot
(unsupervised.py generates such files):

    XxY 1
    XxY 1
    XxY 2

Where X is the spot X coordinate and Y is the spot Y coordinate and 1,1 and 2 are
spot classes (regions).
A example run would be:

    slice_regions_matrix.py --counts-matrix dataset.tsv --spot-classes classes.txt --regions 1 3

### To perform Differential Expression Analysis (DEA)
You can perform a D.E.A between ST datasets (most likely regions of interests)
The scripts generates different plots and the list of D.E. genes in a text file for each comparison.
Basically the script needs one or more matrices of counts with ST data (genes as columns) a list
of condition labels for each dataset and a list of comparisons to make. 
The condition labels list should look like:

DATASET_INDEX:CONDITION DATASET_INDEX:CONDITION

Where DATASET_INDEX is the number (starting by 0) of the position of the dataset 
in the input and CONDITION is any string. 

The comparisons list should lool like:

CONDITION-CONDITION CONDITION-CONDITION 

Where CONDITION must be one of the CONDITIONS given in the previous list.

The scripts allows for different normalization methods and
different D.E.A. algorithms (see --help). An example run would be:

    differential_analysis.py --input-data stdata_region1.tsv stdata_region2.tsv --conditions 0:A 1:B --comparisons A-B
    
To know more about the parameters you can type --help
