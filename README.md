# Spatial Transcriptomics Analysis 

Set of tools for visualization, processing and analysis (supervised and un-supervised learning,
image alignment, etc..) of Spatial Transcriptomics datasets (can also be used for single cell data
and Visium datasets).

The package is compatible with the output format of the data generated with the
ST Pipeline (https://github.com/SpatialTranscriptomicsResearch/st_pipeline) and 
it is compatible with any single cell dataset where the data is stored as 
a matrix of counts (genes as columns and spot/cells as rows).

### License
MIT License, see LICENSE file.

### Authors
See AUTHORS file.

### Contact
For bugs, feedback or help you can contact Jose Fernandez Navarro <jc.fernandez.navarro@gmail.com>

### Input Format
The referred matrix format is the ST data format, a matrix of counts where spot coordinates are row names
and the genes are column names. This matrix format (.TSV) is generated with the
[ST Pipeline](https://github.com/jfnavarro/st_pipeline)

### Installation
Before you install the ST Analysis package we recommend that you create a Python 3 virtual
environment. We recommend [Anaconda](https://anaconda.org/anaconda/python).

The ST Analysis is only computatible with Python 3. 

The following instructions are for installing the ST Analysis package with Python 3.6 and Anaconda

    conda install matplotlib
    conda install pandas
    conda install scikit-learn
    git clone https://github.com/jfnavarro/st_analysis.git
    cd st_analysis
    python setup.py install
    

A bunch of scripts (described behind) will then be available in your system.
Note that you can always type script_name.py --help to get more information
about how the script works. 

## Analysis tools (NOTE the interface will be ported to CLI soon)

### To do un-supervised learning
To see how spots cluster together based on their expression profiles you can run:

    unsupervised.py --counts-files matrix_counts.tsv --normalization REL --num-clusters 5 --clustering KMeans --dimensionality tSNE --use-log-scale 
    
The script can be given one or serveral datasets (matrices with counts). It will perform dimesionality reduction
and then cluster the spots together based on the dimesionality reduced space.
It generates a scatter plot of the clusters. 
It also generate a file with the predicted classes for each spot that can be used in other analysis.
To know more about the parameters you can type --help

### To do supervised learning
You can train a classifier with the expression profiles of a set of spots
where you know the class (spot type) and then predict on a new dataset
of the same tissue. For that you can use the following script:

    supervised.py --train-data data_matrix.tsv --test-data data_matrix.tsv --train-casses train_classes.txt --test-classes test_classes.txt
    
This will generate some statistics, a file with the predicted classes for each spot and a plot of
the predicted spots on top of the tissue image (if the image and the alignment matrix are given).
The script allows for several options for normalization and classification settings and algorithms. 
The test/train classes file shoud look like:

    XxY 1
    XxY 1
    XxY 2

Where X is the spot X coordinate and Y is the spot Y coordinate and 1,1 and 2 are
spot classes (regions).
To know more about the parameters you can type --help

NOTE: there is version that uses GPU and NN (supervised_torch.py)

### To visualize ST data (output from the ST Pipeline) 
Use the script st_data_plotter.py to plot ST data, you can use different thresholds and
filters (counts) and different normalization and visualization options. 
It plots one image for each gene given in the --show-genes option (one sub-image for each input dataset).
You need one or many matrices with the spots as rows and the genes as columns. 

    data_plotter.py --cutoff 2 --show-genes Actb --counts-files data_matrix.tsv --normalization REL
    
This will generate a scatter plot of the expression of the spots that contain a gene Actb and with higher expression than 2.

More info if you type --help
  
### To filter a matrix of counts (keep or remove genes)

    filter_genes_matrix.py --help
    keep_genes_matrix.py --help

### To align ST datasets using the images 

    align_sections.py --help
    
### To visualize covariants on a manifold (dimensionality reduction)

    dimredu_plotter.py --help
