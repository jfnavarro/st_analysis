# Spatial Transcriptomics Analysis Tools

Set of tools for visualization, processing and analysis (supervised, unsupervised,
image alignment, etc..) of Spatial Transcriptomics datasets.

The package is compatible with the output format of the data generated with the
ST Pipeline (https://github.com/jfnavarro/st_pipeline).

### License
MIT License, see LICENSE file.

### Authors
See AUTHORS file.

### Contact
For bugs, feedback or help you can contact Jose Fernandez Navarro <jc.fernandez.navarro@gmail.com>

### Input Format
The input format is a matrix of counts (tab delimited) where spot coordinates are row names
and the genes are column names. 

### Installation
Before you install the ST Analysis package we recommend that you create a Python 3 virtual
environment. We recommend [Anaconda](https://anaconda.org/anaconda/python).

The ST Analysis is only computatible with Python 3. 

The following instructions are for installing the ST Analysis package with Python 3.6 and Anaconda


    git clone https://github.com/jfnavarro/st_analysis.git
    cd st_analysis
    python setup.py install
    

A set of scripts (described below) will then be available in your system or
the environment if you chose to work on a specific environment.

Note that you can always type ´´script_name.py --help´´ to get more information
about how a script works. 

## Analysis tools

### Unsupervised clustering
To cluster spot together based on their expression profiles you can run:

    unsupervised.py --counts matrix_counts.tsv --normalization REL --num-clusters 5 --clustering KMeans --dimensionality tSNE --use-log-scale 
    
The script can be given one or serveral datasets (matrices of counts). 
The script allows for multiple normalization and filtering options.
The script will perform dimesionality reduction and then cluster the spot 
together based on the manifold space.
The script implements multiple options for clustering and dimensionality reduction.
The script generates a scatter plot of the clustered spots in a 2D or 3D manifold. 
The script will write the computer clusters per spot in a file (tab delimited). 

To know more about the parameters you can type --help

### Supervised classification
You can train a classifier with the expression profiles of a set of spots
where you know the class (cluster) and then predict the class of the spots 
of a new dataset of the same tissue. For that you can use the following script:

    supervised.py --train-data data_matrix.tsv --test-data data_matrix.tsv --train-casses train_classes.txt --test-classes test_classes.txt
    
This will generate some statistics and a file with the predicted classes/clusters for each spot.
The script allows for several options for normalization and classification settings and algorithms. 
The test/train classes file shoud look like:

    XxY 1
    XxY 1
    XxY 2

Where X is the spot X coordinate and Y is the spot Y coordinate and 1,1 and 2 are
spot classes (regions).
To know more about the parameters you can type --help

NOTE: there is version that uses GPU and Neural Networks (supervised_torch.py)

### To visualize Spatial Transcriptomics (ST) datasets
Use the script data_plotter.py to visualize ST data, you can use different thresholds and
filters (counts) and different normalization and visualization options. 
The script allows to plot clusters as well as gene sets. 
The script plots one image for each gene given in the --show-genes option (one sub-image for each input dataset).
You need one or more matrices of counts where the spots are rows and the genes are columns. 

    data_plotter.py --cutoff 2 --show-genes Actb Apoe --counts data_matrix.tsv --normalization REL
    
This will generate a scatter plot of the expression of the spots that contain a gene Actb and with higher expression than 2.

More info if you type --help
  
### To filter a matrix of counts (keep or remove genes)

    filter_genes_matrix.py --counts data_matrix.tsv --filter-genes Malat1 Actb
    keep_genes_matrix.py --counts data_matrix.tsv --keep-genes Malat1 Actb
    
More info if you type --help

### To merge matrices of counts into one
An index corresponding to each dataset will be appended to the spot ids. 

    merge_counts.py --counts data_matrix1.tsv data_matrix2.tsv
    
More info if you type --help
    
### To merge Spatial Transcriptomics datasets into one
This script will merge Spatial Transcriptomics datasets into one (matrices
of counrs, spot coordinates and HE images). The matrices of counts will be
merged as in the previous script. The HE images will be stiched together
and the spoot coordinates will be merged together. An index corresponding
to each dataset will be appended to the spot ids. 

    merge_datasets.py --counts data_matrix1.tsv data_matrix2.tsv --coordinates spots1.txt spots2.txt --images image1.jpg --images image2.jpg

More info if you type --help

### To align Spatial Transcriptomics datasets of the same tissue using the HE images 
If you have multiple sections (dataset) of the same tissue you may want to align them
so they all have the same orienation and angle. This enables better visuaalizations. 
The script align_sections.py takes as input a list of matrices of counts, spot coordinates
and HE images corresponding to the datasets that must be aligned. It will output a list of
aligned matrices of counts, aligned spot coordinates and aligned HE images. The script supports
different algorithms for the image detection and alignment process. 

    align_sections.py --counts data_matrix1.tsv data_matrix2.tsv --coordinates spots1.txt spots2.txt --images image1.jpg --images image2.jpg
    
More info if you type --help
    
### To visualize variables or genes on a manifold (dimensionality reduction) space
This script takes as input a list of matrices of counts and file with the reduced coordinates
of the spots (2D) and a meta-file (spots and variables). The script will generate a list of 
scatter where each variable will be plotted onto the 2D manifold of the datasets. The script
can also plot genes if given as input. It allows to use different normalization, filtering
and visualization options. 

    dimredu_plotter.py --counts data_matrix.tsv --dim-redu-file dimred.txt --meta-file meta.tsv --show-genes Actb Apoe
    
More info if you type --help

### To transform Visium datasets to the standard ST format 
This script will transformt a dataset in Visium format to the standard ST format (matrix of counts, 
spot coordinates and HE image). 

    visiumToST.py --help
    
### To transform old ST spot coordinates to new format (including pixel coordinates) 

    convert_spot_coordinates.py --help
    

