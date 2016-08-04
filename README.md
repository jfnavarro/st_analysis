# Spatial Transcriptomics Analysis 

Different tools for visualization, conversion and analysis of Spatial Transcriptomics data

### License
MIT License, see LICENSE file.

### Authors
See AUTHORS file.

### Contact
For bugs, feedback or help you can contact Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>

### Note
The referred matrix format is the ST data format, a matrix of counts where spot coordinates are row names
and the genes are column names.

The scripts that allow you to pass the tissue images can optionally take a 3x3 alignment file. 
If the images are cropped to exact the array boundaries the alignment file is not needed
unless you want to plot the image in the original image size. If the image is un-cropped
then you need the alignment file to convert from spot coordinates to pixel coordinates.

The alignment file should look like :

a11 a12 a13 a21 a22 a23 a31 a32 a33

### Installation

To install this packate just clone or download the repository and type:

    python setup.py install
    
A bunch of scripts will then be available in your system.
Note that you can always type name_script.py --help to get more information
about how the script works.

## Conversion tools

###To convert JSON output from the ST Pipeline (older versions) to a matrix format (ST Data format)

    json_to_matrix.py --json-file file.json --output file_table.tsv

###To adjust the BED output of the ST Pipeline from older version to newer versions (X and Y coordinates intead of BARCODE tags)

    adjust_bed_file.py --bed-file stdata.bed --barcodes-ids ids.txt
    
  Where ids.txt is the file containin the barcodes and coordinates (BARCODE X Y).
  
###To adjust the counts of a selection made in the ST Viewer (Old version, tab delimited file)
Lets say that you made a selection in the ST Viewer and exported it. Imagine
that you have different versions of that dataset (newer pipeline, etc..). You
can readjust the gene counts by running this :

    convert_selection_counts.py --bed-file stdata.bed --viewer-selection selection.txt --outfile new_selection.txt
  
###To convert the ST data in matrix format to JSON format (Compatible with older versions of the ST Viewer)
If you have ST data in matrix (data frame) and you want to convert it to JSON (compatibility
with older versions of the ST Viewer for example). Then you can use the following script 

    matrix_to_json.py --data-file stdata.tsv --outfile stdata.json
    
###To adjust the spot coordinates to new coordinates in a ST BED file.
For instance to adjust for printing errors or to convert to pixel coordinates
(spots not found in the coordinates files will be discarded). You can use the following script:

    adjust_bed_coordinates.py --coordinates-file coordinates.txt --outfile new_stdata.bed stdata.bed
    
###To adjust the spot coordinates to new coordinates in a ST Data in matrix format.
For instance to adjust for printing errors or to convert to pixel coordinates
(spots not found in the coordinates files will be discarded). You can use the following script:

    adjust_matrix_coordinates.py --coordinates-file coordinates.txt --outfile new_stdata.tsv --counts-matrix stdata.tsv

###To adjust the spot coordinates to pixel coordinates in a JSON file.
For instance to adjust for printing errors or to convert to pixel coordinates
(spots not found in the coordinates files will be discarded). You can use the following script:

    adjust_json_coordinates.py --json-file stdata.json --coordinates-file new_coordinates.txt --outfile new_stdata.json
    
###To convert selections extracted from the ST Viewer to matrix format (ST data format)
Older versions of the ST Viewer export the selections in tab delimited format. 
To convert this file to a matrix (data frame) you can use the following :

    tab_to_matrix.py --tab-file selection.txt --outfile selection.tsv

## Analysis tools

###To do un-supervised learning
To see how spots cluster together based on their expression profiles you can run : 

    unsupervised.py --counts-table-files matrix_counts.tsv --normalization DESeq --num-clusters 5 --clustering-algorithm KMeans --dimensionality-algorithm tSNE --alignment-files alignment_file.txt --image-files tissue_image.JPG
    
  The script can be given one or serveral datasets (matrices with counts). It will perform dimesionality reduction
  and then cluster the spots together based the dimesionality reduced coordiantes. 
  It generates a scatter plot of the clusters. It also generates an image for
  each dataset of the predicted classes on top of the tissue image (tissue image for each dataset must be given and optionally 
  an alignment file to convert to pixel coordiantes)
  
  To describe the parameters you can type --help 

###To do supervised learning
You can train a classifier with the expression profiles of a set of spots
where you know the class (cell type) and then predict on a new dataset
of the same tissue. For that you can use the following script :

    supervised.py --train-data data_matrix.tsv --test-data data_matrix.tsv --train-casses train_classes.txt --test-classes test_classes.txt --alignment alignment_file.txt --image tissue_image.jpg
    
  This will generate some statistics, a file with the predicted classes for each spot and a plot of the predicted spots on top of the tissue image (if the image and the alignment matrix are given). The script has been updated to be able to take as input more than one dataset/alignment/image for the training data.
  
  To know more about the parameters you can type --help

###To visualize ST data (output from the ST Pipeline) 

Use the script st_data_plotter.py. It can plot ST data, it can use
filters (counts or genes) it can highlight spots with reg. expressions
of genes and it can highlight spots by giving a file with spot coordinates
and labels. You need a matrix with the gene counts by spot and optionally
the a tissue image and an alignment matrix. A example run would be : 

    st_data_plotter.py --cutoff 2 --filter-genes Actb* --image tissue_image.jpg --alignment alignment_file.txt data_matrix.tsv
    
  This will generate a scatter plot of the expression of the spots that contain a gene Actb and with higher expression than 2 and it will use the tissue image as background. You could optionally pass a list of spots with their classes (Generated with unsupervised.py) to highlight spots in the scatter plot. More info if you type --help
