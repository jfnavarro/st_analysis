# Spatial Transcriptomics Analysis 

Different tools for visualization, conversion and analysis of Spatial Transcriptomics data

To install this packate just clone or download the repository and type:

    python setup.py install
    
A bunch of scripts will then be available in your system. 

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
    
###To adjust the spot coordinates to pixel coordinates in a JSON file.
If you have obtained pixel coordiantes for a dataset and you want to update
the JSON file with the new coordinates you can use the following script :

    adjust_json_coordinates.py --json-file stdata.json --coordinates-file new_coordinates.txt --outfile stdata_aligned.json
    
###To convert selections extracted from the ST Viewer to matrix format (ST data format)
Older versions of the ST Viewer export the selections in tab delimited format. 
To convert this file to a matrix (data frame) you can use the following :

    tab_to_matrix.py --tab-file selection.txt --outfile selection.tsv

###To do un-supervised learning
To see how spots cluster together based on their expression profiles you can run : 

    unsupervised.py --counts-table-files matrix_counts.tsv --normalization DESeq --num-clusters 5 --clustering-algorithm KMeans --dimensionality-algorithm tSNE --alignment-files alignment_file.txt --image-files tissue_image.JPG
    
  The script can be given one or serveral datasets (matrices with counts). It will perform dimesionality reduction
  and cluster the spots together. It generates a scatter plot of the clusters. It also generates an image for
  each dataset of the predicted classes on top of the tissue image (tissue image for each dataset must be given and optionally 
  an alignment file to convert to pixel coordiantes)
  
  To describe the parameters you can type --help 
  
        unsupervised.py --help
        
        A script that does un-supervised classification on single cell data (Mainly
        used for Spatial Transcriptomics) It takes a list of data frames as input and
        outputs : - the normalized counts as a data frame (one for each dataset) - a
        scatter plot with the predicted classes for each spot - a file with the
        predicted classes for each spot and the spot coordinates (one for each
        dataset) The spots in the output file will have the index of the dataset
        appended. For instance if two datasets are given the indexes will be (1 and
        2). The user can select what clustering algorithm to use and what
        dimensionality reduction technique to use. The user can optionally give a list
        of images and image alignments to plot the predicted classes on top of the
        image. Then one image for each dataset will be generated. @Author Jose
        Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>

        optional arguments:
            -h, --help            show this help message and exit
            --counts-table-files COUNTS_TABLE_FILES
                                One or more matrices with gene counts per feature/spot (genes as
                                columns)
            --normalization [STR]
                                Normalize the counts using (RAW - DESeq - TPM)
                                (default: DESeq)
            --num-clusters [INT]  If given the number of clusters will be adjusted.
                                Otherwise they will be pre-computed (default: 3)
            --clustering-algorithm [STR]
                                What clustering algorithm to use after the
                                dimensionality reduction (Hierarchical - KMeans)
                                (default: KMeans)
            --dimensionality-algorithm [STR]
                                What dimensionality reduction algorithm to use (tSNE -
                                PCA - ICA - SPCA) (default: tSNE)
            --alignment-files ALIGNMENT_FILES
                                One of more tag delimited files containing and alignment matrix for the images
                                (array coordinates to pixel coordinates) as a 3x3 matrix in one row. Optional.
            --image-files IMAGE_FILES      
                                When given the data will plotted on top of the image,
                                if the alignment matrix is given the data points will be transformed to pixel coordinates.
                                It can be one ore more, ideally one for each input dataset
            --outdir OUTDIR       
                                Path to output dir

###To do supervised learning
You can train a classifier with the expression profiles of a set of spots
where you know the class (cell type) and then predict on a new dataset
of the same tissue. For that you can use the following script :

    supervised.py --train-data data_matrix.tsv --test-data data_matrix.tsv --train-casses train_classes.txt --test-classes test_classes.txt --alignment alignment_file.txt --image tissue_image.jpg
    
  This will generate some statistics, a file with the predicted classes for each spot and a plot of the predicted spots on top of the tissue image (if the image and the alignment matrix are given). The script has been updated to be able to take as input more than one dataset/alignment/image for the training data.
  
  To know more about the parameters you can type --help
  
        supervised.py --help
        
        This script performs a supervised prediction using a training set and a test
        set. The training set will be a data frame with normalized counts from single
        cell data and the test set will also be a data frame with counts. A file with
        class labels for the training set is needed so the classifier knows what class
        each spot(row) in the training set belongs to. It will then try to predict the
        classes of the spots(rows) in the test set. If class labels for the test sets
        are given the script will compute accuracy of the prediction. The script will
        output the predicted classes and the spots plotted on top of an image if the
        image is given. @Author Jose Fernandez Navarro
        <jose.fernandez.navarro@scilifelab.se>

        optional arguments:
            -h, --help            show this help message and exit
            --train-data TRAIN_DATA
                                One or more data frames with normalized counts
            --test-data TEST_DATA
                                The data frame with the normalized counts for testing
            --train-classes TRAIN_CLASSES
                                One of more files with the class of each spot in the train data
            --test-classes TEST_CLASSES
                                A tab delimited file mapping barcodes to their classes
                                for testing
            --alignment ALIGNMENT
                                A file containing the alignment image (array
                                coordinates to pixel coordinates) as a 3x3 matrix
            --image IMAGE       When given the data will plotted on top of the image,
                                if the alignment matrix is given the data will be
                                aligned
            --outdir OUTDIR     Path to output dir

###To visualize ST data (output from the ST Pipeline) 

Use the script st_data_plotter.py. It can plot ST data, it can use
filters (counts or genes) it can highlight spots with reg. expressions
of genes and it can highlight spots by giving a file with spot coordinates
and labels. You need a matrix with the gene counts by spot and optionally
the a tissue image and an alignment matrix. A example run would be : 

    st_data_plotter.py --cutoff 2 --filter-genes Actb* --image tissue_image.jpg --alignment alignment_file.txt data_matrix.tsv
    
  This will generate a scatter plot of the expression of the spots that contain a gene Actb and with higher expression than 2 and it will use the tissue image as background. You could optionally pass a list of spots with their classes (Generated with unsupervised.py) to highlight spots in the scatter plot. More info if you type --help
  
        st_data_plotter.py --help
        
        Script that creates a quality scatter plot from a ST-data file in data frame
        format. The output will be a .png file with the same name as the input file if
        no name if given. It allows to highlight spots with colors using a file with
        the following format : CLASS_NUMBER X Y It allows to choose transparency for
        the data points It allows to pass an image so the spots are plotted on top of
        it (an alignment file can be passed along to convert spot coordinates to pixel
        coordinates) It allows to normalize the counts using DESeq It allows to filter
        out by counts or gene names (following a reg-exp pattern) what spots to plot
        @Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>

        positional arguments:
            input_data            
                        A data frame with counts from ST data (genes as columns)

        optional arguments:
            -h, --help            
                        show this help message and exit
            --image IMAGE         
                        When given the data will plotted on top of the image,
                        if the alignment matrix is given the data will be
                        aligned
            --cutoff [FLOAT]     
                        Do not include genes below this reads cut off (default: 0.0)
            --highlight-spots HIGHLIGHT_SPOTS
                        A file containing spots (x,y) and the class/label they
                        belong to CLASS_NUMBER X Y
            --alignment ALIGNMENT
                        A file containing the alignment image (array
                        coordinates to pixel coordinates) as a 3x3 matrix
            --data-alpha [FLOAT]  
                        The transparency level for the data points, 0 min and 1 max (default: 1.0)
            --highlight-alpha [FLOAT]
                        The transparency level for the highlighted barcodes, 0
                        min and 1 max (default: 1.0)
            --dot-size [INT]    
                        The size of the dots (default: 50)
            --normalize-counts  
                        If given the counts in the imput table will be normalized using DESeq
            --filter-genes FILTER_GENES
                        Regular expression for gene symbols to filter out. Can
                        be given several times.
            --outfile OUTFILE     
                        Name of the output file
