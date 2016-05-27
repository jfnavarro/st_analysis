# ST Analysis 

Different scripts for visualization and analysis of single cell data (Mainly Spatial Transcriptomics data)

###To convert JSON output from the ST Pipeline (older versions) to a data frame

    json_to_table.py --json-file file.json --output file_table.tsv

###To adjust the BED output of the ST Pipeline from older version to newer versions (X and Y intead of BARCODE tags)

    adjust_bed_file.py --bed-file stdata.bed --barcodes-ids ids.txt
    
  Where ids.txt is the file containin the barcodes and coordinates (BARCODE X Y).
  
###To adjust the counts of a selection made in the ST Viewer
Lets say that you made a selection in the ST Viewer and export it. Imagine
that you have different versions of that dataset (newer pipeline, etc..). You
can readjust the gene counts by running this :

    convert_selection_counts.py --bed-file stdata.bed --viewer-selection selection.txt --outfile new_selection.txt
  
###To compute ST_TTs

ST_TTs gives means to obtain counts using the Transcription Termination Sites
instead of the whole gene locus. This gives higher resolution. 

- Take the bed file from the output of the ST Pipeline and decide how to filter it out 
(remove or not spots outside tissue or use spots from a certain area of the tissue). For
this you can do :
 
        filter_st_data.py --barcodes-files selection.txt stdata.bed

  Where selection.txt is a selection exported from the ST Viewer (can pass more than one)
  Where stdata.bed is the output from the ST Pipeline in BED format.
  
- Compute ST TTs with several values of the min_distance and min_value parameters to generate ST_TTs.

    compute_st_tts.py --min-data-value 30 --max-cluster-size 200 stdata.bed
  
  Script that takes as input a BED file generated from the ST Pipeline and
  computes peaks clusters based on the transcription termination site and the
  strand. It uses paraclu to compute the clusters and paraclu-cut to filter out
  (optional) The computed ST-TTs are written to a file in BED format. The ST BED
  file must has the following format: CHROMOSOME START END READ SCORE STRAND
  GENE X Y @Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>

  positional arguments:
    bed_file              BED ST-data file

  optional arguments:
    -h, --help            show this help message and exit
    --min-data-value [INT]
                          Omits grouped entries whose total count is lower than
                          this (default: 30)
    --disable-filter      Disable second filter(paraclu-cut)
    --max-cluster-size [INT]
                          Discard clusters whose size in positions is bigger
                          than this (default: 200)
    --min-density-increase [INT]
                          Discard clusters whose density is lower than this
    --output OUTPUT       The name and path of the output file (default: None)    
  
- Then look in Zenbu or any other genome brower and choose the parameters that fit the best to the data
  (use clusters_to_igv.py with the output of compute_st_tts.py to convert the output to genome browsers format)
- Intersect the computed ST TTs with the original counts from the ST Pipeline to build a matrix of counts

    tag_clusters_to_table.py tag_clusters.bed stdata.bed --outfile tag_counts_matrix.tsv

- You can use now the data frame to do clustersing or anything else

###To do un-supervised learning
To see how spots cluster together based on their expression profiles you can run : 

    unsupervised.py --counts-table matrix_counts.tsv --normalization DESeq --num-clusters 5 --clustering-algorithm KMeans --dimensionality-algorithm tSNE --alignment alignment_file.txt --image tissue_image.JPG
    
  The script will generate two plots (scatter plot with the dots colored by their class and the colored spots plotted on top of the tissue image if the tissue image and alignment were given), it will also write the computed classes to file and the normalized counts to a file. 
  
  To describe the parameters you can type --help 
  
  A script that does un-supervised classification on single cell data. It takes
  a data frame as input and outputs the normalized counts (data frame), a
  scatter plot with the predicted classes and a file with the predicted classes
  and the spot coordinates. The user can select what clustering algorithm to use
  and what dimensionality reduction technique to use. If more than one data
  frame is given as input they will be merged together to do the dimensionality
  reduction and then generate plots/files for each one. @Author Jose Fernandez
  Navarro <jose.fernandez.navarro@scilifelab.se>

  optional arguments:
    -h, --help            show this help message and exit
    --counts-table COUNTS_TABLE
                          A table with gene counts per feature/spot (genes as
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
    --alignment ALIGNMENT
                          A file containing the alignment image (array
                          coordinates to pixel coordinates) as a 3x3 matrix
    --image IMAGE         When given the data will plotted on top of the image,
                          if the alignment matrix is given the data will be
                          aligned
    --outdir OUTDIR       Path to output dir

###To do supervised learning
You can train a classifier with the expression profiles of a set of spots
where you know the class (cell type) and then predict on a new dataset
of the same tissue. For that you can use the following script :

    supervised.py --train-data data_matrix.tsv --test-data data_matrix.tsv --train-casses train_classes.txt --test-classes test_classes.txt --alignment alignment_file.txt --image tissue_image.jpg
    
  This will generate some statistics, a file with the predicted classes for each spot and a plot of the predicted spots on     top of the tissue image (if the image and the alignment matrix are given). 
  To know more about the parameters you can type --help
  
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
                          The data frame with the normalized counts for training
    --test-data TEST_DATA
                          The data frame with the normalized counts for testing
    --train-classes TRAIN_CLASSES
                          A tab delimited file mapping barcodes to their classes
                          for training
    --test-classes TEST_CLASSES
                          A tab delimited file mapping barcodes to their classes
                          for testing
    --alignment ALIGNMENT
                          A file containing the alignment image (array
                          coordinates to pixel coordinates) as a 3x3 matrix
    --image IMAGE         When given the data will plotted on top of the image,
                          if the alignment matrix is given the data will be
                          aligned
    --outdir OUTDIR         Path to output dir

###To visualize ST data (output from the ST Pipeline) 

Use the script st_data_plotter.py. It can plot ST data, it can use
filters (counts or genes) it can highlight spots with reg. expressions
of genes and it can highlight spots by giving a file with spot coordinates
and labels. You need a matrix with the gene counts by spot and optionally
the a tissue image and an alignment matrix. A example run would be : 

    st_data_plotter.py --cutoff 2 --filter-genes Actb* --image tissue_image.jpg --alignment alignment_file.txt data_matrix.tsv
    
  This will generate a scatter plot of the expression of the spots that contain a gene Actb and with higher expression than 2 and it will use the tissue image as background. You could optionally pass a list of spots with their classes (Generated with unsupervised.py) to highlight spots in the scatter plot. More info if you type --help
  
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
    input_data            A data frame with counts from ST data (genes as
                          columns)

  optional arguments:
    -h, --help            show this help message and exit
    --image IMAGE         When given the data will plotted on top of the image,
                          if the alignment matrix is given the data will be
                          aligned
    --cutoff [FLOAT]      Do not include genes below this reads cut off
                          (default: 0.0)
    --highlight-spots HIGHLIGHT_SPOTS
                          A file containing spots (x,y) and the class/label they
                          belong to CLASS_NUMBER X Y
    --alignment ALIGNMENT
                          A file containing the alignment image (array
                          coordinates to pixel coordinates) as a 3x3 matrix
    --data-alpha [FLOAT]  The transparency level for the data points, 0 min and
                          1 max (default: 1.0)
    --highlight-alpha [FLOAT]
                          The transparency level for the highlighted barcodes, 0
                          min and 1 max (default: 1.0)
    --dot-size [INT]      The size of the dots (default: 50)
    --normalize-counts    If given the counts in the imput table will be
                          normalized using DESeq
    --filter-genes FILTER_GENES
                          Regular expression for gene symbols to filter out. Can
                          be given several times.
    --outfile OUTFILE     Name of the output file
