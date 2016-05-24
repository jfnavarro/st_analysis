# ST Analsis 

Different scripts for visualization and analysis of single cell data (Mainly Spatial Transcriptomics data)

##To convert JSON output from the ST Pieline to a data frame

The current format of the ST Pipeline for the gene counts per spot is JSON
but that is being changed to a data frame format. However, to
quickly convert from JSON to a data frame (genes as columns) the script json_to_table.py 
can be used

##To compute ST_TTs

ST_TTs gives means to obtain counts using the Transcription Termination Sites
instead of the whole gene locus. This gives higher resolution. 

- Take the bed file from the output of the ST Pipeline and decide how to filter it out 
(include or not non annotated transcripts and remove or not features outside tissue) 
To remove non annotated simply use a grep (__no_feature) 
To remove spots outside the tissue use the script filterSTData.py 
with a selection file exported from the ST viewer on the same dataset. 
- Make sure that the selection made in the viewer is from the exactly 
the same run as the original bed file. 
Otherwise update the counts by using convert_selection_counts.py
- Run countClusters.py with several values of the min_distance and min_value parameters to generate ST_TTs.
Then look in Zenbu or any other genome brower and choose the parameters that fit the best 
(use clusters_to_igv.py to add to zenbu the ST_TTs files 
and compute_bed_counts.sh to transform the bed file and add ist to Zenbu). 
- Run tag_clusters_to_table.py on the optimal output from the step before. 
So to obtain a counts table (TTs as columns and spots as rows).                                                

##To do un-supervised learning

- Run the unsupervised.py script (you need to know how many clusters you want to find beforehand)
Basically you need :
- Path to a data frame with counts (spots as rows and gene as columns)
- Clustering algorithm to use
- Dimensionality reduction algorithm to use
- Image of the tissue (optional)
- Alignment matrix to transform spot coordinates to image pixel coordinates (optional)

##To do supervised learning

- Run supervised.py with the output from unsupervised.py
You need :
- A train set (the normalized counts)
- A test set (the normalized counts)
- Annotation for training (CLASS X Y)
- Annotation for testing if available (CLASS X Y)
- Image of the tissue (optional)
- Alignment matrix to transform spot coordinates to image pixel coordinates (optional)

##To visualize ST data (output from the ST Pipeline) 
