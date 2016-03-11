#!/bin/bash
# Takes a BED file as input obtained combined different ST data BED files
# and computes a counts file that is compatible with genome browsing.
set -e
echo "Sorting pooled ctss"
echo $1

sort -k 1,1 -k 2,2n -k 3,3n -k6,6 $1 > bed_file_sorted
awk -F "\t" '{print $1"\t"$2"\t"$3"\t"$6"\t"$7"\t"$8}' < bed_file_sorted > bed_file_sorted_arranged

echo "Dividing pos and neg strands"
awk -F'\t' '{ if ($4 == "+" ) print $0 }' bed_file_sorted_arranged > bed_file_sorted_arranged_pos
awk -F'\t' '{ if ($4 == "-" ) print $0 }' bed_file_sorted_arranged > bed_file_sorted_arranged_neg

echo "CTSS counts: postive strand"
groupBy -i bed_file_sorted_arranged_pos -g 1,2 -c 6 -o count > bed_file_sorted_arranged_pos_counts

echo "CTSS counts: negative strand"
groupBy -i bed_file_sorted_arranged_neg -g 1,3 -c 6 -o count > bed_file_sorted_arranged_neg_counts

echo "CTSS output to BED6 format (e.g. for Zenbu upload"
awk -F "\t" '{print $1"\t"$2"\t"$2+1"\t"$1":"$2"-"$2",+""\t"$3"\t""+"}' < bed_file_sorted_arranged_pos_counts > bed_file_sorted_arranged_pos_counts_4z
awk -F "\t" '{print $1"\t"$2-1"\t"$2"\t"$1":"$2"-"$2",-""\t"$3"\t""-"}' < bed_file_sorted_arranged_neg_counts > bed_file_sorted_arranged_neg_counts_4Z

cat bed_file_sorted_arranged_pos_counts_4z bed_file_sorted_arranged_neg_counts_4Z > bed_file_counts.bed
sed 's/^/chr/' bed_file_counts.bed > output.tmp
mv output.tmp bed_file_counts.bed

rm bed_file_sorted_arranged_pos_counts_4z
rm bed_file_sorted_arranged_neg_counts_4Z
rm bed_file_sorted_arranged_pos_counts
rm bed_file_sorted_arranged_neg_counts
rm bed_file_sorted_arranged_pos
rm bed_file_sorted_arranged_neg
rm bed_file_sorted_arranged
rm bed_file_sorted

echo "DONE"
