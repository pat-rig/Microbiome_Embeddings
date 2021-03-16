#!/usr/bin/env Rscript

## CHANGE FORMAT OF EMBEDDING MATRIX ##
# Relabel names of words, i.e. ASVs with id's from 1 to no.ASVs for further processing.

# Read seed from command line execution with syntax:
# Rscript --vanilla label_qualvec_transfrom_mat.R $seed
args = commandArgs(trailingOnly=TRUE)
input_file = args[1] # is a string!
output_txt = args[2]
output_fasta = args[3]

data_dir = args[4]
setwd(data_dir)

# Read original embedding matrix produced by runGlove.sh
qual_vecs_100 <- read.table(input_file, row.names = 1)
# create ids
headers <- paste("embed", seq(1, nrow(qual_vecs_100)), sep = "")
# retrieve ASVs
seqs <- rownames(qual_vecs_100)
# change rownames to ids
rownames(qual_vecs_100) <- headers
# save relabeled matrix
write.table(qual_vecs_100, output_txt, quote = F, col.names = F)

# Reformat ASVs to ">embedXX \n ASV" and save
headers_new <- paste(">", headers, sep= "")
fasta <- paste(headers_new, seqs, sep = "\n", collapse = "\n")
write(fasta, output_fasta)
