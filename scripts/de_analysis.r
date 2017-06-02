#!/usr/local/bin/R
# example:
# Rscript de_analysis.r ../data/20160128_project_1930/chipdata_geneset_x_valid_chips.txt ../data/20160128_project_1930/chipdata_1000_limma_genes.txt

library(limma)

# parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
file_input <- args[1]
file_labels <- args[2]
group <- args[3]
thld_pval <- as.numeric(args[4])
thld_fc <- as.numeric(args[5])
file_output <- args[6]

# Load expression matrix
expr <- read.table(file_input, header=TRUE, row.names=1, sep="\t", check.names=FALSE)

# Create appropriate design matrix and assign column names, then
# appropriate contrast matrix for pairwise comparisons
labels_table <- read.table(file_labels, header=FALSE, sep="\t")
labels <- labels_table[,2]
design <- model.matrix(~0+factor(labels))

if (group == "N_vs_C") {
	# Two group analysis
	# cat("   group", group, "\n")
	colnames(design) <- c("Normal", "Cancer")
	contrast.matrix <- makeContrasts(CvsN=Cancer-Normal, levels=design)
} else if (group == "N_vs_P_vs_C") {
	# Three group analysis
	# cat("   group", group, "\n")
	colnames(design) <- c("Normal", "Polyp", "Cancer")
	contrast.matrix <- makeContrasts(CvsN=Cancer-Normal, CvsP=Cancer-Polyp, PvsN=Polyp-Normal, levels=design)
}

# print(cbind(labels_table, design))

# Find samples matching to the labels list
sample_indx <- match(as.vector(labels_table[,1]), as.vector(colnames(expr)))
expr <- expr[, sample_indx]

# Fit a linear model for each gene based on the given series of arrays
fit <- lmFit(expr, design, method="ls")

# Compute estimated coefficients and standard errors for a given set of contrasts
fit2 <- contrasts.fit(fit, contrast.matrix)

# Compute moderated t-statistics and log-odds of differential expression by empirical Bayes shrinkage of the standard errors towards a common value
fit2 <- eBayes(fit2)

# Generate list of top 10 DEGs for first comparison
top_genes <- topTable(fit2, coef=1, adjust.method="BH", sort.by="P", number=200)
row_indx <- which(top_genes$logFC > log2(thld_fc) & top_genes$P.Value < thld_pval) 
write.table(top_genes[row_indx,], file=file_output, sep='\t', quote=FALSE)
# vennDiagram(decideTests(fit2))
