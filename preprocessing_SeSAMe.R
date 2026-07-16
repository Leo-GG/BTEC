# =============================================================================
# DNA Methylation Data Preprocessing with SeSAMe
# =============================================================================
#
# Description:
#   This script processes raw Illumina DNA methylation array data (IDAT files)
#   using the SeSAMe (SEnsible Step-wise Analysis of DNA MEthylation) pipeline.
#   SeSAMe provides comprehensive preprocessing including:
#   - Detection P-value masking
#   - Background subtraction (noob)
#   - Dye bias correction
#   - Non-linear correction for Type I/II probe bias
#
# Input:
#   - Raw IDAT files from Illumina 450K, EPIC, or EPICv2 methylation arrays
#
# Output:
#   - CSV file containing SeSAMe-normalized beta values (methylation levels 0-1)
#
# Dependencies:
#   - sesame: Main preprocessing package
#   - sesameData: Annotation data (installed automatically)
#
# Reference:
#   Zhou W, Triche TJ Jr, Laird PW, Shen H. SeSAMe: reducing artifactual 
#   detection of DNA methylation by Infinium BeadChips in genomic deletions.
#   Nucleic Acids Research. 2018.
#
# Author: [Your Name]
# Date: [Date]
# =============================================================================

# -----------------------------------------------------------------------------
# 1. SETUP AND CONFIGURATION
# -----------------------------------------------------------------------------

# Clear workspace
rm(list = ls())

# Load required library
library(sesame)

# Set path to the directory containing IDAT files
# IDAT files should be named: [SampleID]_Grn.idat and [SampleID]_Red.idat
idat_path <- "path/to/your/idat/files/"

# Set output filename
output_file <- "SeSAMe_normalized_betas.csv"

# -----------------------------------------------------------------------------
# 2. CONFIGURE ARRAY MANIFEST
# -----------------------------------------------------------------------------

cat("=== Configuring Array Manifest ===\n")

# Build address file for the appropriate array platform
# Uncomment the line matching your array type:

# For Illumina 450K array:
manifest <- sesameAnno_buildAddressFile(
  "https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/HM450/homo_sapiens.tsv.gz"
)

# For Illumina EPIC (850K) array:
# manifest <- sesameAnno_buildAddressFile(
#   "https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/EPIC/homo_sapiens.tsv.gz"
# )

# For Illumina EPICv2 array:
# manifest <- sesameAnno_buildAddressFile(
#   "https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/EPICv2/hg38.tsv.gz"
# )

cat("Manifest loaded successfully\n")

# -----------------------------------------------------------------------------
# 3. IDENTIFY IDAT FILES
# -----------------------------------------------------------------------------

cat("\n=== Identifying IDAT Files ===\n")

# Find all green channel IDAT files (each sample has _Grn.idat and _Red.idat)
idat_files <- list.files(idat_path, pattern = "_Grn.idat$", full.names = TRUE)

# Extract base names (without _Grn.idat suffix)
basenames <- sub("_Grn.idat$", "", idat_files)

cat("Found", length(basenames), "samples to process\n")

if (length(basenames) == 0) {
  stop("No IDAT files found in the specified directory!")
}

# Display sample names
cat("\nSamples:\n")
for (i in 1:min(5, length(basenames))) {
  cat("  ", basename(basenames[i]), "\n")
}
if (length(basenames) > 5) {
  cat("  ... and", length(basenames) - 5, "more\n")
}

# -----------------------------------------------------------------------------
# 4. PROCESS SAMPLES WITH SeSAMe
# -----------------------------------------------------------------------------

cat("\n=== Processing Samples with SeSAMe ===\n")

# Initialize list to store beta values
beta_list <- list()

# Track processing status
processing_status <- data.frame(
  sample = basename(basenames),
  status = "pending",
  n_probes = NA,
  stringsAsFactors = FALSE
)

# Process each sample
for (i in seq_along(basenames)) {
  sample_basename <- basenames[i]
  sample_name <- basename(sample_basename)
  
  cat("  [", i, "/", length(basenames), "] ", sample_name, "... ", sep = "")
  
  tryCatch({
    # openSesame performs the full preprocessing pipeline:
    # 1. Read IDAT files
    # 2. Detection P-value calculation and masking
    # 3. Background subtraction (noob)
    # 4. Dye bias correction
    # 5. Non-linear correction (for Type I/II bias)
    # 6. Extract beta values
    beta <- openSesame(sample_basename, manifest = manifest)
    
    # Store results
    beta_list[[sample_name]] <- beta
    processing_status$status[i] <- "success"
    processing_status$n_probes[i] <- length(beta)
    
    cat("done (", length(beta), " probes)\n", sep = "")
    
  }, error = function(e) {
    warning(paste("Failed for sample:", sample_name, "\n", e$message))
    processing_status$status[i] <<- paste("failed:", e$message)
    cat("FAILED\n")
  })
}

# Report processing summary
cat("\n=== Processing Summary ===\n")
cat("Successful:", sum(processing_status$status == "success"), "\n")
cat("Failed:", sum(grepl("^failed", processing_status$status)), "\n")

# -----------------------------------------------------------------------------
# 5. COMBINE RESULTS INTO MATRIX
# -----------------------------------------------------------------------------

cat("\n=== Combining Results ===\n")

# Check if any samples were processed successfully
if (length(beta_list) == 0) {
  stop("No samples were processed successfully!")
}

# Get union of all probe names across samples
# This handles cases where different samples may have slightly different probe sets
all_probes <- Reduce(union, lapply(beta_list, names))
cat("Total unique probes:", length(all_probes), "\n")

# Create combined beta matrix
# Initialize with NA for missing values
beta_matrix <- sapply(beta_list, function(b) {
  tmp <- rep(NA, length(all_probes))
  names(tmp) <- all_probes
  tmp[names(b)] <- b
  return(tmp)
})
rownames(beta_matrix) <- all_probes

cat("Final matrix dimensions:", dim(beta_matrix), "\n")

# -----------------------------------------------------------------------------
# 6. QUALITY METRICS
# -----------------------------------------------------------------------------

cat("\n=== Quality Metrics ===\n")

# Calculate per-sample statistics
sample_stats <- data.frame(
  sample = colnames(beta_matrix),
  n_valid = colSums(!is.na(beta_matrix)),
  n_missing = colSums(is.na(beta_matrix)),
  mean_beta = colMeans(beta_matrix, na.rm = TRUE),
  median_beta = apply(beta_matrix, 2, median, na.rm = TRUE)
)

cat("\nPer-sample statistics (first 5 samples):\n")
print(head(sample_stats, 5))

# Calculate per-probe statistics
probe_missing <- rowSums(is.na(beta_matrix))
cat("\nProbes with missing values:\n")
cat("  0 missing:", sum(probe_missing == 0), "\n")
cat("  1-10% missing:", sum(probe_missing > 0 & probe_missing <= ncol(beta_matrix) * 0.1), "\n")
cat("  >10% missing:", sum(probe_missing > ncol(beta_matrix) * 0.1), "\n")

# -----------------------------------------------------------------------------
# 7. SAVE OUTPUT
# -----------------------------------------------------------------------------

cat("\n=== Saving Results ===\n")

# Save normalized beta values
write.csv(beta_matrix, output_file)
cat("Normalized beta values saved to:", output_file, "\n")

# Save processing status log
status_file <- sub(".csv$", "_status.csv", output_file)
write.csv(processing_status, status_file, row.names = FALSE)
cat("Processing status log saved to:", status_file, "\n")

# Save sample statistics
stats_file <- sub(".csv$", "_sample_stats.csv", output_file)
write.csv(sample_stats, stats_file, row.names = FALSE)
cat("Sample statistics saved to:", stats_file, "\n")

# Final summary
cat("\n=== Processing Complete ===\n")
cat("Output file:", output_file, "\n")
cat("Dimensions:", nrow(beta_matrix), "probes x", ncol(beta_matrix), "samples\n")
cat("Beta value range:", round(range(beta_matrix, na.rm = TRUE), 4), "\n")

# -----------------------------------------------------------------------------
# OPTIONAL: PARALLEL PROCESSING
# -----------------------------------------------------------------------------

# For large datasets, you can use parallel processing:
# 
# library(BiocParallel)
# 
# # Set number of cores (adjust based on your system)
# n_cores <- 4
# 
# # Process all samples in parallel
# beta_matrix <- openSesame(
#   idat_path,
#   BPPARAM = MulticoreParam(n_cores),
#   manifest = manifest
# )
#
# write.csv(beta_matrix, output_file)

# -----------------------------------------------------------------------------
# END OF SCRIPT
# -----------------------------------------------------------------------------
