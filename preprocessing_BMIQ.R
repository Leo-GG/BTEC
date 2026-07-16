# =============================================================================
# DNA Methylation Data Preprocessing with BMIQ Normalization
# =============================================================================
#
# Description:
#   This script processes raw Illumina DNA methylation array data (IDAT files)
#   and applies BMIQ (Beta-MIxture Quantile) normalization to correct for the
#   bias between Type I and Type II probe designs on the array.
#
# Input:
#   - Raw IDAT files from Illumina 450K or EPIC methylation arrays
#
# Output:
#   - CSV file containing BMIQ-normalized beta values (methylation levels 0-1)
#
# Dependencies:
#   - minfi: For reading IDAT files and extracting beta values
#   - wateRmelon: For BMIQ normalization algorithm
#   - Array annotation packages (450K or EPIC)
#
# Author: [Your Name]
# Date: [Date]
# =============================================================================

# -----------------------------------------------------------------------------
# 1. SETUP AND CONFIGURATION
# -----------------------------------------------------------------------------

# Clear workspace
rm(list = ls())

# Load required libraries
library(minfi)
library(wateRmelon)  # Contains BMIQ function

# Load array annotation package (uncomment the appropriate one)
library(IlluminaHumanMethylation450kanno.ilmn12.hg19)  # For 450K array
# library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)  # For EPIC array

# Set path to the directory containing IDAT files
# IDAT files should be named: [SampleID]_Grn.idat and [SampleID]_Red.idat
idat_path <- "path/to/your/idat/files/"

# Set output filename
output_file <- "BMIQ_normalized_betas.csv"

# -----------------------------------------------------------------------------
# 2. READ RAW DATA
# -----------------------------------------------------------------------------

cat("=== Reading IDAT Files ===\n")
cat("Input directory:", idat_path, "\n")

# Read all IDAT files in the directory
# The function automatically pairs _Grn.idat and _Red.idat files
rgSet <- read.metharray.exp(base = idat_path, recursive = TRUE)

cat("Number of samples:", ncol(rgSet), "\n")
cat("Number of probes:", nrow(rgSet), "\n")

# -----------------------------------------------------------------------------
# 3. EXTRACT BETA VALUES
# -----------------------------------------------------------------------------

cat("\n=== Extracting Beta Values ===\n")

# Extract raw beta values with offset to avoid extreme values (0 or 1)
# Beta = Methylated / (Methylated + Unmethylated + offset)
# The offset (100) prevents division issues and extreme beta values
beta_values <- getBeta(rgSet, offset = 100)

cat("Beta matrix dimensions:", dim(beta_values), "\n")
cat("Beta value range:", round(range(beta_values, na.rm = TRUE), 4), "\n")

# Convert to standard matrix format
beta_matrix <- as.matrix(beta_values)

# -----------------------------------------------------------------------------
# 4. GET PROBE TYPE ANNOTATIONS
# -----------------------------------------------------------------------------

cat("\n=== Retrieving Probe Annotations ===\n")

# Get probe annotations from the array manifest
# This includes probe type (Type I or Type II) needed for BMIQ
annotation <- getAnnotation(rgSet)

# Extract probe types (I or II)
probe_types <- annotation$Type
names(probe_types) <- rownames(annotation)

# Subset to match probes in beta matrix
probe_types_filtered <- probe_types[rownames(beta_matrix)]

cat("Type I probes:", sum(probe_types_filtered == "I", na.rm = TRUE), "\n")
cat("Type II probes:", sum(probe_types_filtered == "II", na.rm = TRUE), "\n")

# -----------------------------------------------------------------------------
# 5. QUALITY CONTROL - HANDLE MISSING ANNOTATIONS
# -----------------------------------------------------------------------------

cat("\n=== Quality Control ===\n")

# Check for probes with missing type annotations
missing_annotation <- is.na(probe_types_filtered)

if (any(missing_annotation)) {
  n_missing <- sum(missing_annotation)
  cat("WARNING:", n_missing, "probes have missing type annotations - removing\n")
  
  # Remove probes with missing annotations
  valid_probes <- !missing_annotation
  beta_matrix <- beta_matrix[valid_probes, ]
  probe_types_filtered <- probe_types_filtered[valid_probes]
  
  cat("Remaining probes after QC:", nrow(beta_matrix), "\n")
} else {
  cat("All probes have valid type annotations\n")
}

# Report probe type distribution per sample
cat("\nProbe type distribution:\n")
probe_summary <- sapply(1:ncol(beta_matrix), function(i) {
  beta_col <- beta_matrix[, i]
  valid <- !is.na(beta_col) & !is.na(probe_types_filtered)
  c(TypeI = sum(probe_types_filtered[valid] == "I"),
    TypeII = sum(probe_types_filtered[valid] == "II"),
    Missing = sum(is.na(beta_col)))
})
colnames(probe_summary) <- colnames(beta_matrix)
print(probe_summary[, 1:min(5, ncol(probe_summary))])  # Show first 5 samples
if (ncol(probe_summary) > 5) cat("... and", ncol(probe_summary) - 5, "more samples\n")

# -----------------------------------------------------------------------------
# 6. BMIQ NORMALIZATION
# -----------------------------------------------------------------------------

cat("\n=== BMIQ Normalization ===\n")

# Convert probe types to numeric format required by BMIQ
# Type I = 1, Type II = 2
design_vector <- ifelse(probe_types_filtered == "I", 1, 2)

# Initialize output matrix for normalized values
beta_BMIQ <- matrix(NA, 
                    nrow = nrow(beta_matrix), 
                    ncol = ncol(beta_matrix),
                    dimnames = dimnames(beta_matrix))

# Track normalization success/failure
normalization_status <- data.frame(
  sample = colnames(beta_matrix),
  status = "pending",
  stringsAsFactors = FALSE
)

# Apply BMIQ normalization to each sample
cat("Processing", ncol(beta_matrix), "samples...\n")

for (i in 1:ncol(beta_matrix)) {
  sample_name <- colnames(beta_matrix)[i]
  cat("  [", i, "/", ncol(beta_matrix), "] ", sample_name, "... ", sep = "")
  
  tryCatch({
    # Run BMIQ normalization
    # Parameters:
    #   beta.v: Vector of beta values for one sample
    #   design.v: Vector indicating probe type (1=Type I, 2=Type II)
    #   nL: Number of mixture components (default 3: unmethylated, hemi, methylated)
    #   doH: Whether to normalize hemimethylated probes
    #   nfit: Number of probes used for fitting
    #   th1.v: Thresholds for Type I probe fitting
    #   niter: Maximum iterations for optimization
    #   plots: Whether to generate diagnostic plots
    bmiq_result <- BMIQ(
      beta.v = beta_matrix[, i],
      design.v = design_vector,
      nL = 3,
      doH = TRUE,
      nfit = 5000,
      th1.v = c(0.2, 0.75),
      th2.v = NULL,
      niter = 5,
      tol = 0.001,
      plots = FALSE,
      pri = FALSE
    )
    
    # Store normalized beta values
    beta_BMIQ[, i] <- bmiq_result$nbeta
    normalization_status$status[i] <- "success"
    cat("done\n")
    
  }, error = function(e) {
    warning(paste("BMIQ failed for sample", sample_name, ":", e$message))
    normalization_status$status[i] <<- paste("failed:", e$message)
    cat("FAILED\n")
  })
}

# Report normalization summary
cat("\n=== Normalization Summary ===\n")
cat("Successful:", sum(normalization_status$status == "success"), "\n")
cat("Failed:", sum(normalization_status$status != "success" & 
                   normalization_status$status != "pending"), "\n")

# -----------------------------------------------------------------------------
# 7. SAVE OUTPUT
# -----------------------------------------------------------------------------

cat("\n=== Saving Results ===\n")

# Save normalized beta values
write.csv(beta_BMIQ, output_file)
cat("Normalized beta values saved to:", output_file, "\n")

# Save normalization status log
status_file <- sub(".csv$", "_status.csv", output_file)
write.csv(normalization_status, status_file, row.names = FALSE)
cat("Normalization status log saved to:", status_file, "\n")

# Final summary
cat("\n=== Processing Complete ===\n")
cat("Output dimensions:", dim(beta_BMIQ), "\n")
cat("Beta value range (normalized):", round(range(beta_BMIQ, na.rm = TRUE), 4), "\n")

# -----------------------------------------------------------------------------
# END OF SCRIPT
# -----------------------------------------------------------------------------
