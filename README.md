# BTEC - Breast Tissue-specific Epigenetic Clock

Code accompanying the paper:

> **A breast tissue-specific epigenetic clock provides accurate chronological age predictions and reveals de-correlation of age and DNA methylation in tumor-adjacent and tumor samples**
>
> Leonardo D. Garma, Sonia Pernas, Bartomeu Fullana, Andrea Vethencourt, David Vicente Baz, Teresa García Manrique, Rosario García Campelo, Cristina Reboredo, Josefa Terrasa, Antonia Perello, Ramón Colomer, Desirée Jiménez, Ruth Vera García, Susana de la Cruz Sánchez, Begoña Bermejo, Marta Tapia, Santiago González Santiago, Miguel Quintela-Fandino

## Overview

BTEC is a breast tissue-specific epigenetic clock that predicts chronological age from DNA methylation data. Unlike pan-tissue clocks (Horvath, Hannum, PhenoAge, AltumAge), BTEC was specifically trained on breast tissue samples and achieves superior performance:

| Model | Normalization | RMSE (years) | MAE (years) | r |
|-------|---------------|--------------|-------------|------|
| **BTEC** | **SeSAMe** | **5.59** | **2.95** | **0.918** |
| **BTEC** | **BMIQ** | **5.84** | **3.19** | **0.912** |
| AltumAge | SeSAMe | 8.50 | 6.31 | 0.873 |
| Hannum | BMIQ | 11.25 | 7.05 | 0.737 |
| Horvath | SeSAMe | 16.27 | 15.34 | 0.771 |
| PhenoAge | SeSAMe | 20.54 | 15.72 | 0.540 |

BTEC uses ElasticNet regression with linear and quadratic terms from CpG probes conserved across 450K, EPICv1, and EPICv2 arrays.

## Repository Structure

```
BTEC/
├── data/                          # Training data (not included)
│   ├── Features_training_BMIQ.parquet
│   ├── Features_training_Sesame.parquet
│   └── merged_filtered_mdata.csv
├── train_BMIQ.py                  # Train BTEC on BMIQ-normalized data
├── train_SeSAMe.py                # Train BTEC on SeSAMe-normalized data
├── predict_age.py                 # Predict age using BTEC and other clocks
├── preprocessing_BMIQ.R           # IDAT preprocessing with BMIQ normalization
├── preprocessing_SeSAMe.R         # IDAT preprocessing with SeSAMe pipeline
├── BTEC_BMIQ.csv                  # Trained model coefficients (BMIQ)
├── BTEC_Sesame.csv                # Trained model coefficients (SeSAMe)
├── LOCO_results_*.csv             # Leave-One-Cohort-Out validation results
└── LOCO_predictions_*.csv         # LOCO predictions per sample
```

## Installation

### Python Dependencies

```bash
pip install pandas numpy scikit-learn pyaging anndata
```

### R Dependencies

```r
install.packages("BiocManager")
BiocManager::install(c("minfi", "wateRmelon", "sesame"))
BiocManager::install("IlluminaHumanMethylation450kanno.ilmn12.hg19")  # For 450K
# BiocManager::install("IlluminaHumanMethylationEPICanno.ilm10b4.hg19")  # For EPIC
```

## Usage

### 1. Data Preprocessing (R)

Process raw IDAT files to obtain normalized beta values:

**BMIQ Normalization:**
```r
# Edit preprocessing_BMIQ.R to set your idat_path
source("preprocessing_BMIQ.R")
```

**SeSAMe Normalization:**
```r
# Edit preprocessing_SeSAMe.R to set your idat_path
source("preprocessing_SeSAMe.R")
```

### 2. Training BTEC Models (Python)

Train new BTEC models on your data:

```bash
python train_SeSAMe.py  # For SeSAMe-normalized data
python train_BMIQ.py    # For BMIQ-normalized data
```

The training scripts:
- Run 5-fold cross-validation to find optimal alpha
- Perform Leave-One-Cohort-Out (LOCO) validation
- Save model coefficients and intercept to CSV
- Save LOCO predictions for evaluation

### 3. Predicting Epigenetic Age (Python)

Predict age using BTEC and other epigenetic clocks:

```bash
# With metadata (evaluates predictions against chronological age)
python predict_age.py --betas betas.parquet --metadata metadata.csv --output predictions.csv

# Without metadata (predictions only)
python predict_age.py --betas betas.parquet --output predictions.csv
```

**Arguments:**
- `--betas`: Parquet file with beta values (probes × samples)
- `--metadata`: CSV file with sample metadata (must have "Age" column for evaluation)
- `--output`: Output CSV file for predictions (default: `age_predictions.csv`)
- `--btec-sesame-model`: Path to BTEC SeSAMe model (default: `BTEC_Sesame.csv`)
- `--btec-bmiq-model`: Path to BTEC BMIQ model (default: `BTEC_BMIQ.csv`)

**Supported Clocks:**
- `BTEC_SeSAMe` - Breast tissue-specific (SeSAMe normalization)
- `BTEC_BMIQ` - Breast tissue-specific (BMIQ normalization)
- `horvath2013` - Pan-tissue clock (via pyaging)
- `hannum` - Blood-based clock (via pyaging)
- `dnamphenoage` - Phenotypic age clock (via pyaging)
- `altumage` - Deep learning clock (via pyaging)

### Output Files

**Predictions:**
- `predictions.csv` - Predicted ages for each sample and clock
- `predictions_metrics.csv` - RMSE, MAE, and correlation for each clock (if metadata provided)

**Training outputs:**
- `BTEC_*.csv` - Model coefficients with intercept
- `LOCO_results_*.csv` - Per-cohort validation metrics
- `LOCO_predictions_*.csv` - Per-sample LOCO predictions

## Model Format

The trained BTEC models are stored as CSV files with the following structure:

| Column | Description |
|--------|-------------|
| (index) | CpG probe ID or "Intercept" |
| Coefficient | Model coefficient |
| order | 1 for linear terms, 2 for quadratic terms, 0 for intercept |

## Key Findings

- **Tumor-adjacent samples** show slight negative epigenetic age acceleration (EAA ≈ -2.3 years)
- **Tumor samples** show pronounced negative EAA (-14 to -15 years on average)
- **Molecular subtype effects**: TNBC > HER2+ > HR+ in terms of negative EAA
- **No ancestry bias**: BTEC performs consistently across ancestry groups, unlike pan-tissue clocks

## Citation

If you use BTEC in your research, please cite:

```
Garma, Leonardo D., Sonia Pernas, David Vicente Baz, Rosario García Campelo, 
Josefa Terrasa, Ramón Colomer, Desirée Jiménez et al. "A breast tissue-specific 
epigenetic clock provides accurate chronological age predictions and reveals 
de-correlation of age and DNA methylation in tumor-adjacent and tumor samples." 
bioRxiv (2025): 2025-02.
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact: leonardo.garma@gmail.es