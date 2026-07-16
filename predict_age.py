"""
Epigenetic Age Prediction Script
================================
Predicts epigenetic age using multiple clocks:
- BTEC_SeSAMe: Custom clock trained on SeSAMe-normalized data
- BTEC_BMIQ: Custom clock trained on BMIQ-normalized data
- Horvath2013: Pan-tissue clock (Horvath, 2013)
- Hannum: Blood-based clock (Hannum et al., 2013)
- DNAmPhenoAge: Phenotypic age clock (Levine et al., 2018)
- AltumAge: Deep learning clock (de Lima Camillo et al., 2022)

Usage:
    python predict_age.py --betas betas.parquet --metadata metadata.csv --output predictions.csv
    python predict_age.py --betas betas.parquet --output predictions.csv  # Without metadata
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, median_absolute_error
import pyaging as pya
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# BTEC CLOCK PREDICTION FUNCTION
# =============================================================================

def predict_btec(clock_df, intercept, betas_df):
    """
    Predict age using BTEC clock (supports first and second order terms).
    
    Parameters:
    -----------
    clock_df : pd.DataFrame
        Clock coefficients with 'Coefficient' and 'order' columns
    intercept : float
        Model intercept
    betas_df : pd.DataFrame
        Beta values matrix (probes x samples)
    
    Returns:
    --------
    pd.Series
        Predicted ages for each sample
    """
    # First order terms
    betas_1 = betas_df.merge(
        clock_df.loc[clock_df.order == 1][['Coefficient']], 
        left_index=True, 
        right_index=True
    )
    
    # Second order terms
    betas_2 = betas_df.merge(
        clock_df.loc[clock_df.order == 2][['Coefficient']], 
        left_index=True, 
        right_index=True
    )
    
    # Calculate predictions
    # First order: sum(beta * coefficient)
    # Second order: sum(beta^2 * coefficient)
    age_pred = (
        betas_1.iloc[:, :-1].T.fillna(0).dot(betas_1[['Coefficient']]) +
        betas_2.iloc[:, :-1].T.fillna(0).pow(2).dot(betas_2[['Coefficient']]) +
        intercept
    )
    
    return age_pred.iloc[:, 0]


# =============================================================================
# PYAGING CLOCK PREDICTIONS
# =============================================================================

def predict_pyaging_clocks(betas_df, clock_names):
    """
    Predict age using pyaging library clocks.
    
    Parameters:
    -----------
    betas_df : pd.DataFrame
        Beta values matrix (probes x samples)
    clock_names : list
        List of pyaging clock names to use
    
    Returns:
    --------
    pd.DataFrame
        Predictions for each clock
    """
    # Transpose for pyaging (samples x probes)
    df_transposed = betas_df.T
    
    # Create temporary AnnData object for pyaging
    import anndata as ad
    adata = ad.AnnData(df_transposed)
    adata.var_names = betas_df.index.tolist()
    adata.obs_names = betas_df.columns.tolist()
    
    results = pd.DataFrame(index=betas_df.columns)
    
    for clock_name in clock_names:
        print(f"  Running {clock_name}...")
        try:
            # Predict using pyaging
            pya.pred.predict_age(adata, clock_name, verbose=False)
            results[clock_name] = adata.obs[clock_name].values
        except Exception as e:
            print(f"    WARNING: {clock_name} failed - {str(e)}")
            results[clock_name] = np.nan
    
    return results


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_predictions(predictions_df, age_column='Age'):
    """
    Calculate evaluation metrics for each clock.
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with predictions and actual age
    age_column : str
        Name of the column with actual ages
    
    Returns:
    --------
    pd.DataFrame
        Metrics for each clock
    """
    actual = predictions_df[age_column]
    clock_columns = [c for c in predictions_df.columns if c not in [age_column, 'Dataset', 'Sample']]
    
    metrics = []
    for clock in clock_columns:
        pred = predictions_df[clock]
        valid = ~(pred.isna() | actual.isna())
        
        if valid.sum() < 2:
            metrics.append({
                'Clock': clock,
                'N': valid.sum(),
                'RMSE': np.nan,
                'MAE': np.nan,
                'Correlation': np.nan
            })
            continue
        
        rmse = np.sqrt(mean_squared_error(actual[valid], pred[valid]))
        mae = median_absolute_error(actual[valid], pred[valid])
        corr = np.corrcoef(actual[valid], pred[valid])[0, 1]
        
        metrics.append({
            'Clock': clock,
            'N': valid.sum(),
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'Correlation': round(corr, 4)
        })
    
    return pd.DataFrame(metrics)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Predict epigenetic age using multiple clocks'
    )
    parser.add_argument(
        '--betas', 
        type=str, 
        required=True,
        help='Path to parquet file with beta values (probes x samples)'
    )
    parser.add_argument(
        '--metadata', 
        type=str, 
        default=None,
        help='Path to CSV file with metadata (must have "Age" column for evaluation)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='age_predictions.csv',
        help='Output file for predictions (default: age_predictions.csv)'
    )
    parser.add_argument(
        '--btec-sesame-model',
        type=str,
        default='BTEC_Sesame.csv',
        help='Path to BTEC SeSAMe model coefficients'
    )
    parser.add_argument(
        '--btec-bmiq-model',
        type=str,
        default='BTEC_BMIQ.csv',
        help='Path to BTEC BMIQ model coefficients'
    )
    parser.add_argument(
        '--impute',
        action='store_true',
        default=True,
        help='Impute missing values using kNN (default: True)'
    )
    
    args = parser.parse_args()
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print("=== Loading Data ===")
    print(f"Beta values: {args.betas}")
    
    betas = pd.read_parquet(args.betas)
    print(f"  Shape: {betas.shape} (probes x samples)")
    
    # Impute missing values if requested
    if args.impute:
        n_missing = betas.isna().sum().sum()
        if n_missing > 0:
            print(f"  Imputing {n_missing} missing values using kNN...")
            imputer = KNNImputer(n_neighbors=5)
            imputed_array = imputer.fit_transform(betas)
            betas = pd.DataFrame(imputed_array, columns=betas.columns, index=betas.index)
    
    # Load metadata if provided
    metadata = None
    if args.metadata:
        print(f"Metadata: {args.metadata}")
        metadata = pd.read_csv(args.metadata, index_col=0)
        print(f"  Shape: {metadata.shape}")
        
        # Extract dataset from sample names if not present
        if 'Dataset' not in metadata.columns:
            metadata['Dataset'] = [i.split('_')[-1] for i in metadata.index.values]
    
    # -------------------------------------------------------------------------
    # BTEC Predictions
    # -------------------------------------------------------------------------
    print("\n=== BTEC Clock Predictions ===")
    
    predictions = pd.DataFrame(index=betas.columns)
    predictions.index.name = 'Sample'
    
    # BTEC SeSAMe
    try:
        print(f"  Loading BTEC_SeSAMe from {args.btec_sesame_model}...")
        clock_sesame = pd.read_csv(args.btec_sesame_model, index_col=0)
        intercept_sesame = clock_sesame.loc['Intercept', 'Coefficient']
        clock_sesame = clock_sesame.drop('Intercept')
        predictions['BTEC_SeSAMe'] = predict_btec(clock_sesame, intercept_sesame, betas)
        print(f"    Predictions: {predictions['BTEC_SeSAMe'].notna().sum()} samples")
    except FileNotFoundError:
        print(f"    WARNING: {args.btec_sesame_model} not found, skipping BTEC_SeSAMe")
    except Exception as e:
        print(f"    WARNING: BTEC_SeSAMe failed - {str(e)}")
    
    # BTEC BMIQ
    try:
        print(f"  Loading BTEC_BMIQ from {args.btec_bmiq_model}...")
        clock_bmiq = pd.read_csv(args.btec_bmiq_model, index_col=0)
        intercept_bmiq = clock_bmiq.loc['Intercept', 'Coefficient']
        clock_bmiq = clock_bmiq.drop('Intercept')
        predictions['BTEC_BMIQ'] = predict_btec(clock_bmiq, intercept_bmiq, betas)
        print(f"    Predictions: {predictions['BTEC_BMIQ'].notna().sum()} samples")
    except FileNotFoundError:
        print(f"    WARNING: {args.btec_bmiq_model} not found, skipping BTEC_BMIQ")
    except Exception as e:
        print(f"    WARNING: BTEC_BMIQ failed - {str(e)}")
    
    # -------------------------------------------------------------------------
    # PyAging Clock Predictions
    # -------------------------------------------------------------------------
    print("\n=== PyAging Clock Predictions ===")
    
    pyaging_clocks = ['horvath2013', 'hannum', 'dnamphenoage', 'altumage']
    pyaging_results = predict_pyaging_clocks(betas, pyaging_clocks)
    
    # Merge with predictions
    for clock in pyaging_clocks:
        if clock in pyaging_results.columns:
            predictions[clock] = pyaging_results[clock]
    
    # -------------------------------------------------------------------------
    # Merge with Metadata and Evaluate
    # -------------------------------------------------------------------------
    if metadata is not None:
        print("\n=== Merging with Metadata ===")
        
        # Merge predictions with metadata
        common_samples = predictions.index.intersection(metadata.index)
        print(f"  Common samples: {len(common_samples)}")
        
        predictions = predictions.loc[common_samples]
        predictions['Age'] = metadata.loc[common_samples, 'Age']
        
        if 'Dataset' in metadata.columns:
            predictions['Dataset'] = metadata.loc[common_samples, 'Dataset']
        
        # Evaluate if Age column exists
        if 'Age' in predictions.columns and predictions['Age'].notna().sum() > 0:
            print("\n=== Evaluation Metrics ===")
            metrics = evaluate_predictions(predictions, age_column='Age')
            print(metrics.to_string(index=False))
            
            # Save metrics
            metrics_file = args.output.replace('.csv', '_metrics.csv')
            metrics.to_csv(metrics_file, index=False)
            print(f"\nMetrics saved to: {metrics_file}")
    
    # -------------------------------------------------------------------------
    # Save Predictions
    # -------------------------------------------------------------------------
    print(f"\n=== Saving Predictions ===")
    predictions.to_csv(args.output)
    print(f"Predictions saved to: {args.output}")
    print(f"  Shape: {predictions.shape}")
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
