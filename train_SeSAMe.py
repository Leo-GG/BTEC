"""
Minimal version of SeSAMe model training
Trains an ElasticNet model on methylation data to predict age
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error


def resample_group(group, N, noise=False, noise_mag=2, resample=True):
    """Resample group to N representatives with optional noise"""
    n_representatives = N
    group = group.sample(n=n_representatives, replace=resample, random_state=42)
    if noise:
        noise_vals = np.random.uniform(-noise_mag, noise_mag, group.shape[0])
        group.Age = group.Age + noise_vals
    return group


def main():
    # Load data
    print("Loading data...")
    data = pd.read_parquet('data/Features_training_Sesame.parquet')
    mdata = pd.read_csv('data/merged_filtered_mdata.csv', index_col=0)
    
    # Align metadata with data
    mdata = mdata.loc[data.index.values].copy()
    mdata['Dataset'] = [i.split('_')[-1] for i in mdata.index.values]
    data = data.loc[mdata.index]
    
    print(f"Data shape: {data.shape}")
    print(f"Metadata shape: {mdata.shape}")
    
    # Resample by dataset
    print("\nResampling data...")
    resampled_mdata = mdata.groupby('Dataset', group_keys=False).apply(resample_group, 200)
    resampled_data = data.loc[resampled_mdata.index].copy()
    
    print(f"Resampled data shape: {resampled_data.shape}")
    print(f"Dataset distribution:\n{resampled_mdata.Dataset.value_counts()}")
    
    # Prepare training data
    X = resampled_data[(resampled_mdata.Dataset != 'GS0E88883') & 
                       (resampled_mdata.Dataset != 'GSE213478') &
                       (resampled_mdata.Age > 0)]
    y = resampled_mdata.Age[(resampled_mdata.Dataset != 'GS0E88883') & 
                            (resampled_mdata.Dataset != 'GSE213478') &
                            (resampled_mdata.Age > 0)]
    
    print(f"\nTraining data shape: {X.shape}")
    
    # Cross-validation to find optimal alpha
    print("\n=== Cross-Validation ===")
    print("Running ElasticNetCV to find optimal alpha...")
    regr_cv = ElasticNetCV(random_state=42,
                           l1_ratio=[.5],
                           alphas=np.logspace(-2, -4, 10),
                           eps=1e-4,
                           n_jobs=-1,
                           cv=5,
                           max_iter=5000,
                           verbose=1)
    regr_cv.fit(X, y)
    
    print(f"Optimal alpha: {regr_cv.alpha_}")
    print(f"Optimal l1_ratio: {regr_cv.l1_ratio_}")
    print(f"Number of non-zero coefficients: {np.sum(regr_cv.coef_ != 0)}")
    print(f"Intercept: {regr_cv.intercept_}")
    
    # Evaluate CV model
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    y_pred_cv = regr_cv.predict(X_test)
    rmse_cv = np.sqrt(mean_squared_error(y_test, y_pred_cv))
    corr_cv = np.corrcoef(y_test, y_pred_cv)[0, 1]
    
    print(f"CV Test RMSE: {rmse_cv:.4f}")
    print(f"CV Test correlation: {corr_cv:.4f}")
    
    # Leave-One-Cohort-Out (LOCO) validation
    print("\n=== Leave-One-Cohort-Out Validation ===")
    sel_alpha = 0.0013
    loco_results = pd.DataFrame(columns=['Dataset', 'N', 'RMSE', 'MAE', 'r'])
    
    for dataset in mdata.Dataset.unique():
        print(f"\nLOCO - Testing on: {dataset}")
        
        # Train on all datasets except current one
        X_train_loco = data[mdata.Dataset != dataset]
        y_train_loco = mdata.Age[mdata.Dataset != dataset]
        
        # Test on current dataset
        X_test_loco = data[mdata.Dataset == dataset]
        y_test_loco = mdata[mdata.Dataset == dataset].Age
        
        # Train model
        regr_loco = ElasticNet(random_state=42,
                               l1_ratio=0.5,
                               alpha=sel_alpha,
                               max_iter=10000)
        regr_loco.fit(X_train_loco, y_train_loco)
        
        # Predict and evaluate
        y_pred_loco = regr_loco.predict(X_test_loco)
        n_samples = len(y_test_loco)
        rmse_loco = np.sqrt(mean_squared_error(y_pred_loco, y_test_loco))
        mae_loco = median_absolute_error(y_pred_loco, y_test_loco)
        corr_loco = np.corrcoef(y_pred_loco, y_test_loco)[0, 1]
        
        loco_results.loc[len(loco_results)] = [dataset, n_samples, rmse_loco, mae_loco, corr_loco]
        
        print(f"  N={n_samples}, RMSE={rmse_loco:.4f}, MAE={mae_loco:.4f}, r={corr_loco:.4f}")
    
    print("\nLOCO Results Summary:")
    print(loco_results)
    loco_results.to_csv('LOOC_results_Sesame.csv', index=False)
    print("Saved LOCO results to LOOC_results_Sesame.csv")
    
    # Train final model on full resampled data
    print("\n=== Training Final Model ===")
    print("Training ElasticNet model on full resampled dataset...")
    regr = ElasticNet(random_state=42,
                      l1_ratio=0.5,
                      alpha=sel_alpha,
                      max_iter=5000)
    regr.fit(X, y)
    
    print(f"Number of non-zero coefficients: {np.sum(regr.coef_ != 0)}")
    print(f"Intercept: {regr.intercept_}")
    
    # Save model coefficients
    print("\nSaving model coefficients...")
    var_clock = pd.DataFrame(regr.coef_, columns=['Coefficient'], 
                            index=regr.feature_names_in_)
    var_clock['order'] = 1
    
    # Mark second-order features
    var_clock.loc[[i.split('_')[-1] == '2' for i in var_clock.index], 'order'] = 2
    
    # Clean feature names
    var_clock.index = [i if i.split('_')[-1] != '2' else i.split('_')[0] 
                      for i in var_clock.index]
    
    # Keep only non-zero coefficients
    var_clock = var_clock[var_clock.Coefficient != 0]
    
    print(f"Saving {var_clock.shape[0]} non-zero coefficients to VAR_model_Sesame.csv")
    var_clock.to_csv('VAR_model_Sesame.csv')
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
