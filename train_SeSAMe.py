"""
Minimal version of SeSAMe model training
Trains an ElasticNet model on methylation data to predict age
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error


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
    
    # Prepare training data
    X = data[(mdata.Age > 0)]
    y = mdata.Age[(mdata.Age > 0)]
    
    print(f"\nTraining data shape: {X.shape}")
    
    # Cross-validation to find optimal alpha
    print("\n=== Cross-Validation ===")
    print("Running ElasticNetCV to find optimal alpha...")
    regr_cv = ElasticNetCV(random_state=42,
                           l1_ratio=[.5],
                           alphas=np.logspace(1, -3, 10),
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
    sel_alpha = 0.0077 # Rounded from 0.007742636826811277 regr_cv.alpha_
    loco_results = pd.DataFrame(columns=['Dataset', 'N', 'RMSE', 'MAE', 'r'])
    loco_predictions = pd.DataFrame(columns=['Sample', 'Dataset', 'Age', 'Predicted_Age'])
    
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
        
        # Store predictions
        pred_df = pd.DataFrame({
            'Sample': X_test_loco.index,
            'Dataset': dataset,
            'Age': y_test_loco.values,
            'Predicted_Age': y_pred_loco
        })
        loco_predictions = pd.concat([loco_predictions, pred_df], ignore_index=True)
        
        print(f"  N={n_samples}, RMSE={rmse_loco:.4f}, MAE={mae_loco:.4f}, r={corr_loco:.4f}")
    
    print("\nLOCO Results Summary:")
    print(loco_results)
    loco_results.to_csv('LOCO_results_Sesame.csv', index=False)
    print("Saved LOCO results to LOCO_results_Sesame.csv")
    
    loco_predictions.to_csv('LOCO_predictions_Sesame.csv', index=False)
    print(f"Saved {len(loco_predictions)} predictions to LOCO_predictions_Sesame.csv")
    
    # Train final model on full data
    print("\n=== Training Final Model ===")
    print("Training ElasticNet model on full dataset...")
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
    
    # Add intercept as a row
    var_clock.loc['Intercept'] = [regr.intercept_, 0]
    
    print(f"Saving {var_clock.shape[0] - 1} non-zero coefficients + intercept to BTEC_Sesame.csv")
    var_clock.to_csv('BTEC_Sesame.csv')
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
