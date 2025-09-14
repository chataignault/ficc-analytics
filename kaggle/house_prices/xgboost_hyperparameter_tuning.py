#!/usr/bin/env python3
"""
XGBoost Hyperparameter Tuning Script for House Price Prediction
================================================================

This script performs comprehensive hyperparameter tuning for XGBoost
on the Kaggle House Prices dataset with RMSE as the objective metric.

Author: Claude Code
Date: 2025-09-14
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import json
from datetime import datetime
import os

class HousePriceXGBoostTuner:
    """
    A comprehensive XGBoost hyperparameter tuning class for house price prediction.
    """

    def __init__(self, data_path='train.csv', random_state=42):
        """
        Initialize the tuner with data path and random state.

        Args:
            data_path (str): Path to the training data CSV file
            random_state (int): Random state for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.preprocessor = None
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.feature_names = None

    def load_and_explore_data(self):
        """
        Load the dataset and perform initial exploration.
        """
        print("Loading and exploring data...")
        print("=" * 50)

        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Target variable: SalePrice")

        # Basic info
        print(f"\nTarget variable statistics:")
        print(self.df['SalePrice'].describe())

        # Missing values
        missing_counts = self.df.isnull().sum()
        missing_features = missing_counts[missing_counts > 0].sort_values(ascending=False)

        if len(missing_features) > 0:
            print(f"\nFeatures with missing values ({len(missing_features)} total):")
            for feature, count in missing_features.head(10).items():
                pct = (count / len(self.df)) * 100
                print(f"  {feature}: {count} ({pct:.1f}%)")

        # Data types
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features.remove('SalePrice')  # Remove target
        if 'Id' in numeric_features:
            numeric_features.remove('Id')  # Remove ID column

        categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()

        print(f"\nNumeric features: {len(numeric_features)}")
        print(f"Categorical features: {len(categorical_features)}")

        return numeric_features, categorical_features

    def preprocess_data(self, numeric_features, categorical_features):
        """
        Preprocess the data with imputation and encoding.

        Args:
            numeric_features (list): List of numeric feature names
            categorical_features (list): List of categorical feature names
        """
        print("\nPreprocessing data...")
        print("=" * 30)

        # Prepare features and target
        all_features = numeric_features + categorical_features
        self.X = self.df[all_features].copy()
        self.y = self.df['SalePrice'].copy()

        # Log transform target to handle skewness
        self.y = np.log1p(self.y)
        print(f"Applied log1p transformation to target variable")

        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', LabelEncoder())
        ])

        # Handle categorical features separately due to LabelEncoder limitations
        X_processed = self.X.copy()

        # Process numeric features
        if numeric_features:
            numeric_imputer = SimpleImputer(strategy='median')
            scaler = StandardScaler()
            X_processed[numeric_features] = scaler.fit_transform(
                numeric_imputer.fit_transform(X_processed[numeric_features])
            )

        # Process categorical features
        for feature in categorical_features:
            # Fill missing values
            X_processed[feature] = X_processed[feature].fillna('Missing')
            # Label encode
            le = LabelEncoder()
            X_processed[feature] = le.fit_transform(X_processed[feature].astype(str))

        self.X = X_processed
        self.feature_names = all_features

        print(f"Preprocessed features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")

    def feature_engineering(self):
        """
        Create additional features and perform feature selection.
        """
        print("\nPerforming feature engineering...")
        print("=" * 40)

        # Create some basic engineered features
        original_features = len(self.X.columns)

        # Total area features
        area_features = [col for col in self.X.columns if 'SF' in str(col) or 'Area' in str(col)]
        if len(area_features) >= 2:
            self.X['TotalArea'] = self.X[area_features].sum(axis=1)

        # Age features
        if 'YearBuilt' in self.X.columns and 'YrSold' in self.X.columns:
            self.X['HouseAge'] = self.X['YrSold'] - self.X['YearBuilt']

        # Quality-related features
        quality_features = [col for col in self.X.columns if 'Qual' in str(col)]
        if len(quality_features) >= 2:
            self.X['OverallQuality'] = self.X[quality_features].mean(axis=1)

        new_features = len(self.X.columns) - original_features
        print(f"Added {new_features} engineered features")
        print(f"Final feature set: {self.X.shape[1]} features")

        # Update feature names
        self.feature_names = list(self.X.columns)

    def create_train_val_split(self, test_size=0.2):
        """
        Split data into training and validation sets.

        Args:
            test_size (float): Fraction of data to use for validation
        """
        print(f"\nSplitting data (test_size={test_size})...")
        print("=" * 40)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state
        )

        print(f"Training set: {self.X_train.shape}")
        print(f"Validation set: {self.X_val.shape}")

    def baseline_model(self):
        """
        Train a baseline XGBoost model to establish initial performance.
        """
        print("\nTraining baseline XGBoost model...")
        print("=" * 45)

        # Basic XGBoost parameters
        baseline_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': self.random_state,
            'n_estimators': 100
        }

        # Train baseline model
        baseline_model = xgb.XGBRegressor(**baseline_params)
        baseline_model.fit(self.X_train, self.y_train)

        # Evaluate
        train_pred = baseline_model.predict(self.X_train)
        val_pred = baseline_model.predict(self.X_val)

        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))

        print(f"Baseline RMSE - Train: {train_rmse:.4f}, Validation: {val_rmse:.4f}")

        return val_rmse

    def hyperparameter_tuning(self, method='randomized', n_iter=100):
        """
        Perform hyperparameter tuning using GridSearch or RandomizedSearch.

        Args:
            method (str): 'grid' for GridSearchCV, 'randomized' for RandomizedSearchCV
            n_iter (int): Number of iterations for RandomizedSearch
        """
        print(f"\nPerforming hyperparameter tuning using {method} search...")
        print("=" * 60)

        # Define parameter space
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [1, 1.5, 2, 5],
            'min_child_weight': [1, 3, 5],
        }

        # Base estimator
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=self.random_state,
            n_jobs=-1
        )

        # Choose search method
        if method == 'grid':
            # Reduce parameter space for grid search
            param_grid_reduced = {
                'n_estimators': [200, 300],
                'max_depth': [4, 5, 6],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
            }
            search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid_reduced,
                cv=5,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=5,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )

        # Fit the search
        print("Starting hyperparameter search...")
        search.fit(self.X_train, self.y_train)

        # Store results
        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        self.best_score = -search.best_score_

        print(f"\nBest RMSE (CV): {self.best_score:.4f}")
        print(f"\nBest parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")

        return search

    def evaluate_final_model(self):
        """
        Evaluate the final tuned model on validation set.
        """
        print("\nEvaluating final model...")
        print("=" * 35)

        if self.best_model is None:
            print("No tuned model found. Please run hyperparameter tuning first.")
            return

        # Predictions
        train_pred = self.best_model.predict(self.X_train)
        val_pred = self.best_model.predict(self.X_val)

        # Convert back from log space
        train_pred_orig = np.expm1(train_pred)
        val_pred_orig = np.expm1(val_pred)
        y_train_orig = np.expm1(self.y_train)
        y_val_orig = np.expm1(self.y_val)

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_pred_orig))
        val_rmse = np.sqrt(mean_squared_error(y_val_orig, val_pred_orig))
        train_mae = mean_absolute_error(y_train_orig, train_pred_orig)
        val_mae = mean_absolute_error(y_val_orig, val_pred_orig)

        print(f"Final Model Performance:")
        print(f"  Train RMSE: ${train_rmse:,.2f}")
        print(f"  Validation RMSE: ${val_rmse:,.2f}")
        print(f"  Train MAE: ${train_mae:,.2f}")
        print(f"  Validation MAE: ${val_mae:,.2f}")

        # Feature importance
        feature_importance = self.best_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 Most Important Features:")
        for idx, row in feature_importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        return val_rmse, feature_importance_df

    def save_results(self, output_dir='xgboost_results'):
        """
        Save the trained model and results.

        Args:
            output_dir (str): Directory to save results
        """
        print(f"\nSaving results to {output_dir}...")
        print("=" * 40)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save best model
        if self.best_model is not None:
            model_path = os.path.join(output_dir, 'best_xgboost_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            print(f"Model saved to: {model_path}")

        # Save best parameters
        if self.best_params is not None:
            params_path = os.path.join(output_dir, 'best_parameters.json')
            with open(params_path, 'w') as f:
                json.dump(self.best_params, f, indent=2)
            print(f"Parameters saved to: {params_path}")

        # Save summary report
        report = {
            'timestamp': datetime.now().isoformat(),
            'best_rmse_cv': self.best_score if self.best_score else None,
            'best_parameters': self.best_params,
            'feature_count': len(self.feature_names) if self.feature_names else None,
            'training_samples': len(self.X_train) if self.X_train is not None else None,
            'validation_samples': len(self.X_val) if self.X_val is not None else None
        }

        report_path = os.path.join(output_dir, 'tuning_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {report_path}")

    def run_complete_pipeline(self, tuning_method='randomized', n_iter=100):
        """
        Run the complete hyperparameter tuning pipeline.

        Args:
            tuning_method (str): 'grid' or 'randomized'
            n_iter (int): Number of iterations for randomized search
        """
        print("XGBoost Hyperparameter Tuning Pipeline")
        print("=" * 50)
        print(f"Start time: {datetime.now()}")
        print()

        # Step 1: Load and explore data
        numeric_features, categorical_features = self.load_and_explore_data()

        # Step 2: Preprocess data
        self.preprocess_data(numeric_features, categorical_features)

        # Step 3: Feature engineering
        self.feature_engineering()

        # Step 4: Create train/validation split
        self.create_train_val_split()

        # Step 5: Baseline model
        baseline_rmse = self.baseline_model()

        # Step 6: Hyperparameter tuning
        search_results = self.hyperparameter_tuning(method=tuning_method, n_iter=n_iter)

        # Step 7: Final evaluation
        final_rmse, feature_importance = self.evaluate_final_model()

        # Step 8: Save results
        self.save_results()

        print(f"\nPipeline completed!")
        print(f"End time: {datetime.now()}")
        print(f"Baseline RMSE: {baseline_rmse:.4f}")
        print(f"Final RMSE: {final_rmse:,.2f}")
        print(f"Improvement: {((baseline_rmse - self.best_score) / baseline_rmse * 100):.1f}%")


def main():
    """
    Main function to run the XGBoost hyperparameter tuning.
    """
    # Initialize tuner
    tuner = HousePriceXGBoostTuner(data_path='train.csv', random_state=42)

    # Run complete pipeline
    tuner.run_complete_pipeline(
        tuning_method='randomized',  # Change to 'grid' for grid search
        n_iter=50  # Adjust based on available time/compute
    )


if __name__ == "__main__":
    main()

    