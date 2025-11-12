import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def run_basic_mlr(filepath):
    """Minimal pipeline to fit an MLR model on all_products_merged.csv"""
    print("=" * 60)
    print("BASIC MULTIPLE LINEAR REGRESSION FIT")
    print("=" * 60)

    df = pd.read_csv(filepath)
    if 'Month' not in df.columns or 'Year' not in df.columns:
        if 'Time' not in df.columns:
            raise ValueError("Dataset must include 'Time' column to derive Month and Year.")
        time_parts = df['Time'].str.split()
        df['Month'] = time_parts.str[0]
        df['Year'] = time_parts.str[1].astype(int)

    required_cols = ['Price', 'CPI', 'FuelPrice', 'Holiday Month', 'Month', 'Product']
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    month_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    df['Month_encoded'] = month_encoder.fit_transform(df['Month'])
    df['Product_encoded'] = product_encoder.fit_transform(df['Product'])

    feature_cols = ['CPI', 'FuelPrice', 'Holiday Month', 'Month_encoded', 'Product_encoded']
    X = df[feature_cols]
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

    print(f"Train R²: {train_r2:.4f}")
    print(f"Test  R²: {test_r2:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test  RMSE: {test_rmse:.4f}")
    print(f"CV R² Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    linear_model = pipeline.named_steps['model']
    coeffs = linear_model.coef_
    print("Coefficients:")
    for name, coef in zip(feature_cols, coeffs):
        print(f"  {name}: {coef:.4f}")
    print(f"Intercept: {linear_model.intercept_:.4f}")

    return {
        'pipeline': pipeline,
        'month_encoder': month_encoder,
        'product_encoder': product_encoder,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'cv_scores': cv_scores,
    }

class StratifiedMLR:
    """Multiple Linear Regression with Month-based Stratification for Food Price Prediction"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.label_encoders = {}
        self.month_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        self.is_fitted = False
        
    def load_and_prepare_data(self, filepath):
        """Load and prepare data for modeling"""
        print("Loading data...")
        
        # Load data
        self.df = pd.read_csv(filepath)
        print(f"Dataset shape: {self.df.shape}")
        
        # Extract month and year from Time column
        time_parts = self.df['Time'].str.split()
        self.df['Month'] = time_parts.str[0]
        self.df['Year'] = time_parts.str[1].astype(int)
        
        # Create month-year combination for stratification
        self.df['MonthYear'] = self.df['Month'] + '_' + self.df['Year'].astype(str)
        
        # Encode categorical variables
        self.df['Month_encoded'] = self.month_encoder.fit_transform(self.df['Month'])
        self.df['Product_encoded'] = self.product_encoder.fit_transform(self.df['Product'])
        
        print(f"Unique months: {sorted(self.df['Month'].unique())}")
        print(f"Unique products: {self.df['Product'].nunique()}")
        print(f"Date range: {self.df['Time'].min()} to {self.df['Time'].max()}")
        
        return self.df
    
    def create_stratified_splits(self, n_splits=5):
        """Create stratified splits based on month-year combinations"""
        print(f"\nCreating stratified splits (n_splits={n_splits})...")
        
        # Use MonthYear as stratification variable
        unique_month_years = self.df['MonthYear'].unique()
        print(f"Number of unique month-year combinations: {len(unique_month_years)}")
        
        # Create stratification groups
        month_year_encoder = LabelEncoder()
        self.df['MonthYear_encoded'] = month_year_encoder.fit_transform(self.df['MonthYear'])
        
        # Prepare features and target
        feature_cols = ['CPI', 'FuelPrice', 'Holiday Month', 'Month_encoded', 'Product_encoded']
        X = self.df[feature_cols].copy()
        y = self.df['Price'].copy()
        
        # Create stratified splits
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        self.stratified_splits = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, self.df['MonthYear_encoded'])):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.stratified_splits.append({
                'fold': fold + 1,
                'X_train': X_train,
                'X_val': X_val,
                'y_train': y_train,
                'y_val': y_val,
                'train_month_years': self.df.iloc[train_idx]['MonthYear'].unique(),
                'val_month_years': self.df.iloc[val_idx]['MonthYear'].unique()
            })
            
            print(f"Fold {fold + 1}: Train size={len(train_idx)}, Val size={len(val_idx)}")
            print(f"  Train month-years: {len(self.stratified_splits[-1]['train_month_years'])}")
            print(f"  Val month-years: {len(self.stratified_splits[-1]['val_month_years'])}")
        
        return self.stratified_splits
    
    def train_stratified_model(self):
        """Train MLR model using stratified approach"""
        print("\nTraining stratified MLR model...")
        
        if not hasattr(self, 'stratified_splits'):
            self.create_stratified_splits()
        
        # Store results for each fold
        fold_results = []
        
        for split in self.stratified_splits:
            print(f"\nTraining Fold {split['fold']}...")
            
            # Scale features with a fold-specific scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(split['X_train'])
            X_val_scaled = scaler.transform(split['X_val'])
            
            # Train model
            model = LinearRegression()
            model.fit(X_train_scaled, split['y_train'])
            
            # Make predictions
            y_train_pred = model.predict(X_train_scaled)
            y_val_pred = model.predict(X_val_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(split['y_train'], y_train_pred)
            val_r2 = r2_score(split['y_val'], y_val_pred)
            train_rmse = np.sqrt(mean_squared_error(split['y_train'], y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(split['y_val'], y_val_pred))
            train_mae = mean_absolute_error(split['y_train'], y_train_pred)
            val_mae = mean_absolute_error(split['y_val'], y_val_pred)
            
            fold_result = {
                'fold': split['fold'],
                'model': model,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'train_size': len(split['X_train']),
                'val_size': len(split['X_val']),
                'scaler': scaler
            }
            
            fold_results.append(fold_result)
            
            print(f"  Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")
            print(f"  Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
            print(f"  Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")
        
        self.fold_results = fold_results
        
        # Calculate average performance
        avg_train_r2 = np.mean([r['train_r2'] for r in fold_results])
        avg_val_r2 = np.mean([r['val_r2'] for r in fold_results])
        avg_train_rmse = np.mean([r['train_rmse'] for r in fold_results])
        avg_val_rmse = np.mean([r['val_rmse'] for r in fold_results])
        avg_train_mae = np.mean([r['train_mae'] for r in fold_results])
        avg_val_mae = np.mean([r['val_mae'] for r in fold_results])
        
        print(f"\n=== STRATIFIED MODEL PERFORMANCE SUMMARY ===")
        print(f"Average Train R²: {avg_train_r2:.4f}")
        print(f"Average Validation R²: {avg_val_r2:.4f}")
        print(f"Average Train RMSE: {avg_train_rmse:.4f}")
        print(f"Average Validation RMSE: {avg_val_rmse:.4f}")
        print(f"Average Train MAE: {avg_train_mae:.4f}")
        print(f"Average Validation MAE: {avg_val_mae:.4f}")
        
        return fold_results
    
    def train_non_stratified_model(self):
        """Train MLR model without stratification for comparison"""
        print("\nTraining non-stratified MLR model for comparison...")
        
        # Prepare features and target
        feature_cols = ['CPI', 'FuelPrice', 'Holiday Month', 'Month_encoded', 'Product_encoded']
        X = self.df[feature_cols]
        y = self.df['Price']
        
        # Simple train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print(f"Non-stratified model performance:")
        print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"  Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"  Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
        
        return {
            'model': model,
            'scaler': scaler,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
    
    def analyze_feature_importance(self):
        """Analyze feature importance across all folds"""
        print("\nAnalyzing feature importance...")
        
        feature_names = ['CPI', 'FuelPrice', 'Holiday Month', 'Month_encoded', 'Product_encoded']
        feature_importance = {name: [] for name in feature_names}
        
        for fold_result in self.fold_results:
            model = fold_result['model']
            coefs = model.coef_
            
            for i, name in enumerate(feature_names):
                feature_importance[name].append(abs(coefs[i]))
        
        # Calculate average importance
        avg_importance = {}
        for name in feature_names:
            avg_importance[name] = np.mean(feature_importance[name])
        
        # Sort by importance
        sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("Feature Importance (Average |Coefficient| across folds):")
        for name, importance in sorted_importance:
            print(f"  {name}: {importance:.4f}")
        
        return sorted_importance
    
    def plot_results(self):
        """Create visualization plots"""
        print("\nCreating visualization plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Performance comparison across folds
        folds = [r['fold'] for r in self.fold_results]
        train_r2s = [r['train_r2'] for r in self.fold_results]
        val_r2s = [r['val_r2'] for r in self.fold_results]
        
        axes[0, 0].plot(folds, train_r2s, 'o-', label='Train R²', color='blue')
        axes[0, 0].plot(folds, val_r2s, 's-', label='Validation R²', color='red')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('R² Score Across Folds (Stratified)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. RMSE comparison across folds
        train_rmses = [r['train_rmse'] for r in self.fold_results]
        val_rmses = [r['val_rmse'] for r in self.fold_results]
        
        axes[0, 1].plot(folds, train_rmses, 'o-', label='Train RMSE', color='blue')
        axes[0, 1].plot(folds, val_rmses, 's-', label='Validation RMSE', color='red')
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('RMSE Across Folds (Stratified)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature importance
        importance_data = self.analyze_feature_importance()
        features, importances = zip(*importance_data)
        
        axes[1, 0].bar(features, importances, color='skyblue', alpha=0.7)
        axes[1, 0].set_xlabel('Features')
        axes[1, 0].set_ylabel('Average |Coefficient|')
        axes[1, 0].set_title('Feature Importance')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Price distribution by month
        monthly_prices = self.df.groupby('Month')['Price'].mean().sort_values()
        axes[1, 1].bar(range(len(monthly_prices)), monthly_prices.values, color='lightgreen', alpha=0.7)
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Average Price')
        axes[1, 1].set_title('Average Price by Month')
        axes[1, 1].set_xticks(range(len(monthly_prices)))
        axes[1, 1].set_xticklabels(monthly_prices.index, rotation=45)
        
        plt.tight_layout()
        plt.savefig('mlr_stratified_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Plots saved as 'mlr_stratified_results.png'")
    
    def run_complete_analysis(self, filepath):
        """Run complete analysis pipeline"""
        print("="*60)
        print("STRATIFIED MULTIPLE LINEAR REGRESSION ANALYSIS")
        print("="*60)
        
        # Load and prepare data
        self.load_and_prepare_data(filepath)
        
        # Train stratified model
        stratified_results = self.train_stratified_model()
        
        # Train non-stratified model for comparison
        non_stratified_results = self.train_non_stratified_model()
        
        # Analyze feature importance
        feature_importance = self.analyze_feature_importance()
        
        # Create plots
        self.plot_results()
        
        # Summary comparison
        print("\n" + "="*60)
        print("STRATIFIED vs NON-STRATIFIED COMPARISON")
        print("="*60)
        
        avg_stratified_val_r2 = np.mean([r['val_r2'] for r in stratified_results])
        non_stratified_test_r2 = non_stratified_results['test_r2']
        
        avg_stratified_val_rmse = np.mean([r['val_rmse'] for r in stratified_results])
        non_stratified_test_rmse = non_stratified_results['test_rmse']
        
        print(f"Stratified Model - Average Validation R²: {avg_stratified_val_r2:.4f}")
        print(f"Non-Stratified Model - Test R²: {non_stratified_test_r2:.4f}")
        print(f"R² Improvement: {avg_stratified_val_r2 - non_stratified_test_r2:.4f}")
        print()
        print(f"Stratified Model - Average Validation RMSE: {avg_stratified_val_rmse:.4f}")
        print(f"Non-Stratified Model - Test RMSE: {non_stratified_test_rmse:.4f}")
        print(f"RMSE Improvement: {non_stratified_test_rmse - avg_stratified_val_rmse:.4f}")
        
        return {
            'stratified_results': stratified_results,
            'non_stratified_results': non_stratified_results,
            'feature_importance': feature_importance
        }

    def predict_future_prices(self, target_month, target_year, cpi_estimate=None, fuel_price_estimate=None):
        """
        Predict food prices for a specific future month-year
        
        Args:
            target_month (str): Target month (e.g., 'December')
            target_year (int): Target year (e.g., 2025)
            cpi_estimate (float): Estimated CPI for the target period (optional)
            fuel_price_estimate (float): Estimated fuel price for the target period (optional)
        """
        print(f"\nPredicting prices for {target_month} {target_year}...")
        
        if not self.is_fitted:
            print("Model not fitted yet. Please run complete analysis first.")
            return None
        
        # Get latest available data for trend estimation
        latest_data = self.df[self.df['Year'] == self.df['Year'].max()].copy()
        latest_month_data = latest_data[latest_data['Month'] == target_month]
        
        if len(latest_month_data) == 0:
            # If no data for target month, use average of latest year
            latest_month_data = latest_data.groupby('Product').agg({
                'CPI': 'mean',
                'FuelPrice': 'mean',
                'Holiday Month': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
            }).reset_index()
        else:
            latest_month_data = latest_month_data.groupby('Product').agg({
                'CPI': 'mean',
                'FuelPrice': 'mean',
                'Holiday Month': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
            }).reset_index()
        
        # Estimate CPI if not provided
        if cpi_estimate is None:
            # Calculate CPI trend from recent years
            recent_cpi = self.df[self.df['Year'] >= self.df['Year'].max() - 2]['CPI'].values
            cpi_trend = np.polyfit(range(len(recent_cpi)), recent_cpi, 1)[0]
            months_ahead = (target_year - self.df['Year'].max()) * 12 + (12 - list(self.month_encoder.classes_).index(target_month))
            cpi_estimate = latest_month_data['CPI'].mean() + (cpi_trend * months_ahead / 12)
            print(f"Estimated CPI for {target_month} {target_year}: {cpi_estimate:.2f}")
        
        # Estimate fuel price if not provided
        if fuel_price_estimate is None:
            # Calculate fuel price trend from recent years
            recent_fuel = self.df[self.df['Year'] >= self.df['Year'].max() - 2]['FuelPrice'].values
            fuel_trend = np.polyfit(range(len(recent_fuel)), recent_fuel, 1)[0]
            months_ahead = (target_year - self.df['Year'].max()) * 12 + (12 - list(self.month_encoder.classes_).index(target_month))
            fuel_price_estimate = latest_month_data['FuelPrice'].mean() + (fuel_trend * months_ahead / 12)
            print(f"Estimated Fuel Price for {target_month} {target_year}: {fuel_price_estimate:.2f}")
        
        # Prepare prediction data
        predictions = []
        
        for _, product_row in latest_month_data.iterrows():
            product = product_row['Product']
            
            # Get product encoding
            if product in self.product_encoder.classes_:
                product_encoded = self.product_encoder.transform([product])[0]
            else:
                # Use average encoding for unknown products
                product_encoded = self.product_encoder.transform(self.product_encoder.classes_).mean()
            
            # Prepare features
            features = np.array([[
                cpi_estimate,
                fuel_price_estimate,
                1 if target_month in ['December', 'January'] else 0,  # Holiday month assumption
                self.month_encoder.transform([target_month])[0],
                product_encoded
            ]])
            
            # Get predictions from all folds and average them
            fold_predictions = []
            for fold_result in self.fold_results:
                scaler = fold_result.get('scaler')
                if scaler is None:
                    raise ValueError("Scaler not stored for fold; re-run stratified training.")
                features_scaled = scaler.transform(features)
                pred = fold_result['model'].predict(features_scaled)[0]
                fold_predictions.append(pred)
            
            avg_prediction = np.mean(fold_predictions)
            std_prediction = np.std(fold_predictions)
            
            predictions.append({
                'Product': product,
                'Predicted_Price': avg_prediction,
                'Prediction_Std': std_prediction,
                'CPI_Used': cpi_estimate,
                'FuelPrice_Used': fuel_price_estimate
            })
        
        return pd.DataFrame(predictions)
    
    def analyze_prediction_confidence(self, predictions_df):
        """Analyze prediction confidence and provide insights"""
        print("\n=== PREDICTION CONFIDENCE ANALYSIS ===")
        
        # Calculate confidence intervals
        predictions_df['Lower_CI'] = predictions_df['Predicted_Price'] - 1.96 * predictions_df['Prediction_Std']
        predictions_df['Upper_CI'] = predictions_df['Predicted_Price'] + 1.96 * predictions_df['Prediction_Std']
        predictions_df['CI_Width'] = predictions_df['Upper_CI'] - predictions_df['Lower_CI']
        
        # Sort by prediction confidence (lower std = higher confidence)
        predictions_df_sorted = predictions_df.sort_values('Prediction_Std')
        
        print(f"Most Confident Predictions (Top 10):")
        print(predictions_df_sorted[['Product', 'Predicted_Price', 'Prediction_Std', 'CI_Width']].head(10).to_string(index=False))
        
        print(f"\nLeast Confident Predictions (Top 10):")
        print(predictions_df_sorted[['Product', 'Predicted_Price', 'Prediction_Std', 'CI_Width']].tail(10).to_string(index=False))
        
        print(f"\nOverall Prediction Statistics:")
        print(f"Average Predicted Price: ${predictions_df['Predicted_Price'].mean():.2f}")
        print(f"Price Range: ${predictions_df['Predicted_Price'].min():.2f} - ${predictions_df['Predicted_Price'].max():.2f}")
        print(f"Average Prediction Std: {predictions_df['Prediction_Std'].mean():.2f}")
        print(f"Average CI Width: {predictions_df['CI_Width'].mean():.2f}")
        
        return predictions_df_sorted

if __name__ == "__main__":
    run_basic_mlr('all_products_merged.csv')
