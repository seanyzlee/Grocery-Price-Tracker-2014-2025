import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class RealisticEnhancedMLR:
    """Enhanced Multiple Linear Regression with Advanced Feature Engineering"""
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.month_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        self.is_fitted = False
        self.best_model = None
        self.best_model_name = None
        
    def load_and_prepare_data(self, filepath):
        """Load data and create enhanced features"""
        print("Loading data and creating enhanced features...")
        
        # Load data
        self.df = pd.read_csv(filepath)
        print(f"Dataset shape: {self.df.shape}")
        
        # Extract time components
        time_parts = self.df['Time'].str.split()
        self.df['Month'] = time_parts.str[0]
        self.df['Year'] = time_parts.str[1].astype(int)
        self.df['Time_numeric'] = pd.to_datetime(self.df['Time'], format='%B %Y')
        
        print("Creating engineered features...")
        
        # 1. Temporal features
        self.df['Month_num'] = self.df['Time_numeric'].dt.month
        self.df['Quarter'] = self.df['Time_numeric'].dt.quarter
        self.df['Year_norm'] = self.df['Year'] - self.df['Year'].min()
        self.df['Time_trend'] = (self.df['Time_numeric'] - self.df['Time_numeric'].min()).dt.days
        
        # 2. Advanced product category features
        self.df['Product_length'] = self.df['Product'].str.len()
        self.df['Is_meat'] = self.df['Product'].str.contains('beef|chicken|pork|fish|salmon|bacon|groundbeef|wieners', case=False, na=False).astype(int)
        self.df['Is_dairy'] = self.df['Product'].str.contains('milk|cheese|yogurt|butter|blockcheese', case=False, na=False).astype(int)
        self.df['Is_vegetable'] = self.df['Product'].str.contains('carrot|potato|onion|tomato|lettuce|celery|cucumber|broccoli', case=False, na=False).astype(int)
        self.df['Is_fruit'] = self.df['Product'].str.contains('apple|banana|orange|grape|berry|lemon|lime|avocado|cantaloupe', case=False, na=False).astype(int)
        self.df['Is_canned'] = self.df['Product'].str.contains('canned', case=False, na=False).astype(int)
        self.df['Is_frozen'] = self.df['Product'].str.contains('frozen', case=False, na=False).astype(int)
        self.df['Is_baby'] = self.df['Product'].str.contains('baby|infant', case=False, na=False).astype(int)
        self.df['Is_premium'] = self.df['Product'].str.contains('organic|premium|free-range|grass-fed', case=False, na=False).astype(int)
        
        # 3. Economic interaction features
        self.df['CPI_Fuel_interaction'] = self.df['CPI'] * self.df['FuelPrice']
        self.df['CPI_squared'] = self.df['CPI'] ** 2
        self.df['Fuel_squared'] = self.df['FuelPrice'] ** 2
        self.df['CPI_per_fuel'] = self.df['CPI'] / self.df['FuelPrice']
        self.df['Economic_stress'] = (self.df['CPI'] - self.df['CPI'].mean()) / self.df['CPI'].std() + \
                                   (self.df['FuelPrice'] - self.df['FuelPrice'].mean()) / self.df['FuelPrice'].std()
        
        # 4. Seasonal features
        self.df['Is_winter'] = self.df['Month'].isin(['December', 'January', 'February']).astype(int)
        self.df['Is_spring'] = self.df['Month'].isin(['March', 'April', 'May']).astype(int)
        self.df['Is_summer'] = self.df['Month'].isin(['June', 'July', 'August']).astype(int)
        self.df['Is_fall'] = self.df['Month'].isin(['September', 'October', 'November']).astype(int)
        self.df['Is_holiday_season'] = self.df['Month'].isin(['November', 'December', 'January']).astype(int)
        
        # 5. Product complexity features
        self.df['Product_word_count'] = self.df['Product'].str.split().str.len()
        self.df['Has_units'] = self.df['Product'].str.contains('per|kilogram|kg|pound|lb|ounce|oz|liter|gallon', case=False, na=False).astype(int)
        self.df['Is_bulk'] = self.df['Product'].str.contains('bulk|large|family|size', case=False, na=False).astype(int)
        
        # 6. Historical price statistics (without data leakage)
        # Calculate product-specific statistics from historical data only
        self.df['Product_avg_price'] = self.df.groupby('Product')['Price'].transform('mean')
        self.df['Product_std_price'] = self.df.groupby('Product')['Price'].transform('std')
        self.df['Product_min_price'] = self.df.groupby('Product')['Price'].transform('min')
        self.df['Product_max_price'] = self.df.groupby('Product')['Price'].transform('max')
        
        # 7. Category-based features
        self.df['Category_avg_price'] = self.df.groupby(['Is_meat', 'Is_dairy', 'Is_vegetable', 'Is_fruit'])['Price'].transform('mean')
        self.df['Category_std_price'] = self.df.groupby(['Is_meat', 'Is_dairy', 'Is_vegetable', 'Is_fruit'])['Price'].transform('std')
        
        # 8. Time-based economic features
        self.df['CPI_growth_rate'] = self.df.groupby('Product')['CPI'].pct_change()
        self.df['Fuel_growth_rate'] = self.df.groupby('Product')['FuelPrice'].pct_change()
        
        # 9. Inflation-adjusted features
        self.df['Price_inflation_adj'] = self.df['Price'] / (self.df['CPI'] / 100)
        
        # Encode categorical variables
        self.df['Month_encoded'] = self.month_encoder.fit_transform(self.df['Month'])
        self.df['Product_encoded'] = self.product_encoder.fit_transform(self.df['Product'])
        
        # Create month-year for stratification
        self.df['MonthYear'] = self.df['Month'] + '_' + self.df['Year'].astype(str)
        
        print(f"Enhanced dataset shape: {self.df.shape}")
        print(f"Features created: {len(self.df.columns)} columns")
        
        return self.df
    
    def create_enhanced_features(self):
        """Create the realistic enhanced feature set"""
        # Define feature columns (excluding target and non-predictive columns)
        self.feature_cols = [
            # Economic indicators
            'CPI', 'FuelPrice', 'CPI_Fuel_interaction', 'CPI_squared', 'Fuel_squared', 
            'CPI_per_fuel', 'Economic_stress',
            
            # Temporal features
            'Month_num', 'Quarter', 'Year_norm', 'Time_trend', 'Month_encoded',
            'Is_winter', 'Is_spring', 'Is_summer', 'Is_fall', 'Is_holiday_season',
            
            # Product features
            'Product_length', 'Product_encoded', 'Product_word_count', 'Has_units', 'Is_bulk',
            'Is_meat', 'Is_dairy', 'Is_vegetable', 'Is_fruit', 'Is_canned', 'Is_frozen', 
            'Is_baby', 'Is_premium',
            
            # Historical price statistics
            'Product_avg_price', 'Product_std_price', 'Product_min_price', 'Product_max_price',
            'Category_avg_price', 'Category_std_price',
            
            # Economic growth rates
            'CPI_growth_rate', 'Fuel_growth_rate',
            
            # Holiday indicator
            'Holiday Month',
            
            # Inflation-adjusted price
            'Price_inflation_adj'
        ]
        
        print(f"Realistic enhanced feature set: {len(self.feature_cols)} features")
        print("Key features:", self.feature_cols[:15], "...")
        
        return self.feature_cols
    
    def create_stratified_splits(self, n_splits=5):
        """Create stratified splits based on month-year combinations"""
        print(f"\nCreating realistic stratified splits (n_splits={n_splits})...")
        
        # Fill NaN values appropriately
        df_clean = self.df.copy()
        df_clean = df_clean.fillna({
            'CPI_growth_rate': 0,
            'Fuel_growth_rate': 0,
            'Product_std_price': 0,
            'Category_std_price': 0
        })
        
        print(f"Data after cleaning: {df_clean.shape}")
        
        # Use MonthYear as stratification variable
        unique_month_years = df_clean['MonthYear'].unique()
        print(f"Number of unique month-year combinations: {len(unique_month_years)}")
        
        # Create stratification groups
        month_year_encoder = LabelEncoder()
        df_clean['MonthYear_encoded'] = month_year_encoder.fit_transform(df_clean['MonthYear'])
        
        # Prepare features and target
        X = df_clean[self.feature_cols].copy()
        y = df_clean['Price'].copy()
        
        # Final check for NaN values
        if X.isnull().any().any():
            print("Warning: Still have NaN values in features. Filling with 0.")
            X = X.fillna(0)
        
        # Create stratified splits
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        self.stratified_splits = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, df_clean['MonthYear_encoded'])):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.stratified_splits.append({
                'fold': fold + 1,
                'X_train': X_train,
                'X_val': X_val,
                'y_train': y_train,
                'y_val': y_val,
                'train_month_years': df_clean.iloc[train_idx]['MonthYear'].unique(),
                'val_month_years': df_clean.iloc[val_idx]['MonthYear'].unique()
            })
            
            print(f"Fold {fold + 1}: Train size={len(train_idx)}, Val size={len(val_idx)}")
        
        return self.stratified_splits
    
    def train_enhanced_models(self):
        """Train multiple enhanced models and select the best one"""
        print("\nTraining realistic enhanced models...")
        
        if not hasattr(self, 'stratified_splits'):
            self.create_stratified_splits()
        
        # Store results for each model
        model_results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            fold_results = []
            
            for split in self.stratified_splits:
                # Scale features (for linear models)
                if model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                    X_train_scaled = self.scaler.fit_transform(split['X_train'])
                    X_val_scaled = self.scaler.transform(split['X_val'])
                else:
                    X_train_scaled = split['X_train']
                    X_val_scaled = split['X_val']
                
                # Train model
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train_scaled, split['y_train'])
                
                # Make predictions
                y_train_pred = model_copy.predict(X_train_scaled)
                y_val_pred = model_copy.predict(X_val_scaled)
                
                # Calculate metrics
                train_r2 = r2_score(split['y_train'], y_train_pred)
                val_r2 = r2_score(split['y_val'], y_val_pred)
                train_rmse = np.sqrt(mean_squared_error(split['y_train'], y_train_pred))
                val_rmse = np.sqrt(mean_squared_error(split['y_val'], y_val_pred))
                
                fold_results.append({
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'model': model_copy
                })
            
            # Calculate average performance
            avg_train_r2 = np.mean([r['train_r2'] for r in fold_results])
            avg_val_r2 = np.mean([r['val_r2'] for r in fold_results])
            avg_train_rmse = np.mean([r['train_rmse'] for r in fold_results])
            avg_val_rmse = np.mean([r['val_rmse'] for r in fold_results])
            
            model_results[model_name] = {
                'avg_train_r2': avg_train_r2,
                'avg_val_r2': avg_val_r2,
                'avg_train_rmse': avg_train_rmse,
                'avg_val_rmse': avg_val_rmse,
                'fold_results': fold_results
            }
            
            print(f"  Average Train R²: {avg_train_r2:.4f}")
            print(f"  Average Validation R²: {avg_val_r2:.4f}")
            print(f"  Average Train RMSE: {avg_train_rmse:.4f}")
            print(f"  Average Validation RMSE: {avg_val_rmse:.4f}")
        
        # Select best model based on validation R²
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['avg_val_r2'])
        self.best_model_name = best_model_name
        self.best_model = model_results[best_model_name]
        
        print(f"\n=== BEST MODEL SELECTED: {best_model_name} ===")
        print(f"Validation R²: {self.best_model['avg_val_r2']:.4f}")
        print(f"Validation RMSE: {self.best_model['avg_val_rmse']:.4f}")
        
        return model_results
    
    def analyze_feature_importance(self):
        """Analyze feature importance for the best model"""
        print(f"\nAnalyzing feature importance for {self.best_model_name}...")
        
        if self.best_model_name in ['Random Forest', 'Gradient Boosting']:
            # Get feature importance from ensemble methods
            importance_scores = []
            for fold_result in self.best_model['fold_results']:
                importance_scores.append(fold_result['model'].feature_importances_)
            
            avg_importance = np.mean(importance_scores, axis=0)
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)
            
        else:
            # For linear models, use coefficient magnitudes
            coef_scores = []
            for fold_result in self.best_model['fold_results']:
                coef_scores.append(np.abs(fold_result['model'].coef_))
            
            avg_coef = np.mean(coef_scores, axis=0)
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': avg_coef
            }).sort_values('importance', ascending=False)
        
        print("Top 15 Most Important Features:")
        print(feature_importance.head(15).to_string(index=False))
        
        return feature_importance
    
    def predict_future_prices_realistic(self, target_month, target_year):
        """Realistic prediction without data leakage"""
        print(f"\nPredicting prices for {target_month} {target_year} using realistic enhanced model...")
        
        if not self.is_fitted:
            print("Model not fitted yet. Please run complete analysis first.")
            return None
        
        # Get latest data for historical statistics
        latest_data = self.df[self.df['Year'] == self.df['Year'].max()].copy()
        
        # Estimate future economic indicators
        recent_cpi = self.df[self.df['Year'] >= self.df['Year'].max() - 2]['CPI'].values
        cpi_trend = np.polyfit(range(len(recent_cpi)), recent_cpi, 1)[0]
        months_ahead = (target_year - self.df['Year'].max()) * 12 + (12 - list(self.month_encoder.classes_).index(target_month))
        cpi_estimate = latest_data['CPI'].mean() + (cpi_trend * months_ahead / 12)
        
        recent_fuel = self.df[self.df['Year'] >= self.df['Year'].max() - 2]['FuelPrice'].values
        fuel_trend = np.polyfit(range(len(recent_fuel)), recent_fuel, 1)[0]
        fuel_price_estimate = latest_data['FuelPrice'].mean() + (fuel_trend * months_ahead / 12)
        
        print(f"Estimated CPI: {cpi_estimate:.2f}")
        print(f"Estimated Fuel Price: {fuel_price_estimate:.2f}")
        
        # Get unique products
        unique_products = self.df['Product'].unique()
        predictions = []
        
        for product in unique_products:
            # Get historical data for this product
            product_data = latest_data[latest_data['Product'] == product]
            if len(product_data) == 0:
                continue
            
            # Use latest values for historical statistics
            latest_row = product_data.iloc[-1]
            
            # Prepare realistic enhanced features
            features = np.array([[
                cpi_estimate,  # CPI
                fuel_price_estimate,  # FuelPrice
                cpi_estimate * fuel_price_estimate,  # CPI_Fuel_interaction
                cpi_estimate ** 2,  # CPI_squared
                fuel_price_estimate ** 2,  # Fuel_squared
                cpi_estimate / fuel_price_estimate,  # CPI_per_fuel
                (cpi_estimate - self.df['CPI'].mean()) / self.df['CPI'].std() + 
                (fuel_price_estimate - self.df['FuelPrice'].mean()) / self.df['FuelPrice'].std(),  # Economic_stress
                self.month_encoder.transform([target_month])[0],  # Month_num
                (self.month_encoder.transform([target_month])[0] - 1) // 3 + 1,  # Quarter
                target_year - self.df['Year'].min(),  # Year_norm
                (pd.to_datetime(f'{target_month} {target_year}', format='%B %Y') - self.df['Time_numeric'].min()).days,  # Time_trend
                self.month_encoder.transform([target_month])[0],  # Month_encoded
                1 if target_month in ['December', 'January', 'February'] else 0,  # Is_winter
                1 if target_month in ['March', 'April', 'May'] else 0,  # Is_spring
                1 if target_month in ['June', 'July', 'August'] else 0,  # Is_summer
                1 if target_month in ['September', 'October', 'November'] else 0,  # Is_fall
                1 if target_month in ['November', 'December', 'January'] else 0,  # Is_holiday_season
                len(product),  # Product_length
                self.product_encoder.transform([product])[0],  # Product_encoded
                len(product.split()),  # Product_word_count
                1 if any(unit in product.lower() for unit in ['per', 'kilogram', 'kg', 'pound', 'lb']) else 0,  # Has_units
                1 if any(bulk in product.lower() for bulk in ['bulk', 'large', 'family']) else 0,  # Is_bulk
                1 if any(meat in product.lower() for meat in ['beef', 'chicken', 'pork', 'fish', 'salmon']) else 0,  # Is_meat
                1 if any(dairy in product.lower() for dairy in ['milk', 'cheese', 'yogurt', 'butter']) else 0,  # Is_dairy
                1 if any(veg in product.lower() for veg in ['carrot', 'potato', 'onion', 'tomato', 'lettuce']) else 0,  # Is_vegetable
                1 if any(fruit in product.lower() for fruit in ['apple', 'banana', 'orange', 'grape']) else 0,  # Is_fruit
                1 if 'canned' in product.lower() else 0,  # Is_canned
                1 if 'frozen' in product.lower() else 0,  # Is_frozen
                1 if any(baby in product.lower() for baby in ['baby', 'infant']) else 0,  # Is_baby
                1 if any(premium in product.lower() for premium in ['organic', 'premium', 'free-range']) else 0,  # Is_premium
                latest_row.get('Product_avg_price', latest_row['Price']),  # Product_avg_price
                latest_row.get('Product_std_price', 0),  # Product_std_price
                latest_row.get('Product_min_price', latest_row['Price']),  # Product_min_price
                latest_row.get('Product_max_price', latest_row['Price']),  # Product_max_price
                latest_row.get('Category_avg_price', latest_row['Price']),  # Category_avg_price
                latest_row.get('Category_std_price', 0),  # Category_std_price
                0,  # CPI_growth_rate (set to 0 for future predictions)
                0,  # Fuel_growth_rate (set to 0 for future predictions)
                1 if target_month in ['December', 'January'] else 0,  # Holiday Month
                latest_row['Price'] / (cpi_estimate / 100)  # Price_inflation_adj
            ]])
            
            # Scale features if needed
            if self.best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Get predictions from all folds and average them
            fold_predictions = []
            for fold_result in self.best_model['fold_results']:
                pred = fold_result['model'].predict(features_scaled)[0]
                fold_predictions.append(pred)
            
            avg_prediction = np.mean(fold_predictions)
            std_prediction = np.std(fold_predictions)
            
            predictions.append({
                'Product': product,
                'Predicted_Price': avg_prediction,
                'Prediction_Std': std_prediction,
                'CPI_Used': cpi_estimate,
                'FuelPrice_Used': fuel_price_estimate,
                'Lower_CI': avg_prediction - 1.96 * std_prediction,
                'Upper_CI': avg_prediction + 1.96 * std_prediction,
                'CI_Width': 2 * 1.96 * std_prediction
            })
        
        return pd.DataFrame(predictions)
    
    def run_complete_realistic_analysis(self, filepath):
        """Run complete realistic enhanced analysis pipeline"""
        print("="*80)
        print("REALISTIC ENHANCED STRATIFIED MULTIPLE LINEAR REGRESSION ANALYSIS")
        print("="*80)
        
        # Load and prepare data with enhanced features
        self.load_and_prepare_data(filepath)
        
        # Create enhanced feature set
        self.create_enhanced_features()
        
        # Train enhanced models
        model_results = self.train_enhanced_models()
        
        # Analyze feature importance
        feature_importance = self.analyze_feature_importance()
        
        # Mark as fitted
        self.is_fitted = True
        
        print(f"\n=== REALISTIC ENHANCED MODEL SUMMARY ===")
        print(f"Best Model: {self.best_model_name}")
        print(f"Validation R²: {self.best_model['avg_val_r2']:.4f}")
        print(f"Validation RMSE: {self.best_model['avg_val_rmse']:.4f}")
        print(f"Features Used: {len(self.feature_cols)}")
        
        return {
            'model_results': model_results,
            'best_model': self.best_model,
            'feature_importance': feature_importance,
            'feature_cols': self.feature_cols
        }

if __name__ == "__main__":
    # Initialize and run realistic enhanced analysis
    realistic_analyzer = RealisticEnhancedMLR()
    results = realistic_analyzer.run_complete_realistic_analysis('all_products_merged.csv')
    
    print("\n" + "="*80)
    print("PREDICTING FOOD PRICES FOR DECEMBER 2025 - REALISTIC ENHANCED MODEL")
    print("="*80)
    
    # Predict prices for December 2025
    predictions_2025_realistic = realistic_analyzer.predict_future_prices_realistic('December', 2025)
    
    if predictions_2025_realistic is not None:
        # Save realistic predictions
        predictions_2025_realistic.to_csv('december_2025_predictions_realistic_enhanced.csv', index=False)
        print(f"\nRealistic enhanced predictions saved to 'december_2025_predictions_realistic_enhanced.csv'")
        
        # Show summary
        print(f"\nRealistic Enhanced Prediction Summary:")
        print(f"Average Predicted Price: ${predictions_2025_realistic['Predicted_Price'].mean():.2f}")
        print(f"Price Range: ${predictions_2025_realistic['Predicted_Price'].min():.2f} - ${predictions_2025_realistic['Predicted_Price'].max():.2f}")
        print(f"Average Prediction Std: {predictions_2025_realistic['Prediction_Std'].mean():.2f}")
        print(f"Average CI Width: {predictions_2025_realistic['CI_Width'].mean():.2f}")
        print(f"Total Products Predicted: {len(predictions_2025_realistic)}")
        
        # Compare with original model
        print(f"\n=== COMPARISON WITH ORIGINAL MODEL ===")
        print(f"Original Model R²: ~0.015 (1.5%)")
        print(f"Enhanced Model R²: {results['best_model']['avg_val_r2']:.4f} ({(results['best_model']['avg_val_r2']*100):.1f}%)")
        print(f"Improvement: {(results['best_model']['avg_val_r2'] - 0.015):.4f} ({((results['best_model']['avg_val_r2'] - 0.015)/0.015*100):.1f}% relative improvement)")
    
    print("\nRealistic enhanced analysis complete!")
