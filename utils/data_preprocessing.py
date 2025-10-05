import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import re
import warnings
warnings.filterwarnings('ignore')

class ExoplanetPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        self.log_transform_features = []
        
    def detect_label_column(self, df):
        """Detect the disposition/label column in the dataset"""
        label_keywords = ['dispos', 'disp', 'status', 'koi_disposition', 'tfopwg_disp']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in label_keywords):
                return col
        
        return None
    
    def standardize_labels(self, labels):
        """Convert various disposition labels to standard format"""
        standardized = []
        
        for label in labels:
            if pd.isna(label):
                standardized.append(np.nan)
                continue
                
            label_str = str(label).upper()
            
            if any(keyword in label_str for keyword in ['CONFIRMED', 'CP', 'KP']):
                standardized.append('Confirmed')
            elif any(keyword in label_str for keyword in ['CANDIDATE', 'PC', 'PC?', 'PC+']):
                standardized.append('Candidate')  
            elif any(keyword in label_str for keyword in ['FALSE', 'FP', 'REFUTED']):
                standardized.append('False Positive')
            else:
                standardized.append(np.nan)
                
        return standardized
    
    def select_features(self, df):
        """Select relevant numerical features for classification"""
        # Priority keywords for feature selection
        priority_keywords = [
            'period', 'duration', 'prad', 'depth', 'snr', 'mes', 
            'teff', 'insol', 'mag', 'flux', 'impact', 'rad', 'teq'
        ]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # First, find features matching priority keywords
        selected_features = []
        for col in numeric_cols:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in priority_keywords):
                selected_features.append(col)
        
        # If we have fewer than 6 features, add more numeric columns
        if len(selected_features) < 6:
            remaining_numeric = [col for col in numeric_cols if col not in selected_features]
            needed = min(30 - len(selected_features), len(remaining_numeric))
            selected_features.extend(remaining_numeric[:needed])
        
        # Limit to 30 features maximum
        selected_features = selected_features[:30]
        
        return selected_features
    
    def identify_log_transform_features(self, df, features):
        """Identify features that should be log-transformed due to high skewness"""
        log_features = []
        
        for feature in features:
            if feature in df.columns:
                # Check if feature is positive and highly skewed
                values = df[feature].dropna()
                if len(values) > 0 and (values > 0).all():
                    skewness = values.skew()
                    if skewness > 2:  # Highly right-skewed
                        # Common features that benefit from log transform
                        if any(keyword in feature.lower() for keyword in 
                               ['period', 'depth', 'insol', 'flux', 'duration']):
                            log_features.append(feature)
        
        return log_features
    
    def preprocess_data(self, df):
        """Complete preprocessing pipeline"""
        try:
            # Detect label column
            label_column = self.detect_label_column(df)
            if not label_column:
                raise ValueError("Could not find disposition/label column")
            
            print(f"Found label column: {label_column}")
            
            # Standardize labels
            y = self.standardize_labels(df[label_column])
            y = pd.Series(y)
            
            # Remove rows with missing labels
            valid_labels = ~y.isna()
            y = y[valid_labels]
            df_filtered = df[valid_labels].copy()
            
            print(f"Label distribution after filtering: {y.value_counts()}")
            
            # Select features
            selected_features = self.select_features(df_filtered)
            if len(selected_features) < 3:
                raise ValueError("Insufficient features found")
            
            print(f"Selected {len(selected_features)} features: {selected_features}")
            
            # Filter features and remove rows with too many missing values
            X = df_filtered[selected_features].copy()
            
            # Require at least 3 non-null features per row
            min_features = min(3, len(selected_features))
            valid_rows = X.count(axis=1) >= min_features
            X = X[valid_rows]
            y = y[valid_rows]
            
            print(f"After filtering: {len(X)} rows, {len(X.columns)} features")
            
            if len(X) < 10:
                raise ValueError("Insufficient data after preprocessing")
            
            # Identify log transform features
            self.log_transform_features = self.identify_log_transform_features(X, selected_features)
            print(f"Log transform features: {self.log_transform_features}")
            
            # Apply log transform
            for feature in self.log_transform_features:
                if feature in X.columns:
                    # Add small constant to avoid log(0)
                    X[feature] = np.log1p(X[feature].clip(lower=0))
            
            # Impute missing values
            X_imputed = self.imputer.fit_transform(X)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_imputed)
            
            # Store feature names
            self.feature_names = selected_features
            
            return X_scaled, y.values, selected_features
            
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            return None, None, None
    
    def transform_new_data(self, df, feature_names=None):
        """Transform new data using fitted preprocessor"""
        try:
            if feature_names is None:
                feature_names = self.feature_names
            
            if feature_names is None:
                raise ValueError("No feature names available. Please fit the preprocessor first.")
            
            # Check if required features exist
            missing_features = [f for f in feature_names if f not in df.columns]
            if missing_features:
                print(f"Warning: Missing features {missing_features}")
                # Create missing columns with NaN
                for feature in missing_features:
                    df[feature] = np.nan
            
            # Select and order features
            X = df[feature_names].copy()
            
            # Apply same log transforms
            for feature in self.log_transform_features:
                if feature in X.columns:
                    X[feature] = np.log1p(X[feature].clip(lower=0))
            
            # Apply imputation and scaling
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed)
            
            return X_scaled
            
        except Exception as e:
            print(f"Transform error: {str(e)}")
            return None
