import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class ExoplanetFeatureEngineer:
    def __init__(self):
        self.poly_features = None
        self.derived_feature_names = []
        self.original_feature_names = []
        
    def create_ratio_features(self, df, feature_names):
        """Create ratio features from pairs of related features"""
        ratio_features = pd.DataFrame(index=df.index)
        ratio_names = []
        
        # Define feature pairs that make sense for ratios
        ratio_pairs = [
            ('prad', 'srad'),  # Planet radius to star radius
            ('depth', 'duration'),  # Transit depth to duration
            ('period', 'duration'),  # Orbital period to transit duration
            ('insol', 'teq'),  # Insolation to equilibrium temperature
        ]
        
        for feat1_key, feat2_key in ratio_pairs:
            # Find matching features
            feat1_cols = [f for f in feature_names if feat1_key in f.lower()]
            feat2_cols = [f for f in feature_names if feat2_key in f.lower()]
            
            for f1 in feat1_cols:
                for f2 in feat2_cols:
                    if f1 in df.columns and f2 in df.columns:
                        # Avoid division by zero
                        denominator = df[f2].replace(0, np.nan)
                        ratio = df[f1] / denominator
                        
                        if not ratio.isna().all():  # Only add if not all NaN
                            ratio_name = f"{f1}_to_{f2}_ratio"
                            ratio_features[ratio_name] = ratio
                            ratio_names.append(ratio_name)
        
        return ratio_features, ratio_names
    
    def create_difference_features(self, df, feature_names):
        """Create difference features from related features"""
        diff_features = pd.DataFrame(index=df.index)
        diff_names = []
        
        # Features that make sense to subtract
        diff_keywords = ['err1', 'err2', 'upper', 'lower']
        
        # Group features by base name
        base_features = {}
        for feat in feature_names:
            base = feat.split('_err')[0].split('_upper')[0].split('_lower')[0]
            if base not in base_features:
                base_features[base] = []
            base_features[base].append(feat)
        
        # Create differences for error bounds
        for base, feats in base_features.items():
            err1_feat = [f for f in feats if 'err1' in f.lower()]
            err2_feat = [f for f in feats if 'err2' in f.lower()]
            
            if len(err1_feat) == 1 and len(err2_feat) == 1:
                if err1_feat[0] in df.columns and err2_feat[0] in df.columns:
                    diff_name = f"{base}_error_range"
                    diff_features[diff_name] = df[err1_feat[0]] - df[err2_feat[0]]
                    diff_names.append(diff_name)
        
        return diff_features, diff_names
    
    def create_interaction_features(self, df, feature_names, max_interactions=10):
        """Create multiplicative interaction features"""
        interaction_features = pd.DataFrame(index=df.index)
        interaction_names = []
        
        # Select important feature types for interactions
        important_keywords = ['period', 'depth', 'prad', 'duration', 'snr', 'teq']
        important_features = []
        
        for keyword in important_keywords:
            matching = [f for f in feature_names if keyword in f.lower() and f in df.columns]
            if matching:
                important_features.append(matching[0])  # Take first match
        
        # Limit to avoid explosion
        important_features = important_features[:5]
        
        # Create pairwise interactions
        count = 0
        for f1, f2 in combinations(important_features, 2):
            if count >= max_interactions:
                break
                
            if f1 in df.columns and f2 in df.columns:
                interaction_name = f"{f1}_x_{f2}"
                interaction_features[interaction_name] = df[f1] * df[f2]
                interaction_names.append(interaction_name)
                count += 1
        
        return interaction_features, interaction_names
    
    def create_polynomial_features(self, df, feature_names, degree=2, max_features=5):
        """Create polynomial features for top features"""
        # Select top features based on variance
        numeric_df = df[feature_names].select_dtypes(include=[np.number])
        
        # Calculate variance and select top features
        variances = numeric_df.var()
        top_features = variances.nlargest(min(max_features, len(variances))).index.tolist()
        
        if len(top_features) == 0:
            return pd.DataFrame(index=df.index), []
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        
        poly_array = poly.fit_transform(df[top_features].fillna(0))
        poly_feature_names = poly.get_feature_names_out(top_features)
        
        # Remove original features (they're already in the dataset)
        new_features_idx = [i for i, name in enumerate(poly_feature_names) 
                          if name not in top_features]
        
        poly_df = pd.DataFrame(
            poly_array[:, new_features_idx],
            columns=[poly_feature_names[i] for i in new_features_idx],
            index=df.index
        )
        
        self.poly_features = poly
        
        return poly_df, poly_df.columns.tolist()
    
    def engineer_features(self, df, feature_names, include_ratios=True, include_differences=True, 
                         include_interactions=True, include_polynomial=False):
        """Apply all feature engineering techniques"""
        engineered_df = df[feature_names].copy()
        new_feature_names = feature_names.copy()
        self.original_feature_names = feature_names.copy()
        
        print(f"Starting with {len(feature_names)} original features")
        
        # Create ratio features
        if include_ratios:
            ratio_df, ratio_names = self.create_ratio_features(df, feature_names)
            if len(ratio_names) > 0:
                engineered_df = pd.concat([engineered_df, ratio_df], axis=1)
                new_feature_names.extend(ratio_names)
                print(f"Added {len(ratio_names)} ratio features")
        
        # Create difference features
        if include_differences:
            diff_df, diff_names = self.create_difference_features(df, feature_names)
            if len(diff_names) > 0:
                engineered_df = pd.concat([engineered_df, diff_df], axis=1)
                new_feature_names.extend(diff_names)
                print(f"Added {len(diff_names)} difference features")
        
        # Create interaction features
        if include_interactions:
            interaction_df, interaction_names = self.create_interaction_features(df, feature_names)
            if len(interaction_names) > 0:
                engineered_df = pd.concat([engineered_df, interaction_df], axis=1)
                new_feature_names.extend(interaction_names)
                print(f"Added {len(interaction_names)} interaction features")
        
        # Create polynomial features
        if include_polynomial:
            poly_df, poly_names = self.create_polynomial_features(df, feature_names, degree=2)
            if len(poly_names) > 0:
                engineered_df = pd.concat([engineered_df, poly_df], axis=1)
                new_feature_names.extend(poly_names)
                print(f"Added {len(poly_names)} polynomial features")
        
        self.derived_feature_names = [f for f in new_feature_names if f not in feature_names]
        
        print(f"Total features after engineering: {len(new_feature_names)}")
        print(f"New derived features: {len(self.derived_feature_names)}")
        
        return engineered_df, new_feature_names
    
    def transform_new_data(self, df, original_feature_names):
        """Transform new data using the same feature engineering"""
        # Apply the same transformations
        return self.engineer_features(
            df, 
            original_feature_names,
            include_ratios=True,
            include_differences=True,
            include_interactions=True,
            include_polynomial=False
        )
