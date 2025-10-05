import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Using RandomForest as fallback.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError) as e:
    LIGHTGBM_AVAILABLE = False
    print(f"LightGBM not available ({str(e)}). Using RandomForest as fallback.")

class ExoplanetClassifier:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type.lower()
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def create_model(self, n_classes, class_weights=None):
        """Create the specified model"""
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                objective='multi:softprob',
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss',
                n_jobs=-1
            )
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMClassifier(
                objective='multiclass',
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            # Fallback to RandomForest
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            self.model_type = 'randomforest'
    
    def calculate_class_weights(self, y):
        """Calculate class weights for handling imbalanced data"""
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, class_weights))
    
    def calculate_sample_weights(self, y_encoded, class_weights_dict):
        """Calculate sample weights from class weights"""
        sample_weights = np.array([class_weights_dict[label] for label in y_encoded])
        return sample_weights
    
    def train_and_evaluate(self, X, y, test_size=0.25, random_state=42):
        """Train and evaluate the model"""
        try:
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            n_classes = len(self.label_encoder.classes_)
            
            print(f"Training {self.model_type} model with {n_classes} classes")
            print(f"Classes: {list(self.label_encoder.classes_)}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
            )
            
            # Calculate class weights
            class_weights_dict = self.calculate_class_weights(y_train)
            print(f"Class weights: {class_weights_dict}")
            
            # Create model
            self.create_model(n_classes, class_weights_dict)
            
            # Train model with appropriate weighting
            if self.model_type in ['xgboost', 'lightgbm']:
                # Use sample weights for gradient boosting models
                sample_weights = self.calculate_sample_weights(y_train, class_weights_dict)
                self.model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                # RandomForest uses class_weight parameter
                self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Classification report with original class names
            target_names = list(self.label_encoder.classes_)
            class_report = classification_report(
                y_test, y_pred, 
                target_names=target_names,
                output_dict=True,
                zero_division=0
            )
            
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Feature importance
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = self.model.feature_importances_
            
            results = {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'feature_importance': feature_importance,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            return results
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise e
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions_encoded = self.model.predict(X)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)
