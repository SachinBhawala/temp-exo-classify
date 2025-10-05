import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class EnsembleClassifier:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.stacking_model = None
        self.label_encoder = None
        
    def create_base_estimators(self, class_weights_dict):
        """Create base estimators for the ensemble"""
        estimators = []
        
        # XGBoost estimator (if available)
        if XGBOOST_AVAILABLE:
            xgb_model = xgb.XGBClassifier(
                objective='multi:softprob',
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='mlogloss',
                n_jobs=-1
            )
            estimators.append(('xgboost', xgb_model))
        
        # RandomForest estimator
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        estimators.append(('random_forest', rf_model))
        
        # Add a second RandomForest with different parameters for diversity
        rf_model2 = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            class_weight='balanced',
            random_state=self.random_state + 1,
            n_jobs=-1
        )
        estimators.append(('random_forest_2', rf_model2))
        
        return estimators
    
    def create_stacking_classifier(self, estimators):
        """Create stacking classifier with meta-learner"""
        # Use Logistic Regression as the meta-learner
        meta_learner = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        # Create stacking classifier
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,  # Use 5-fold cross-validation for meta-features
            stack_method='predict_proba',  # Use probabilities as meta-features
            n_jobs=-1,
            verbose=0
        )
        
        return stacking_clf
    
    def train(self, X, y, label_encoder):
        """Train the stacking ensemble"""
        self.label_encoder = label_encoder
        
        # Calculate class weights
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights_dict = dict(zip(classes, class_weights))
        
        print("Creating ensemble with base estimators...")
        # Create base estimators
        estimators = self.create_base_estimators(class_weights_dict)
        
        print(f"Base estimators: {[name for name, _ in estimators]}")
        
        # Create stacking classifier
        print("Creating stacking classifier...")
        self.stacking_model = self.create_stacking_classifier(estimators)
        
        # For XGBoost in ensemble, we need sample weights
        if XGBOOST_AVAILABLE:
            sample_weights = np.array([class_weights_dict[label] for label in y])
            print("Training ensemble with sample weights...")
            self.stacking_model.fit(X, y, sample_weight=sample_weights)
        else:
            print("Training ensemble...")
            self.stacking_model.fit(X, y)
        
        print("Ensemble training completed!")
        
        return self.stacking_model
    
    def evaluate_base_models(self, X, y, cv=5):
        """Evaluate individual base models using cross-validation"""
        results = {}
        
        # Calculate class weights
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights_dict = dict(zip(classes, class_weights))
        
        estimators = self.create_base_estimators(class_weights_dict)
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for name, estimator in estimators:
            print(f"Evaluating {name}...")
            if name == 'xgboost' and XGBOOST_AVAILABLE:
                # For XGBoost, we need to handle sample weights in CV
                sample_weights = np.array([class_weights_dict[label] for label in y])
                scores = cross_val_score(estimator, X, y, cv=cv_splitter, scoring='f1_macro', n_jobs=-1)
            else:
                scores = cross_val_score(estimator, X, y, cv=cv_splitter, scoring='f1_macro', n_jobs=-1)
            
            results[name] = {
                'cv_scores': scores,
                'mean_score': scores.mean(),
                'std_score': scores.std()
            }
            print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return results
    
    def predict(self, X):
        """Make predictions"""
        if self.stacking_model is None:
            raise ValueError("Model not trained yet")
        return self.stacking_model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.stacking_model is None:
            raise ValueError("Model not trained yet")
        return self.stacking_model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance from base models (if available)"""
        if self.stacking_model is None:
            return None
        
        # Try to get feature importance from the first base estimator that supports it
        for name, estimator in self.stacking_model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                return estimator.feature_importances_
        
        return None
