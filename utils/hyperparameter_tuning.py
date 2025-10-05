import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class HyperparameterOptimizer:
    def __init__(self, model_type='xgboost', n_iter=20, cv=5, random_state=42):
        self.model_type = model_type.lower()
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.cv_results = None
        
    def get_param_distributions(self):
        """Get parameter distributions for hyperparameter search"""
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 4, 5, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.5, 1],
                'reg_lambda': [1, 5, 10, 20],
                'min_child_weight': [1, 3, 5, 7]
            }
        else:  # RandomForest
            return {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'criterion': ['gini', 'entropy']
            }
    
    def create_base_model(self, class_weights_dict):
        """Create base model for hyperparameter search"""
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            # XGBoost doesn't use class_weight directly, we'll use sample_weight during fit
            return xgb.XGBClassifier(
                objective='multi:softprob',
                eval_metric='mlogloss',
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            return RandomForestClassifier(
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
    
    def optimize(self, X, y, scoring='f1_macro'):
        """Perform hyperparameter optimization using RandomizedSearchCV"""
        print(f"Starting hyperparameter optimization for {self.model_type}")
        print(f"Number of iterations: {self.n_iter}, CV folds: {self.cv}")
        
        # Calculate class weights
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights_dict = dict(zip(classes, class_weights))
        
        # Create base model
        base_model = self.create_base_model(class_weights_dict)
        
        # Get parameter distributions
        param_distributions = self.get_param_distributions()
        
        # Create stratified k-fold
        cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            scoring=scoring,
            cv=cv_splitter,
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1,
            return_train_score=True
        )
        
        # For XGBoost, we need to pass sample_weight
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            sample_weights = np.array([class_weights_dict[label] for label in y])
            random_search.fit(X, y, sample_weight=sample_weights)
        else:
            random_search.fit(X, y)
        
        # Store results
        self.best_model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        self.cv_results = pd.DataFrame(random_search.cv_results_)
        
        print(f"Best score: {random_search.best_score_:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        return {
            'best_model': self.best_model,
            'best_params': self.best_params,
            'best_score': random_search.best_score_,
            'cv_results': self.cv_results
        }
    
    def get_top_param_sets(self, n=5):
        """Get top N parameter sets from CV results"""
        if self.cv_results is None:
            return None
        
        # Sort by mean test score
        sorted_results = self.cv_results.sort_values('mean_test_score', ascending=False)
        
        # Extract relevant columns
        param_cols = [col for col in sorted_results.columns if col.startswith('param_')]
        score_cols = ['mean_test_score', 'std_test_score', 'rank_test_score']
        
        top_results = sorted_results[param_cols + score_cols].head(n)
        
        return top_results
