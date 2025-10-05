import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Feature explanations will be limited.")

class SHAPExplainer:
    def __init__(self, model, X_background=None):
        self.model = model
        self.explainer = None
        
        if SHAP_AVAILABLE and X_background is not None:
            try:
                # Try TreeExplainer for tree-based models
                if hasattr(model, 'predict_proba') and hasattr(model, 'feature_importances_'):
                    self.explainer = shap.TreeExplainer(model)
                else:
                    # Fallback to KernelExplainer with a small sample
                    sample_size = min(100, len(X_background))
                    background_sample = X_background[:sample_size]
                    self.explainer = shap.KernelExplainer(model.predict_proba, background_sample)
            except Exception as e:
                print(f"Could not initialize SHAP explainer: {e}")
                self.explainer = None
    
    def explain_prediction(self, X):
        """Generate SHAP values for predictions"""
        if not SHAP_AVAILABLE or self.explainer is None:
            return None
        
        try:
            shap_values = self.explainer.shap_values(X)
            return shap_values
        except Exception as e:
            print(f"Error generating SHAP values: {e}")
            return None
    
    def create_summary_plot(self, shap_values, X, feature_names=None, max_display=20):
        """Create SHAP summary plot"""
        if not SHAP_AVAILABLE or shap_values is None:
            return None
        
        try:
            plt.figure(figsize=(10, 6))
            if isinstance(shap_values, list):  # Multi-class case
                # Use the first class for summary
                shap.summary_plot(shap_values[0], X, feature_names=feature_names, 
                                show=False, max_display=max_display)
            else:
                shap.summary_plot(shap_values, X, feature_names=feature_names, 
                                show=False, max_display=max_display)
            
            return plt.gcf()
        except Exception as e:
            print(f"Error creating SHAP summary plot: {e}")
            return None
    
    def create_waterfall_plot(self, shap_values, feature_names, prediction_class):
        """Create SHAP waterfall plot for a single prediction"""
        if not SHAP_AVAILABLE or shap_values is None:
            return None
        
        try:
            # For multi-class, we might need to handle differently
            if isinstance(shap_values, np.ndarray) and len(shap_values.shape) > 1:
                # Take the first class or the predicted class
                shap_values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            
            # Create a simple bar plot since waterfall might not be available in all SHAP versions
            plt.figure(figsize=(10, 6))
            
            if feature_names and len(feature_names) == len(shap_values):
                # Create feature importance bar plot
                importance_data = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.abs(shap_values)
                }).sort_values('importance', ascending=True)
                
                colors = ['red' if x < 0 else 'blue' for x in shap_values]
                plt.barh(range(len(importance_data)), importance_data['importance'], 
                        color=colors[:len(importance_data)])
                plt.yticks(range(len(importance_data)), importance_data['feature'])
                plt.xlabel('Feature Importance (|SHAP value|)')
                plt.title(f'Feature Importance for Prediction: {prediction_class}')
                plt.tight_layout()
                
                return plt.gcf()
            else:
                return None
                
        except Exception as e:
            print(f"Error creating SHAP waterfall plot: {e}")
            return None
    
    def get_top_features(self, shap_values, feature_names, top_k=10):
        """Get top contributing features"""
        if shap_values is None or feature_names is None:
            return None
        
        try:
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Use first class
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # Use first prediction
            
            # Get absolute importance
            abs_importance = np.abs(shap_values)
            
            # Get top features
            top_indices = np.argsort(abs_importance)[-top_k:][::-1]
            
            top_features = []
            for idx in top_indices:
                if idx < len(feature_names):
                    top_features.append({
                        'feature': feature_names[idx],
                        'importance': abs_importance[idx],
                        'shap_value': shap_values[idx]
                    })
            
            return top_features
            
        except Exception as e:
            print(f"Error getting top features: {e}")
            return None
