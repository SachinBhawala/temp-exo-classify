import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from utils.data_preprocessing import ExoplanetPreprocessor
from utils.model_training import ExoplanetClassifier
from utils.shap_utils import SHAPExplainer
from utils.hyperparameter_tuning import HyperparameterOptimizer
from utils.ensemble_models import EnsembleClassifier
from utils.calibration_utils import ModelCalibration
from utils.feature_engineering import ExoplanetFeatureEngineer

# Page configuration
st.set_page_config(
    page_title="ExoDetect: Exoplanet Classifier",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache resources for performance
@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing artifacts"""
    artifacts = {}
    model_dir = "models"
    
    if os.path.exists(f"{model_dir}/model.joblib"):
        artifacts['model'] = joblib.load(f"{model_dir}/model.joblib")
    if os.path.exists(f"{model_dir}/preprocessor.joblib"):
        artifacts['preprocessor'] = joblib.load(f"{model_dir}/preprocessor.joblib")
    if os.path.exists(f"{model_dir}/feature_list.json"):
        with open(f"{model_dir}/feature_list.json", 'r') as f:
            artifacts['features'] = json.load(f)
    if os.path.exists(f"{model_dir}/label_encoder.json"):
        with open(f"{model_dir}/label_encoder.json", 'r') as f:
            artifacts['labels'] = json.load(f)
    if os.path.exists(f"{model_dir}/report.json"):
        with open(f"{model_dir}/report.json", 'r') as f:
            artifacts['report'] = json.load(f)
    
    return artifacts

@st.cache_resource
def load_shap_explainer(_model, X_sample):
    """Load SHAP explainer"""
    return SHAPExplainer(_model, X_sample)

def main():
    st.title("🪐 ExoDetect: Multi-class Exoplanet Classifier")
    st.markdown("### Classify astronomical objects as Confirmed planets, Candidates, or False Positives")
    
    # Sidebar for navigation and model info
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Train Model", "Make Predictions", "Model Performance", "Feature Analysis"])
    
    # Show model info in sidebar if model exists
    st.sidebar.divider()
    artifacts = load_model_artifacts()
    if artifacts and 'report' in artifacts:
        st.sidebar.markdown("### 📊 Current Model")
        report = artifacts['report']
        st.sidebar.metric("Overall Accuracy", f"{report['overall_accuracy']:.1%}")
        if 'training_info' in report:
            st.sidebar.caption(f"Model: {report['training_info'].get('model_type', 'N/A')}")
            st.sidebar.caption(f"Features: {report['training_info'].get('n_features', 'N/A')}")
    else:
        st.sidebar.info("No model trained yet")
    
    if page == "Train Model":
        train_model_page()
    elif page == "Make Predictions":
        prediction_page()
    elif page == "Model Performance":
        model_performance_page()
    elif page == "Feature Analysis":
        feature_analysis_page()

def train_model_page():
    st.header("🔧 Train Exoplanet Classifier")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload NASA Exoplanet Dataset (CSV)", 
        type=['csv'],
        help="Upload the cumulative NASA exoplanet dataset"
    )
    
    if uploaded_file is None:
        # Try to use the default file
        default_file = "attached_assets/cumulative_2025.10.03_23.58.46_1759634390786.csv"
        if os.path.exists(default_file):
            st.info(f"Using default dataset: {default_file}")
            df = pd.read_csv(default_file, comment='#', low_memory=False)
        else:
            st.warning("Please upload the NASA exoplanet dataset to proceed.")
            return
    else:
        df = pd.read_csv(uploaded_file, comment='#', low_memory=False)
    
    st.success(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Show data preview
    with st.expander("Data Preview", expanded=False):
        st.dataframe(df.head(10))
        st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
    
    # Training parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        model_type = st.selectbox("Model Type", ["XGBoost", "LightGBM", "Random Forest", "Ensemble (Stacking)"])
    with col2:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.25)
    with col3:
        random_state = st.number_input("Random State", value=42, min_value=1)
    
    # Advanced options
    st.markdown("### ⚙️ Advanced Options")
    
    # Feature Engineering
    use_feature_engineering = st.checkbox("Enable Feature Engineering", 
                                          help="Create derived features (ratios, interactions) for improved performance")
    
    if use_feature_engineering:
        fe_col1, fe_col2 = st.columns(2)
        with fe_col1:
            include_ratios = st.checkbox("Ratio Features", value=True, 
                                         help="Create ratio features (e.g., planet/star radius)")
            include_interactions = st.checkbox("Interaction Features", value=True,
                                              help="Create multiplicative interactions")
        with fe_col2:
            include_differences = st.checkbox("Difference Features", value=True,
                                             help="Create difference features from error bounds")
            include_polynomial = st.checkbox("Polynomial Features", value=False,
                                            help="Create polynomial features (degree 2)")
    
    # Hyperparameter optimization
    use_hp_optimization = st.checkbox("Enable Hyperparameter Optimization", 
                                      help="Use RandomizedSearchCV to find optimal hyperparameters (slower but better performance)")
    
    if use_hp_optimization:
        hp_col1, hp_col2 = st.columns(2)
        with hp_col1:
            n_iter = st.slider("Number of iterations", 10, 50, 20, 
                             help="Number of parameter settings sampled")
        with hp_col2:
            cv_folds = st.slider("CV folds", 3, 10, 5,
                               help="Number of cross-validation folds")
    
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model... This may take a few minutes."):
            try:
                # Initialize preprocessor and classifier
                preprocessor = ExoplanetPreprocessor()
                classifier = ExoplanetClassifier(model_type=model_type.lower().replace(" ", ""))
                
                # Preprocess data
                st.info("Preprocessing data...")
                X, y, feature_names = preprocessor.preprocess_data(df)
                
                if X is None or y is None:
                    st.error("Failed to preprocess data. Please check the dataset format.")
                    return
                
                st.info(f"Features selected: {len(feature_names)}")
                st.info(f"Class distribution: {dict(pd.Series(y).value_counts())}")
                
                # Feature engineering if enabled
                if use_feature_engineering:
                    st.warning("⚠️ Feature engineering is currently disabled to prevent data leakage. This feature will be re-implemented in a future update.")
                    # TODO: Refactor feature engineering to apply only on training data after train/test split
                    # to prevent data leakage
                
                # Encode labels consistently before any training
                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
                
                # Check if ensemble model is selected
                if model_type == "Ensemble (Stacking)":
                    st.info("🎯 Training ensemble model with stacking...")
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                    
                    # Split data with encoded labels
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
                    )
                    
                    # Create and train ensemble
                    ensemble = EnsembleClassifier(random_state=random_state)
                    
                    # Evaluate base models first
                    st.info("Evaluating base models...")
                    base_results = ensemble.evaluate_base_models(X_train, y_train, cv=5)
                    
                    with st.expander("Base Model Performance"):
                        for name, result in base_results.items():
                            st.write(f"**{name}**: {result['mean_score']:.4f} (+/- {result['std_score']:.4f})")
                    
                    # Train stacking ensemble
                    model = ensemble.train(X_train, y_train, label_encoder)
                    
                    # Evaluate on test set
                    y_pred = ensemble.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    target_names = list(label_encoder.classes_)
                    class_report = classification_report(
                        y_test, y_pred, 
                        target_names=target_names,
                        output_dict=True,
                        zero_division=0
                    )
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    
                    feature_importance = ensemble.get_feature_importance()
                    
                    results = {
                        'model': model,
                        'label_encoder': label_encoder,
                        'accuracy': accuracy,
                        'classification_report': class_report,
                        'confusion_matrix': conf_matrix,
                        'feature_importance': feature_importance,
                        'X_test': X_test,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'hp_optimization': {'enabled': False},
                        'ensemble_info': {
                            'type': 'stacking',
                            'base_models': base_results
                        }
                    }
                # Hyperparameter optimization if enabled (not for ensemble)
                elif use_hp_optimization:
                    st.info("🔍 Optimizing hyperparameters with cross-validation...")
                    from sklearn.model_selection import train_test_split
                    
                    # Split data for optimization (labels already encoded above)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
                    )
                    
                    # Run hyperparameter optimization
                    optimizer = HyperparameterOptimizer(
                        model_type=model_type.lower().replace(" ", ""),
                        n_iter=n_iter,
                        cv=cv_folds,
                        random_state=random_state
                    )
                    
                    hp_results = optimizer.optimize(X_train, y_train, scoring='f1_macro')
                    
                    # Show optimization results
                    st.success(f"✅ Best CV Score: {hp_results['best_score']:.4f}")
                    with st.expander("Best Parameters"):
                        st.json(hp_results['best_params'])
                    
                    # Use optimized model
                    classifier.model = hp_results['best_model']
                    classifier.label_encoder = label_encoder
                    
                    # Evaluate on test set
                    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                    y_pred = classifier.model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    target_names = list(label_encoder.classes_)
                    class_report = classification_report(
                        y_test, y_pred, 
                        target_names=target_names,
                        output_dict=True,
                        zero_division=0
                    )
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    
                    feature_importance = None
                    if hasattr(classifier.model, 'feature_importances_'):
                        feature_importance = classifier.model.feature_importances_
                    
                    results = {
                        'model': classifier.model,
                        'label_encoder': label_encoder,
                        'accuracy': accuracy,
                        'classification_report': class_report,
                        'confusion_matrix': conf_matrix,
                        'feature_importance': feature_importance,
                        'X_test': X_test,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'hp_optimization': {
                            'enabled': True,
                            'best_params': hp_results['best_params'],
                            'best_cv_score': hp_results['best_score']
                        }
                    }
                else:
                    # Train model normally with encoded labels
                    st.info("Training classifier...")
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
                    )
                    
                    # Train using the classifier's create_model and train logic
                    classifier.label_encoder = label_encoder
                    n_classes = len(label_encoder.classes_)
                    class_weights_dict = classifier.calculate_class_weights(y_train)
                    classifier.create_model(n_classes, class_weights_dict)
                    
                    # Train model with sample weights
                    if classifier.model_type in ['xgboost']:
                        sample_weights = classifier.calculate_sample_weights(y_train, class_weights_dict)
                        classifier.model.fit(X_train, y_train, sample_weight=sample_weights)
                    else:
                        classifier.model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = classifier.model.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    target_names = list(label_encoder.classes_)
                    class_report = classification_report(
                        y_test, y_pred, 
                        target_names=target_names,
                        output_dict=True,
                        zero_division=0
                    )
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    
                    feature_importance = None
                    if hasattr(classifier.model, 'feature_importances_'):
                        feature_importance = classifier.model.feature_importances_
                    
                    results = {
                        'model': classifier.model,
                        'label_encoder': label_encoder,
                        'accuracy': accuracy,
                        'classification_report': class_report,
                        'confusion_matrix': conf_matrix,
                        'feature_importance': feature_importance,
                        'X_test': X_test,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'hp_optimization': {'enabled': False}
                    }
                
                # Calibration analysis
                st.info("📊 Performing calibration analysis...")
                try:
                    calibration = ModelCalibration(
                        results['model'], 
                        results['X_test'], 
                        results['y_test'],
                        list(results['label_encoder'].classes_)
                    )
                    
                    # Get threshold recommendations
                    threshold_recs = calibration.get_threshold_recommendations()
                    results['calibration_thresholds'] = threshold_recs
                    
                    st.success("✅ Calibration analysis complete")
                except Exception as e:
                    st.warning(f"Calibration analysis skipped: {str(e)}")
                    results['calibration_thresholds'] = {}
                
                # Save artifacts with versioning
                os.makedirs("models", exist_ok=True)
                os.makedirs("models/versions", exist_ok=True)
                
                # Create version info
                import datetime
                version_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                version_info = {
                    'timestamp': version_timestamp,
                    'model_type': model_type,
                    'accuracy': results['accuracy'],
                    'n_features': len(feature_names),
                    'feature_engineering': use_feature_engineering if 'use_feature_engineering' in locals() else False,
                    'hyperparameter_optimization': use_hp_optimization if 'use_hp_optimization' in locals() else False
                }
                
                # Save current version as active
                joblib.dump(results['model'], "models/model.joblib")
                joblib.dump(preprocessor, "models/preprocessor.joblib")
                joblib.dump({'X_test': results['X_test'], 'y_test': results['y_test']}, "models/test_data.joblib")
                
                # Prepare report data before saving
                report_data = {
                    'overall_accuracy': results['accuracy'],
                    'classification_report': results['classification_report'],
                    'confusion_matrix': results['confusion_matrix'].tolist(),
                    'feature_importance': results['feature_importance'].tolist() if results['feature_importance'] is not None else None,
                    'calibration_thresholds': results.get('calibration_thresholds', {}),
                    'training_info': {
                        'model_type': model_type,
                        'test_size': test_size,
                        'random_state': random_state,
                        'n_features': len(feature_names),
                        'n_samples': len(X),
                        'hyperparameter_optimization': results.get('hp_optimization', {'enabled': False})
                    }
                }
                
                # Save versioned copy with complete artifacts
                version_dir = f"models/versions/v_{version_timestamp}"
                os.makedirs(version_dir, exist_ok=True)
                joblib.dump(results['model'], f"{version_dir}/model.joblib")
                joblib.dump(preprocessor, f"{version_dir}/preprocessor.joblib")
                
                # Save version-specific report
                with open(f"{version_dir}/report.json", 'w') as f:
                    json.dump(report_data, f, indent=2)
                
                # Save feature list for this version
                with open(f"{version_dir}/feature_list.json", 'w') as f:
                    json.dump(feature_names, f)
                
                # Save version metadata
                with open(f"{version_dir}/version_info.json", 'w') as f:
                    json.dump(version_info, f, indent=2)
                
                # Update version history
                history_file = "models/version_history.json"
                if os.path.exists(history_file):
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                else:
                    history = []
                
                history.append(version_info)
                
                with open(history_file, 'w') as f:
                    json.dump(history, f, indent=2)
                
                # Save feature list and labels for active model
                with open("models/feature_list.json", 'w') as f:
                    json.dump(feature_names, f)
                
                with open("models/label_encoder.json", 'w') as f:
                    json.dump(list(results['label_encoder'].classes_), f)
                
                # Save active model report (report_data already defined above for versioning)
                with open("models/report.json", 'w') as f:
                    json.dump(report_data, f, indent=2)
                
                # Display results
                st.success("Model training completed!")
                
                # Prominently display overall accuracy
                st.markdown("### 🎯 Model Performance Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("**Overall Accuracy**", f"{results['accuracy']:.1%}", help="Overall classification accuracy on test set")
                    
                with col2:
                    # Get per-class metrics from classification report
                    class_report = results['classification_report']
                    if 'Confirmed' in class_report:
                        confirmed_f1 = class_report['Confirmed']['f1-score']
                        st.metric("Confirmed F1-Score", f"{confirmed_f1:.3f}")
                
                with col3:
                    if 'macro avg' in class_report:
                        macro_f1 = class_report['macro avg']['f1-score']
                        st.metric("Macro F1-Score", f"{macro_f1:.3f}")
                
                # Show detailed classification report
                st.subheader("📊 Classification Report")
                report_df = pd.DataFrame(results['classification_report']).T
                st.dataframe(report_df)
                
                # Show confusion matrix
                st.subheader("🎯 Confusion Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(results['confusion_matrix'], 
                           annot=True, 
                           fmt='d', 
                           cmap='Blues',
                           xticklabels=results['label_encoder'].classes_,
                           yticklabels=results['label_encoder'].classes_,
                           ax=ax)
                ax.set_title('Confusion Matrix')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                st.pyplot(fig)
                
                # Show version history
                if os.path.exists("models/version_history.json"):
                    with st.expander("📜 Model Version History"):
                        with open("models/version_history.json", 'r') as f:
                            history = json.load(f)
                        
                        if len(history) > 0:
                            history_df = pd.DataFrame(history)
                            history_df = history_df.sort_values('timestamp', ascending=False)
                            st.dataframe(history_df)
                            
                            # Highlight best model
                            best_idx = history_df['accuracy'].idxmax()
                            best_model = history_df.loc[best_idx]
                            st.success(f"🏆 Best model: {best_model['timestamp']} (Accuracy: {best_model['accuracy']:.3f})")
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                st.exception(e)

def prediction_page():
    st.header("🔮 Make Predictions")
    
    # Load model artifacts
    artifacts = load_model_artifacts()
    
    if not artifacts:
        st.error("No trained model found. Please train a model first.")
        return
    
    if not all(key in artifacts for key in ['model', 'preprocessor', 'features', 'labels']):
        st.error("Incomplete model artifacts. Please retrain the model.")
        return
    
    st.success("Model loaded successfully!")
    
    # Prediction mode selection
    prediction_mode = st.radio(
        "Choose prediction mode:",
        ["Upload CSV File", "Manual Feature Entry"]
    )
    
    if prediction_mode == "Upload CSV File":
        batch_prediction_interface(artifacts)
    else:
        manual_prediction_interface(artifacts)

def batch_prediction_interface(artifacts):
    st.subheader("📁 Batch Predictions from CSV")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file for prediction", 
        type=['csv'],
        help="Upload a CSV file with the same features as the training data"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.info(f"File uploaded: {len(df)} rows, {len(df.columns)} columns")
            
            with st.expander("Data Preview", expanded=True):
                st.dataframe(df.head())
            
            if st.button("Generate Predictions", type="primary"):
                with st.spinner("Generating predictions..."):
                    # Preprocess the data
                    preprocessor = artifacts['preprocessor']
                    X_processed = preprocessor.transform_new_data(df, artifacts['features'])
                    
                    if X_processed is not None:
                        # Make predictions
                        model = artifacts['model']
                        predictions = model.predict(X_processed)
                        probabilities = model.predict_proba(X_processed)
                        
                        # Create results dataframe
                        results_df = df.copy()
                        results_df['Predicted_Class'] = predictions
                        
                        # Add probability columns
                        for i, label in enumerate(artifacts['labels']):
                            results_df[f'Probability_{label}'] = probabilities[:, i]
                        
                        st.success(f"Predictions generated for {len(results_df)} rows!")
                        
                        # Show results
                        st.subheader("📊 Prediction Results")
                        st.dataframe(results_df)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="exoplanet_predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Show prediction distribution
                        st.subheader("📈 Prediction Distribution")
                        pred_counts = pd.Series(predictions).value_counts()
                        fig = px.bar(x=pred_counts.index, y=pred_counts.values, 
                                   title="Distribution of Predictions")
                        fig.update_xaxis(title="Predicted Class")
                        fig.update_yaxis(title="Count")
                        st.plotly_chart(fig)
                        
                    else:
                        st.error("Failed to preprocess the uploaded data. Please check the format and features.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def manual_prediction_interface(artifacts):
    st.subheader("✏️ Manual Feature Entry")
    
    features = artifacts['features']
    
    st.info(f"Enter values for the following {len(features)} features:")
    
    # Create input fields for each feature
    feature_values = {}
    
    # Organize features in columns for better layout
    n_cols = 3
    cols = st.columns(n_cols)
    
    for i, feature in enumerate(features):
        col_idx = i % n_cols
        with cols[col_idx]:
            # Provide helpful tooltips and reasonable defaults
            if 'period' in feature.lower():
                feature_values[feature] = st.number_input(
                    f"{feature} (days)", 
                    value=10.0, 
                    help="Orbital period in days"
                )
            elif 'radius' in feature.lower() or 'prad' in feature.lower():
                feature_values[feature] = st.number_input(
                    f"{feature} (Earth radii)", 
                    value=1.0, 
                    help="Planetary radius in Earth radii"
                )
            elif 'depth' in feature.lower():
                feature_values[feature] = st.number_input(
                    f"{feature} (ppm)", 
                    value=1000.0, 
                    help="Transit depth in parts per million"
                )
            elif 'temp' in feature.lower() or 'teff' in feature.lower():
                feature_values[feature] = st.number_input(
                    f"{feature} (K)", 
                    value=5000.0, 
                    help="Temperature in Kelvin"
                )
            elif 'mag' in feature.lower():
                feature_values[feature] = st.number_input(
                    f"{feature} (mag)", 
                    value=15.0, 
                    help="Magnitude"
                )
            else:
                feature_values[feature] = st.number_input(
                    feature, 
                    value=1.0
                )
    
    if st.button("Predict", type="primary"):
        try:
            # Create DataFrame with the input values
            input_df = pd.DataFrame([feature_values])
            
            # Preprocess
            preprocessor = artifacts['preprocessor']
            X_processed = preprocessor.transform_new_data(input_df, features)
            
            if X_processed is not None:
                # Make prediction
                model = artifacts['model']
                prediction = model.predict(X_processed)[0]
                probabilities = model.predict_proba(X_processed)[0]
                
                # Display results
                st.success("Prediction completed!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Predicted Class")
                    st.write(f"**{prediction}**")
                    
                    # Show probabilities
                    st.subheader("📊 Class Probabilities")
                    for i, label in enumerate(artifacts['labels']):
                        st.write(f"**{label}**: {probabilities[i]:.3f}")
                
                with col2:
                    # Probability bar chart
                    fig = px.bar(
                        x=artifacts['labels'], 
                        y=probabilities,
                        title="Prediction Probabilities",
                        color=probabilities,
                        color_continuous_scale="viridis"
                    )
                    fig.update_xaxis(title="Class")
                    fig.update_yaxis(title="Probability")
                    st.plotly_chart(fig)
                
                # SHAP explanation (if possible)
                try:
                    if hasattr(model, 'predict_proba') and len(X_processed) > 0:
                        st.subheader("🔍 Feature Importance (SHAP)")
                        
                        # Create a small sample for SHAP background
                        background_sample = X_processed[:1]  # Use the current prediction as background
                        explainer = load_shap_explainer(model, background_sample)
                        
                        if explainer.explainer:
                            shap_values = explainer.explain_prediction(X_processed)
                            if shap_values is not None:
                                # Create SHAP waterfall plot
                                fig = explainer.create_waterfall_plot(shap_values[0], features, prediction)
                                if fig:
                                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate SHAP explanation: {str(e)}")
            
            else:
                st.error("Failed to preprocess the input data.")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)

def model_performance_page():
    st.header("📈 Model Performance Analysis")
    
    # Load model artifacts
    artifacts = load_model_artifacts()
    
    if not artifacts or 'report' not in artifacts:
        st.error("No model performance data found. Please train a model first.")
        return
    
    report = artifacts['report']
    
    # Display overall accuracy prominently at the top
    st.markdown("## 🎯 Overall Model Accuracy")
    accuracy_col1, accuracy_col2, accuracy_col3 = st.columns([2, 1, 1])
    with accuracy_col1:
        st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>{report['overall_accuracy']:.1%}</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Classification Accuracy on Test Set</p>", unsafe_allow_html=True)
    
    st.divider()
    
    # Overall metrics
    st.subheader("📊 Detailed Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Accuracy", f"{report['overall_accuracy']:.3f}")
    
    # Extract macro and weighted averages from classification report
    class_report = report['classification_report']
    
    with col2:
        if 'macro avg' in class_report:
            macro_f1 = class_report['macro avg']['f1-score']
            st.metric("Macro F1-Score", f"{macro_f1:.3f}")
    
    with col3:
        if 'weighted avg' in class_report:
            weighted_f1 = class_report['weighted avg']['f1-score']
            st.metric("Weighted F1-Score", f"{weighted_f1:.3f}")
    
    with col4:
        if 'Confirmed' in class_report:
            confirmed_recall = class_report['Confirmed']['recall']
            st.metric("Confirmed Recall", f"{confirmed_recall:.3f}")
    
    # Per-class performance
    st.subheader("📊 Per-Class Performance")
    
    # Convert classification report to DataFrame for better display
    class_df = pd.DataFrame(class_report).T
    
    # Filter out the summary rows for the main display
    main_classes = [col for col in class_df.index if col not in ['accuracy', 'macro avg', 'weighted avg']]
    class_metrics_df = class_df.loc[main_classes, ['precision', 'recall', 'f1-score', 'support']]
    
    st.dataframe(class_metrics_df)
    
    # Visualize per-class metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Precision, Recall, F1 comparison
        metrics_data = []
        for class_name in main_classes:
            if class_name in class_report:
                metrics_data.append({
                    'Class': class_name,
                    'Precision': class_report[class_name]['precision'],
                    'Recall': class_report[class_name]['recall'],
                    'F1-Score': class_report[class_name]['f1-score']
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            fig = px.bar(
                metrics_df.melt(id_vars=['Class'], 
                               value_vars=['Precision', 'Recall', 'F1-Score']),
                x='Class', y='value', color='variable',
                title="Per-Class Performance Metrics",
                barmode='group'
            )
            fig.update_yaxis(title="Score")
            st.plotly_chart(fig)
    
    with col2:
        # Support (number of samples) per class
        support_data = []
        for class_name in main_classes:
            if class_name in class_report:
                support_data.append({
                    'Class': class_name,
                    'Support': class_report[class_name]['support']
                })
        
        if support_data:
            support_df = pd.DataFrame(support_data)
            fig = px.pie(
                support_df, 
                values='Support', 
                names='Class',
                title="Class Distribution (Support)"
            )
            st.plotly_chart(fig)
    
    # Confusion Matrix
    st.subheader("🎯 Confusion Matrix")
    
    if 'confusion_matrix' in report and artifacts.get('labels'):
        conf_matrix = np.array(report['confusion_matrix'])
        labels = artifacts['labels']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        ax.set_title('Confusion Matrix', fontsize=16)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        st.pyplot(fig)
        
        # Confusion matrix as heatmap with Plotly for interactivity
        fig = px.imshow(
            conf_matrix,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels,
            y=labels,
            color_continuous_scale="Blues",
            title="Interactive Confusion Matrix"
        )
        fig.update_traces(text=conf_matrix, texttemplate="%{text}")
        fig.update_xaxis(side="bottom")
        st.plotly_chart(fig)
    
    # Calibration and Threshold Analysis
    st.subheader("📐 Model Calibration & Threshold Tuning")
    
    # Load test data and model if available
    if os.path.exists("models/test_data.joblib") and os.path.exists("models/model.joblib"):
        try:
            test_data = joblib.load("models/test_data.joblib")
            model = artifacts['model']
            
            calibration = ModelCalibration(
                model,
                test_data['X_test'],
                test_data['y_test'],
                artifacts['labels']
            )
            
            # Show saved threshold recommendations
            if 'calibration_thresholds' in report and report['calibration_thresholds']:
                st.markdown("#### 🎯 Recommended Probability Thresholds")
                thresh_df = pd.DataFrame(report['calibration_thresholds']).T
                st.dataframe(thresh_df)
            
            # Interactive calibration analysis
            with st.expander("📊 View Calibration Curves", expanded=False):
                # Select class for detailed analysis
                class_idx = st.selectbox(
                    "Select class for detailed calibration analysis:",
                    range(len(artifacts['labels'])),
                    format_func=lambda x: artifacts['labels'][x]
                )
                
                cal_col1, cal_col2 = st.columns(2)
                
                with cal_col1:
                    # Calibration curve
                    fig_cal, brier = calibration.plot_calibration_curve(class_idx)
                    st.pyplot(fig_cal)
                    st.caption(f"Brier Score: {brier:.4f} (lower is better)")
                
                with cal_col2:
                    # Threshold analysis
                    fig_thresh, opt_thresh = calibration.plot_threshold_analysis(class_idx)
                    st.pyplot(fig_thresh)
                    st.caption(f"Optimal F1 threshold: {opt_thresh:.3f}")
        
        except Exception as e:
            st.warning(f"Could not load calibration data: {str(e)}")
    else:
        st.info("Calibration analysis will be available after training a model.")
    
    # Training information
    st.subheader("ℹ️ Training Information")
    
    if 'training_info' in report:
        training_info = report['training_info']
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.write(f"**Model Type:** {training_info.get('model_type', 'N/A')}")
            st.write(f"**Number of Features:** {training_info.get('n_features', 'N/A')}")
            st.write(f"**Number of Samples:** {training_info.get('n_samples', 'N/A')}")
        
        with info_col2:
            st.write(f"**Test Size:** {training_info.get('test_size', 'N/A')}")
            st.write(f"**Random State:** {training_info.get('random_state', 'N/A')}")

def feature_analysis_page():
    st.header("🔍 Feature Analysis")
    
    # Load model artifacts
    artifacts = load_model_artifacts()
    
    if not artifacts:
        st.error("No trained model found. Please train a model first.")
        return
    
    if 'report' in artifacts and 'feature_importance' in artifacts['report']:
        feature_importance = artifacts['report']['feature_importance']
        features = artifacts['features']
        
        if feature_importance and len(feature_importance) == len(features):
            st.subheader("📊 Feature Importance")
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            # Display top features
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top 10 Most Important Features:**")
                st.dataframe(importance_df.head(10))
            
            with col2:
                # Feature importance plot
                fig = px.bar(
                    importance_df.head(15), 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="Top 15 Feature Importance"
                )
                fig.update_yaxis(categoryorder="total ascending")
                st.plotly_chart(fig)
            
            # Full feature importance (expandable)
            with st.expander("All Features Importance", expanded=False):
                st.dataframe(importance_df)
        
        else:
            st.warning("Feature importance data is not available or inconsistent.")
    
    else:
        st.warning("Feature importance data is not available. This might be because the model doesn't support feature importance or it wasn't saved during training.")
    
    # Feature descriptions and statistics
    st.subheader("📋 Feature Descriptions")
    
    if 'features' in artifacts:
        features = artifacts['features']
        
        feature_descriptions = {
            'koi_period': 'Orbital Period (days) - Time for one complete orbit',
            'koi_duration': 'Transit Duration (hours) - How long the planet blocks the star',
            'koi_depth': 'Transit Depth (ppm) - Amount of starlight blocked during transit',
            'koi_prad': 'Planetary Radius (Earth radii) - Size of the planet',
            'koi_teq': 'Equilibrium Temperature (K) - Expected planet temperature',
            'koi_insol': 'Insolation Flux (Earth flux) - Amount of stellar radiation received',
            'koi_model_snr': 'Transit Signal-to-Noise Ratio - Quality of the detection',
            'koi_steff': 'Stellar Effective Temperature (K) - Temperature of the host star',
            'koi_slogg': 'Stellar Surface Gravity - Density indicator of the host star',
            'koi_srad': 'Stellar Radius (Solar radii) - Size of the host star',
            'koi_kepmag': 'Kepler Magnitude - Brightness of the star',
            'koi_impact': 'Impact Parameter - How centrally the planet transits',
        }
        
        st.write("**Feature Descriptions:**")
        for feature in features:
            description = feature_descriptions.get(feature, "No description available")
            st.write(f"• **{feature}**: {description}")

if __name__ == "__main__":
    main()
