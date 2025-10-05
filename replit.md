# ExoDetect: Exoplanet Classification System

## Overview

ExoDetect is a machine learning application for classifying exoplanets from NASA's Kepler mission data. The system uses ensemble methods (XGBoost, LightGBM, Random Forest) to predict object disposition into three categories: Confirmed, Candidate, or False Positive. Built with Streamlit for the web interface, the application provides data preprocessing, model training, hyperparameter optimization, model calibration, and explainability features through SHAP integration.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
**Technology**: Streamlit web framework

**Design Pattern**: Single-page application with cached resources

**Rationale**: Streamlit provides rapid development for data science applications with built-in state management and component rendering. The `@st.cache_resource` decorator ensures model artifacts are loaded once and reused across sessions, improving performance.

**Key Components**:
- Interactive visualizations using Plotly and Matplotlib
- Real-time model predictions and explanations
- File upload capabilities for custom datasets
- Multi-panel layout with sidebar navigation

### Backend Architecture

**Core Processing Modules**:

1. **Data Preprocessing** (`utils/data_preprocessing.py`)
   - **Purpose**: Transform raw NASA exoplanet data into ML-ready format
   - **Approach**: Pipeline-based preprocessing with sklearn's StandardScaler and SimpleImputer
   - **Key Decisions**: 
     - Median imputation for missing values (robust to outliers in astronomical data)
     - Log transformation for highly skewed features (period, depth, insolation)
     - Automatic label detection and standardization from multiple disposition formats
   - **Rationale**: Astronomical data often has heavy-tailed distributions; log transforms normalize these for better model performance

2. **Feature Engineering** (`utils/feature_engineering.py`)
   - **Purpose**: Create domain-specific derived features from raw measurements
   - **Approach**: Generate ratio and difference features between related astronomical properties
   - **Examples**: Planet-to-star radius ratios, transit depth-to-duration ratios
   - **Rationale**: Physical relationships between features (like radius ratios) carry more predictive power than raw measurements alone

3. **Model Training** (`utils/model_training.py`)
   - **Architecture**: Multi-model support with fallback mechanisms
   - **Supported Models**: XGBoost (primary), LightGBM (alternative), Random Forest (fallback)
   - **Design Pattern**: Factory pattern for model creation with dynamic availability checking
   - **Rationale**: XGBoost/LightGBM excel at tabular classification but aren't always available in all environments; Random Forest provides a reliable fallback
   - **Class Imbalance Handling**: Computed class weights to handle imbalanced datasets common in exoplanet detection

4. **Ensemble Methods** (`utils/ensemble_models.py`)
   - **Architecture**: Stacking classifier combining multiple base estimators
   - **Approach**: Use XGBoost and Random Forest as base models with Logistic Regression meta-learner
   - **Rationale**: Different models capture different patterns; stacking combines their strengths while the meta-learner learns optimal weighting

5. **Hyperparameter Optimization** (`utils/hyperparameter_tuning.py`)
   - **Method**: Randomized search with stratified k-fold cross-validation
   - **Rationale**: Randomized search provides good results faster than grid search while maintaining exploration of parameter space
   - **Cross-validation**: Stratified folds preserve class distribution across imbalanced classes

6. **Model Calibration** (`utils/calibration_utils.py`)
   - **Purpose**: Ensure predicted probabilities reflect true likelihood
   - **Metrics**: Calibration curves and Brier scores for each class
   - **Rationale**: Well-calibrated probabilities are crucial for decision-making in astronomical classification where false positives have real observational costs

7. **Model Explainability** (`utils/shap_utils.py`)
   - **Technology**: SHAP (SHapley Additive exPlanations)
   - **Approach**: TreeExplainer for tree-based models with KernelExplainer fallback
   - **Rationale**: SHAP provides theoretically-grounded feature importance with both global and local explanations, essential for scientific validation

### Data Flow

1. **Input**: CSV file from NASA Kepler mission (cumulative dataset)
2. **Preprocessing**: Label detection → standardization → feature selection → imputation → scaling → log transforms
3. **Feature Engineering**: Generate ratio and difference features
4. **Model Training**: Train with class weights → cross-validation → hyperparameter tuning
5. **Ensemble**: Combine multiple models via stacking
6. **Calibration**: Adjust probabilities for reliability
7. **Prediction**: Generate classifications with confidence scores and SHAP explanations
8. **Persistence**: Save model artifacts (joblib) and feature metadata (JSON)

### Model Persistence

**Format**: Joblib for serialized Python objects (models, preprocessors)

**Artifacts Saved**:
- Trained model (`model.joblib`)
- Preprocessing pipeline (`preprocessor.joblib`)
- Feature names list (`feature_list.json`)

**Rationale**: Joblib provides efficient serialization for scikit-learn compatible objects with compression support

### Error Handling Strategy

**Graceful Degradation**: Optional dependencies (XGBoost, LightGBM, SHAP) have try-except imports with fallback logic

**Rationale**: Ensures application runs in constrained environments while providing enhanced features when available

### Performance Optimizations

- Resource caching via Streamlit decorators
- Lazy loading of heavy dependencies
- Stratified sampling for SHAP explanations (limited to 100 samples for kernel explainer)
- Parallel processing where supported (n_jobs=-1)

## External Dependencies

### Core ML Libraries
- **scikit-learn**: Preprocessing pipelines, model evaluation, ensemble methods
- **pandas**: Data manipulation and CSV handling
- **numpy**: Numerical operations and array handling

### Gradient Boosting Frameworks
- **XGBoost** (optional): Primary classification model for tabular data
- **LightGBM** (optional): Alternative gradient boosting implementation

### Explainability
- **SHAP** (optional): Model interpretability and feature importance

### Visualization
- **Plotly**: Interactive charts and graphs for web interface
- **Matplotlib**: Static plots for calibration curves
- **Seaborn**: Enhanced statistical visualizations

### Web Framework
- **Streamlit**: Complete web application framework with built-in components

### Model Persistence
- **joblib**: Model serialization and deserialization

### Data Source
- **NASA Exoplanet Archive**: Cumulative Kepler Object of Interest (KOI) dataset
  - Format: CSV with comment lines
  - Target: `koi_disposition` or similar disposition column
  - Features: Orbital parameters (period, duration), planetary properties (radius, depth), stellar properties (temperature, radius)

### Configuration
- No external configuration management system
- Parameters hardcoded in module initialization
- Random state set to 42 for reproducibility