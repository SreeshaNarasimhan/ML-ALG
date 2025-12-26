import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

# Attempt XGBoost import with fallback
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Page Config
st.set_page_config(page_title="UltraML Dashboard", layout="wide")

# Custom CSS for a unique look
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #4CAF50; color: white; border: none; }
    .stSelectbox, .stMultiSelect { border-radius: 10px; }
    div[data-testid="stMetricValue"] { color: #4CAF50; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 UltraML: Automated Machine Learning Dashboard")
st.write("Upload a CSV, select your target, and run 12 different algorithms instantly.")

# --- SIDEBAR: DATA UPLOAD ---
st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())
    
    # --- SIDEBAR: SETTINGS ---
    st.sidebar.header("2. Configure Model")
    all_columns = df.columns.tolist()
    target_col = st.sidebar.selectbox("Select Target Column (Y)", all_columns)
    feature_cols = st.sidebar.multiselect("Select Feature Columns (X)", [c for c in all_columns if c != target_col], default=[c for c in all_columns if c != target_col])
    
    # Check if target is Classification or Regression
    is_classification = df[target_col].nunique() < 20 or df[target_col].dtype == 'object'
    task_type = "Classification" if is_classification else "Regression"
    st.sidebar.info(f"Detected Task: **{task_type}**")

    algo_choice = st.sidebar.selectbox("3. Select Algorithm", [
        "Logistic/Linear Regression", "Decision Tree", "Random Forest", "KNN", 
        "SVM", "Naive Bayes", "Gradient Boosting", "AdaBoost", "XGBoost", "Lasso", "Ridge"
    ])

    if st.sidebar.button("Run Model"):
        with st.spinner('Processing Data and Training Model...'):
            # Data Preprocessing
            X = df[feature_cols].copy()
            y = df[target_col].copy()

            # Handle Categorical Data
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            if y.dtype == 'object':
                le_y = LabelEncoder()
                y = le_y.fit_transform(y.astype(str))

            X.fillna(X.mean(), inplace=True) # Simple imputation

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scaler for distance-based models
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # --- ALGORITHM LOGIC ---
            model = None
            
            if algo_choice == "Logistic/Linear Regression":
                model = LogisticRegression() if is_classification else Ridge() # Ridge as standard linear
            elif algo_choice == "Decision Tree":
                model = DecisionTreeClassifier() if is_classification else DecisionTreeRegressor()
            elif algo_choice == "Random Forest":
                model = RandomForestClassifier() if is_classification else RandomForestRegressor()
            elif algo_choice == "KNN":
                model = KNeighborsClassifier() if is_classification else KNeighborsRegressor()
                X_train, X_test = X_train_scaled, X_test_scaled
            elif algo_choice == "SVM":
                model = SVC() if is_classification else SVR()
                X_train, X_test = X_train_scaled, X_test_scaled
            elif algo_choice == "Naive Bayes":
                model = GaussianNB() if is_classification else None
                if not is_classification: st.error("Naive Bayes is for Classification only.")
            elif algo_choice == "Gradient Boosting":
                model = GradientBoostingClassifier() if is_classification else GradientBoostingRegressor()
            elif algo_choice == "AdaBoost":
                model = AdaBoostClassifier() if is_classification else AdaBoostRegressor()
            elif algo_choice == "XGBoost":
                if XGB_AVAILABLE:
                    model = XGBClassifier() if is_classification else XGBRegressor()
                else:
                    st.warning("XGBoost not found. Falling back to Gradient Boosting.")
                    model = GradientBoostingClassifier() if is_classification else GradientBoostingRegressor()
            elif algo_choice == "Lasso":
                if is_classification:
                    st.error("Lasso is for Regression only.")
                    model = None
                else:
                    model = Lasso()
            elif algo_choice == "Ridge":
                if is_classification:
                    st.error("Ridge is for Regression only.")
                    model = None
                else:
                    model = Ridge()

            # Train and Predict
            if model:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                # --- OUTPUTS ---
                st.success(f"{algo_choice} Model Trained Successfully!")
                
                col1, col2 = st.columns(2)
                if is_classification:
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average='weighted')
                    col1.metric("Accuracy", f"{acc:.2%}")
                    col2.metric("F1 Score", f"{f1:.2%}")
                else:
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    r2 = r2_score(y_test, preds)
                    col1.metric("RMSE", f"{rmse:.4f}")
                    col2.metric("R² Score", f"{r2:.4f}")

                # Visualization
                if is_classification:
                    cm = confusion_matrix(y_test, preds)
                    fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale='Viridis')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': 'Predicted'}, 
                                     title="Actual vs Predicted Values", template="plotly_dark")
                    fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color="Red", dash="dash"))
                    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a CSV file from the sidebar to begin.")