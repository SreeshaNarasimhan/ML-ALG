import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             mean_squared_error, r2_score, mean_absolute_error, confusion_matrix)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                              GradientBoostingClassifier, GradientBoostingRegressor, 
                              AdaBoostClassifier, AdaBoostRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

# Attempt XGBoost import
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Page Configuration
st.set_page_config(page_title="UltraML Studio", layout="wide", page_icon="🚀")

# Premium Modern Styling
st.markdown("""
    <style>
    .stApp { background: linear-gradient(to right bottom, #0f2027, #203a43, #2c5364); color: #ffffff; }
    .css-1d391kg { padding-top: 1rem; } 
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00e5ff; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1); 
        border-radius: 8px 8px 0px 0px; 
        padding: 10px 20px; 
        color: white;
    }
    .stTabs [aria-selected="true"] { background-color: #00e5ff; color: black; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 UltraML Studio: Advanced AI Dashboard")
st.write("Professional grade AutoML: Analyze, Visualize, Train, and Compare models instantly.")

# --- SIDEBAR & DATA LOADING ---
st.sidebar.header("📂 Data Configuration")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")
use_example = st.sidebar.checkbox("Use Example Dataset (if no upload)")

df = None
if uploaded_file:
    df = load_data(uploaded_file)
elif use_example:
    try:
        df = pd.read_csv("dummy_data.csv")
    except:
        st.sidebar.warning("dummy_data.csv not found.")

if df is not None:
    # --- TABS LAYOUT ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🎨 Custom Visualizer", "🤖 Model Training", "🏆 Model Comparison"])

    # Global Settings
    all_cols = df.columns.tolist()
    target_col = st.sidebar.selectbox("Select Target Variable (Y)", all_cols, index=len(all_cols)-1)
    feature_cols = st.sidebar.multiselect("Select Features (X)", [c for c in all_cols if c != target_col], default=[c for c in all_cols if c != target_col][:10])

    if not feature_cols:
        st.error("Please select at least one feature column.")
        st.stop()

    # Preprocessing Logic
    clean_df = df.dropna(subset=[target_col]) # Drop rows where target is NaN
    X_raw = clean_df[feature_cols]
    y_raw = clean_df[target_col]

    # Detect Task Type
    is_classification = False
    if clean_df[target_col].dtype == 'object' or clean_df[target_col].nunique() < 20:
        is_classification = True
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Task Detected:** {'Classification' if is_classification else 'Regression'}")

    # ================= TAB 1: DATA OVERVIEW =================
    with tab1:
        st.header("Dataset Overview")
        st.dataframe(df.head(), use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", df.isna().sum().sum())

        st.subheader("Distribution Analysis (Subplots)")
        # Create Dynamic Subplots for Variables
        num_cols = list(df.select_dtypes(include=np.number).columns)
        if num_cols:
            rows = (len(num_cols) + 1) // 2
            fig = make_subplots(rows=rows, cols=2, subplot_titles=num_cols)
            
            for i, col in enumerate(num_cols):
                fig.add_trace(go.Histogram(x=df[col], name=col, marker_color='#00e5ff'), 
                              row=(i // 2) + 1, col=(i % 2) + 1)
            
            fig.update_layout(height=300*rows, showlegend=False, template="plotly_dark", title_text="Feature Distributions")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns to plot distributions.")

    # ================= TAB 2: CUSTOM VISUALIZER =================
    with tab2:
        st.header("Custom Data Visualizer")
        
        vc1, vc2, vc3 = st.columns(3)
        x_axis = vc1.selectbox("X Axis", all_cols, index=0)
        y_axis = vc2.selectbox("Y Axis", all_cols, index=1 if len(all_cols)>1 else 0)
        plot_type = vc3.selectbox("Plot Type", ["Scatter", "Line", "Bar", "Box", "Histogram"])
        
        color_col = st.selectbox("Color By (Optional)", ["None"] + all_cols)
        color_seq = st.color_picker("Pick a Primary Color", "#00e5ff")
        
        if st.button("Generate User Plot"):
            color_arg = color_col if color_col != "None" else None
            
            if plot_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_arg, color_discrete_sequence=[color_seq])
            elif plot_type == "Line":
                fig = px.line(df, x=x_axis, y=y_axis, color=color_arg, color_discrete_sequence=[color_seq])
            elif plot_type == "Bar":
                fig = px.bar(df, x=x_axis, y=y_axis, color=color_arg, color_discrete_sequence=[color_seq])
            elif plot_type == "Box":
                fig = px.box(df, x=x_axis, y=y_axis, color=color_arg, color_discrete_sequence=[color_seq])
            elif plot_type == "Histogram":
                fig = px.histogram(df, x=x_axis, color=color_arg, color_discrete_sequence=[color_seq])
            
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # ================= PREPARATION FOR MODELS =================
    # Encoding & Filling
    # 1. Fill X
    X_filled = X_raw.copy()
    # Identify cat cols
    cat_cols = X_filled.select_dtypes(include=['object']).columns
    num_cols_X = X_filled.select_dtypes(include=np.number).columns
    
    # Simple fill
    if len(num_cols_X) > 0:
        X_filled[num_cols_X] = X_filled[num_cols_X].fillna(X_filled[num_cols_X].mean())
    
    # One-Hot Encoding for Features
    X_final = pd.get_dummies(X_filled, columns=cat_cols, drop_first=True)
    
    # 2. Encode Y if Class
    if is_classification:
        le = LabelEncoder()
        # Force conversion to string to handle mixed types/integers safely
        y_final = le.fit_transform(y_raw.astype(str))
    else:
        # Fill Y if Reg
        y_final = y_raw.fillna(y_raw.mean())

    # Split (Stratify if classification to ensure class balance)
    test_size = st.sidebar.slider("Train/Test Split", 0.1, 0.5, 0.2)
    stratify_arg = y_final if is_classification else None
    
    # Handle edge case where stratify fails (e.g. only 1 sample of a class)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=test_size, random_state=42, stratify=stratify_arg)
    except ValueError:
        st.warning("Could not stratify split (class imbalance too high). Performing random split instead.")
        X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=test_size, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Keep as DF for feature names
    X_train_df = pd.DataFrame(X_train_s, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_s, columns=X_test.columns)

    def get_models(task):
        models = {}
        if task == 'Classification':
            models["Logistic Regression"] = LogisticRegression()
            models["Random Forest"] = RandomForestClassifier()
            models["Decision Tree"] = DecisionTreeClassifier()
            models["Support Vector Machine"] = SVC(probability=True)
            models["K-Nearest Neighbors"] = KNeighborsClassifier()
            models["Gradient Boosting"] = GradientBoostingClassifier()
            models["AdaBoost"] = AdaBoostClassifier()
            models["Naive Bayes"] = GaussianNB()
            if XGB_AVAILABLE: models["XGBoost"] = XGBClassifier(verbosity=0, use_label_encoder=False, eval_metric='logloss')
        else:
            models["Linear Regression"] = LinearRegression()
            models["Lasso Regression"] = Lasso()
            models["Ridge Regression"] = Ridge()
            models["Random Forest"] = RandomForestRegressor()
            models["Decision Tree"] = DecisionTreeRegressor()
            models["Support Vector Machine"] = SVR()
            models["K-Nearest Neighbors"] = KNeighborsRegressor()
            models["Gradient Boosting"] = GradientBoostingRegressor()
            models["AdaBoost"] = AdaBoostRegressor()
            if XGB_AVAILABLE: models["XGBoost"] = XGBRegressor(verbosity=0)
        return models

    # ================= TAB 3: MODEL TRAINING =================
    with tab3:
        st.header("Train & Evaluate")
        
        models_dict = get_models('Classification' if is_classification else 'Regression')
        selected_model_name = st.selectbox("Select Algorithm", list(models_dict.keys()))
        
        # Optional Hyperparams
        with st.expander("Hyperparameters (Optional)"):
            st.info("Using optimized default parameters for AutoML.")

        if st.button("🚀 Train Model", key="train_btn"):
            with st.spinner("Training in progress..."):
                try:
                    model = models_dict[selected_model_name]
                    model.fit(X_train_df, y_train)
                    preds = model.predict(X_test_df)
                    
                    # Metrics
                    st.success(f"**{selected_model_name}** Trained Successfully!")
                    met1, met2, met3, met4 = st.columns(4)
                    
                    if is_classification:
                        acc = accuracy_score(y_test, preds)
                        f1 = f1_score(y_test, preds, average='weighted')
                        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
                        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
                        
                        met1.metric("Accuracy", f"{acc:.2%}")
                        met2.metric("F1 Score", f"{f1:.2%}")
                        met3.metric("Precision", f"{prec:.2%}")
                        met4.metric("Recall", f"{rec:.2%}")
                        
                        # Plots
                        c_plot1, c_plot2 = st.columns(2)
                        with c_plot1:
                            cm = confusion_matrix(y_test, preds)
                            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
                            st.plotly_chart(fig_cm, use_container_width=True)
                        
                        with c_plot2:
                            if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
                                # ROC Curve for Binary
                                try:
                                    from sklearn.metrics import roc_curve, auc
                                    probs = model.predict_proba(X_test_df)[:, 1]
                                    # Handle edge case where y_test is not 0/1 (should not happen with LabelEncoder but safe to wrap)
                                    fpr, tpr, _ = roc_curve(y_test, probs, pos_label=1)
                                    roc_auc = auc(fpr, tpr)
                                    fig_roc = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC={roc_auc:.2f})', labels=dict(x='False Positive Rate', y='True Positive Rate'))
                                    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                                    st.plotly_chart(fig_roc, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not plot ROC: {e}")
                            else:
                                st.info("ROC Curve available for Binary Classification with 2 classes in Test set.")

                    else: # Regression
                        mse = mean_squared_error(y_test, preds)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, preds)
                        mae = mean_absolute_error(y_test, preds)
                        
                        met1.metric("R² Score", f"{r2:.4f}")
                        met2.metric("RMSE", f"{rmse:.4f}")
                        met3.metric("MSE", f"{mse:.4f}")
                        met4.metric("MAE", f"{mae:.4f}")
                        
                        # Regression Plots
                        c_plot1, c_plot2 = st.columns(2)
                        with c_plot1:
                            fig_sc = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted")
                            fig_sc.add_shape(type="line", line=dict(color="red", dash="dash"), x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max())
                            st.plotly_chart(fig_sc, use_container_width=True)
                        
                        with c_plot2:
                            residuals = y_test - preds
                            fig_res = px.scatter(x=preds, y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'}, title="Residual Plot")
                            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_res, use_container_width=True)

                    # Feature Importance
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("Feature Importance")
                        fi = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_ if len(model.feature_importances_) == len(feature_cols) else model.feature_importances_[:len(feature_cols)]}) # Safety Check
                        # Adjust if one-hot encoding changed cols:
                        if len(model.feature_importances_) == len(X_train_df.columns):
                             fi = pd.DataFrame({'Feature': X_train_df.columns, 'Importance': model.feature_importances_})
                        
                        fi = fi.sort_values(by='Importance', ascending=False).head(10)
                        fig_fi = px.bar(fi, x='Importance', y='Feature', orientation='h', title="Top 10 Feature Importances", color='Importance', color_continuous_scale='Viridis')
                        st.plotly_chart(fig_fi, use_container_width=True)
                except Exception as e:
                    st.error(f"Training failed: {e}")

    # ================= TAB 4: MODEL COMPARISON =================
    with tab4:
        st.header("🏆 Auto-Compare All Models")
        st.write("Train and rank all available algorithms to find the best performer.")
        
        if st.button("Run Full Comparison"):
            models_dict = get_models('Classification' if is_classification else 'Regression')
            results = []
            
            progress_bar = st.progress(0)
            
            for i, (name, model) in enumerate(models_dict.items()):
                try:
                    model.fit(X_train_df, y_train)
                    preds = model.predict(X_test_df)
                    
                    if is_classification:
                        metric = accuracy_score(y_test, preds)
                        results.append({"Model": name, "Accuracy": metric})
                    else:
                        metric = r2_score(y_test, preds)
                        results.append({"Model": name, "R2 Score": metric})
                except Exception as e:
                    pass # Skip failed models
                
                progress_bar.progress((i + 1) / len(models_dict))
            
            results_df = pd.DataFrame(results).sort_values(by="Accuracy" if is_classification else "R2 Score", ascending=False)
            
            st.balloons()
            st.success("Comparison Complete!")
            
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.dataframe(results_df, use_container_width=True)
            with col_res2:
                metric_name = "Accuracy" if is_classification else "R2 Score"
                fig_comp = px.bar(results_df, x="Model", y=metric_name, color=metric_name, title=f"Model Performance Ranking ({metric_name})", color_continuous_scale='Magma')
                st.plotly_chart(fig_comp, use_container_width=True)
                st.success(f"🏆 Best Model: **{results_df.iloc[0]['Model']}**")

else:
    st.info("Please upload a CSV file or use the Example Dataset to begin.")