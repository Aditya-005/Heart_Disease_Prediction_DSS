"""
Professional Streamlit UI for AI-Driven Predictive Healthcare System
Heart Disease Risk Assessment Application
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import tempfile
import os
from datetime import datetime

# Add parent directory to path to import main module
sys.path.append(str(Path(__file__).parent))

from main import (
    load_ecg, preprocess_ecg, detect_rpeaks, extract_ecg_features,
    preprocess_clinical, build_feature_vector, load_model, predict,
    decision_logic, save_prediction, process_full_pipeline
)

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Assessment | AI Healthcare DSS",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': None,
        'About': "AI-Driven Predictive Healthcare System for Heart Disease Risk Assessment"
    }
)

# Load custom CSS
def load_css():
    css_path = Path(__file__).parent / "static" / "css" / "main.css"
    if css_path.exists():
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Fallback inline CSS
        st.markdown("""
        <style>
            .main-header {
                font-size: 3rem;
                font-weight: 800;
                text-align: center;
                background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 2rem 0;
            }
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 0.75rem;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            }
            .stButton>button {
                background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
                color: white;
                border: none;
                border-radius: 0.5rem;
                padding: 0.75rem 2rem;
                font-weight: 600;
            }
        </style>
        """, unsafe_allow_html=True)

load_css()

# Load custom JavaScript
def load_js():
    js_path = Path(__file__).parent / "static" / "js" / "main.js"
    if js_path.exists():
        with open(js_path, 'r') as f:
            st.markdown(f'<script>{f.read()}</script>', unsafe_allow_html=True)

load_js()


@st.cache_data
def load_sample_ecg():
    """Load sample ECG data if available."""
    sample_path = Path(__file__).parent / "sample_data" / "sample_ecg.csv"
    if sample_path.exists():
        return sample_path
    return None


def plot_interactive_ecg(signal, peaks, fs, title="ECG Signal with R-Peaks"):
    """Create interactive ECG plot using Plotly."""
    time_axis = np.arange(len(signal)) / fs
    
    fig = go.Figure()
    
    # ECG Signal
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=signal,
        mode='lines',
        name='ECG Signal',
        line=dict(color='#2563eb', width=1),
        hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.4f}<extra></extra>'
    ))
    
    # R-Peaks
    if len(peaks) > 0:
        fig.add_trace(go.Scatter(
            x=time_axis[peaks],
            y=signal[peaks],
            mode='markers',
            name='R-Peaks',
            marker=dict(
                color='#ef4444',
                size=8,
                symbol='circle',
                line=dict(width=1, color='white')
            ),
            hovertemplate='R-Peak<br>Time: %{x:.2f}s<br>Amplitude: %{y:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Arial, sans-serif'}
        },
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
        hovermode='closest',
        template='plotly_white',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def create_risk_gauge(probability):
    """Create an interactive gauge chart for risk probability."""
    risk_level = "Low" if probability < 0.3 else "Medium" if probability < 0.7 else "High"
    colors = ['#10b981', '#f59e0b', '#ef4444']
    color = colors[0] if probability < 0.3 else colors[1] if probability < 0.7 else colors[2]
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Probability (%)", 'font': {'size': 24}},
        delta = {'reference': 50, 'position': "top"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': 'lightgray'},
                {'range': [30, 70], 'color': 'gray'},
                {'range': [70, 100], 'color': 'darkgray'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        font={'color': "#1e293b", 'family': "Arial, sans-serif"}
    )
    
    return fig, risk_level


def main():
    """Main Streamlit application."""
    
    # Header with gradient
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; padding: 0;">❤️ AI Heart Disease Risk Assessment</h1>
        <p style="font-size: 1.2rem; font-weight: 400; margin-top: 0.5rem; opacity: 0.9;">
            Advanced Machine Learning-Based Predictive Healthcare System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Info banner
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ Professional Healthcare DSS:</strong> This system combines ECG signal analysis with clinical 
        parameters to provide comprehensive heart disease risk assessment using state-of-the-art machine learning.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with enhanced design
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: white; margin: 0;">⚙️ Settings</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        model_path = st.text_input(
            "🔧 Model Path", 
            value="models/baseline_model.pkl",
            help="Path to the trained model file"
        )
        
        st.markdown("---")
        
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
            <h4 style="color: white; margin-top: 0;">📋 Instructions</h4>
            <ol style="color: white; font-size: 0.9rem;">
                <li>Enter clinical parameters</li>
                <li>Upload or use sample ECG</li>
                <li>Click "Run Assessment"</li>
                <li>Review detailed results</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model status
        model_exists = Path(model_path).exists() if model_path else False
        if model_exists:
            st.success("✅ Model loaded successfully")
        else:
            st.warning("⚠️ Model not found. Training may be required.")
    
    # Main content layout
    tab1, tab2, tab3 = st.tabs(["🔍 Assessment", "📊 Analysis", "ℹ️ About"])
    
    with tab1:
        # Two column layout for input forms
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("""
            <div class="form-container">
                <h2 class="section-header">📋 Clinical Parameters</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Clinical input form with better styling
            with st.form("clinical_form", clear_on_submit=False):
                st.markdown("### 👤 Patient Information")
                
                # Personal Info
                age = st.number_input(
                    "Age (years)", 
                    min_value=1, max_value=120, value=50, step=1,
                    help="Patient's age in years"
                )
                
                gender = st.selectbox(
                    "Gender", 
                    options=["Unknown", "Male", "Female"],
                    help="Patient's gender"
                )
                gender_map = {"Unknown": "unknown", "Male": "M", "Female": "F"}
                
                st.markdown("### 💓 Vital Signs")
                
                # Blood Pressure
                bp_col1, bp_col2 = st.columns(2)
                with bp_col1:
                    bp_systolic = st.number_input(
                        "Systolic BP (mmHg)", 
                        min_value=70, max_value=250, value=120, step=1,
                        help="Systolic blood pressure"
                    )
                with bp_col2:
                    bp_diastolic = st.number_input(
                        "Diastolic BP (mmHg)", 
                        min_value=40, max_value=150, value=80, step=1,
                        help="Diastolic blood pressure"
                    )
                
                # Cholesterol
                cholesterol = st.number_input(
                    "Total Cholesterol (mg/dL)", 
                    min_value=100, max_value=400, value=200, step=1,
                    help="Total cholesterol level"
                )
                
                st.markdown("### 📏 Body Metrics")
                
                # BMI Calculation
                bmi_col1, bmi_col2 = st.columns(2)
                with bmi_col1:
                    height = st.number_input(
                        "Height (cm)", 
                        min_value=100, max_value=250, value=170, step=1,
                        help="Patient's height in centimeters"
                    )
                with bmi_col2:
                    weight = st.number_input(
                        "Weight (kg)", 
                        min_value=30, max_value=200, value=70, step=1,
                        help="Patient's weight in kilograms"
                    )
                
                bmi = weight / ((height / 100) ** 2)
                st.metric("BMI", f"{bmi:.1f}")
                
                st.markdown("### 🚬 Lifestyle")
                
                smoking = st.selectbox(
                    "Smoking Status", 
                    options=["Unknown", "Yes", "No"],
                    help="Current smoking status"
                )
                smoking_map = {"Unknown": "unknown", "Yes": "yes", "No": "no"}
                
                st.markdown("---")
                
                submitted = st.form_submit_button(
                    "🔍 Run Risk Assessment", 
                    use_container_width=True,
                    type="primary"
                )
        
        with col2:
            st.markdown("""
            <div class="form-container">
                <h2 class="section-header">📊 ECG Data Source</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # ECG upload options with better UI
            ecg_option = st.radio(
                "Select ECG Source",
                options=["Upload File", "Use Sample Data"],
                horizontal=True,
                help="Choose to upload your own ECG file or use sample data"
            )
            
            ecg_file = None
            ecg_path = None
            
            if ecg_option == "Upload File":
                st.markdown("### 📤 Upload ECG File")
                uploaded_file = st.file_uploader(
                    "Choose ECG file", 
                    type=['csv', 'dat', 'hea', 'txt'],
                    help="Supported formats: CSV, WFDB (.dat/.hea), TXT"
                )
                if uploaded_file is not None:
                    # Show file info
                    file_size = len(uploaded_file.read())
                    uploaded_file.seek(0)
                    
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.info(f"📄 {uploaded_file.name}")
                    with col_info2:
                        st.info(f"📦 {file_size / 1024:.2f} KB")
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        ecg_path = tmp_file.name
                        st.success("✅ File uploaded successfully")
            else:
                st.markdown("### 📥 Sample Data")
                sample_path = load_sample_ecg()
                if sample_path:
                    ecg_path = str(sample_path)
                    st.success(f"✅ Using sample ECG: {sample_path.name}")
                    st.info("This is a demonstration ECG signal for testing purposes.")
                else:
                    st.warning("⚠️ Sample ECG not found. Please upload your own ECG file.")
                    ecg_path = None
            
            # File format guide
            with st.expander("📖 Supported File Formats"):
                st.markdown("""
                **CSV Format:**
                - Single column of numerical values
                - One value per line
                - No header row
                
                **WFDB Format:**
                - `.dat` file (signal data)
                - `.hea` file (header with metadata)
                - Must be in same directory
                
                **TXT Format:**
                - Plain text with numerical values
                - Space or comma separated
                """)
    
    # Process when form is submitted
    if submitted and ecg_path:
        try:
            with st.spinner("🔄 Processing ECG signal and running AI prediction... This may take a few moments."):
                # Prepare clinical data
                clinical_dict = {
                    'age': age,
                    'bp_systolic': bp_systolic,
                    'bp_diastolic': bp_diastolic,
                    'cholesterol': cholesterol,
                    'gender': gender_map[gender],
                    'height': height,
                    'weight': weight,
                    'smoking': smoking_map[smoking]
                }
                
                # Run full pipeline
                results = process_full_pipeline(ecg_path, clinical_dict, model_path if model_path else None)
                
                # Results Section
                st.markdown("---")
                st.markdown("""
                <div class="result-container">
                    <h2 class="result-header">📈 Prediction Results</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk category display with gauge
                prob = results['probability']
                category = results['risk_category']
                
                # Main metrics in a grid
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.plotly_chart(
                        create_risk_gauge(prob)[0], 
                        use_container_width=True,
                        config={'displayModeBar': False}
                    )
                
                with col_res2:
                    risk_class = f"risk-{category.lower()}"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 2rem;">
                        <p style="font-size: 1.2rem; color: #64748b; margin-bottom: 0.5rem;">Risk Category</p>
                        <span class="risk-badge {risk_class}" style="font-size: 1.5rem; padding: 1rem 2rem;">
                            {category.upper()}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res3:
                    confidence = abs(prob - 0.5) * 2
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Recommendation with enhanced styling
                recommendation_icon = "✅" if category == "Low" else "⚠️" if category == "Medium" else "🚨"
                st.markdown(f"""
                <div class="recommendation-card">
                    <h3 style="margin-top: 0;">
                        {recommendation_icon} Recommendation
                    </h3>
                    <p style="font-size: 1.1rem; line-height: 1.8;">
                        {results['recommendation']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed Analysis Tab
                with tab2:
                    st.markdown("## 📊 Detailed Analysis")
                    
                    # ECG Visualization with interactive plot
                    st.markdown("### 📉 ECG Signal Analysis")
                    
                    # Reload and process ECG for visualization
                    signal, fs = load_ecg(ecg_path)
                    signal, fs = preprocess_ecg(signal, fs)
                    peaks = detect_rpeaks(signal, fs)
                    
                    fig_ecg = plot_interactive_ecg(signal, peaks, fs, "Interactive ECG Signal with Detected R-Peaks")
                    st.plotly_chart(fig_ecg, use_container_width=True)
                    
                    # ECG Statistics
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        st.metric("Heart Rate", f"{results['ecg_features'].get('heart_rate', 0):.1f} BPM")
                    with col_stat2:
                        st.metric("R-Peaks Detected", len(peaks))
                    with col_stat3:
                        st.metric("Signal Duration", f"{len(signal) / fs:.1f} s")
                    with col_stat4:
                        st.metric("Sampling Rate", f"{fs} Hz")
                    
                    # Feature tables with better styling
                    col_feat1, col_feat2 = st.columns(2)
                    
                    with col_feat1:
                        st.markdown("### 🏥 Clinical Features")
                        clinical_df = pd.DataFrame([results['clinical_features']])
                        clinical_df = clinical_df.T
                        clinical_df.columns = ['Value']
                        clinical_df.index.name = 'Feature'
                        clinical_df = clinical_df.round(4)
                        st.dataframe(
                            clinical_df, 
                            use_container_width=True,
                            height=400
                        )
                    
                    with col_feat2:
                        st.markdown("### 📊 ECG Features")
                        ecg_df = pd.DataFrame([results['ecg_features']])
                        ecg_df = ecg_df.T
                        ecg_df.columns = ['Value']
                        ecg_df.index.name = 'Feature'
                        ecg_df = ecg_df.round(4)
                        st.dataframe(
                            ecg_df, 
                            use_container_width=True,
                            height=400
                        )
                    
                    # Feature importance visualization
                    st.markdown("### 📈 Feature Contribution")
                    all_features = {**results['clinical_features'], **results['ecg_features']}
                    feature_df = pd.DataFrame(list(all_features.items()), columns=['Feature', 'Value'])
                    feature_df = feature_df.sort_values('Value', key=abs, ascending=False).head(10)
                    
                    fig_bar = px.bar(
                        feature_df,
                        x='Value',
                        y='Feature',
                        orientation='h',
                        color='Value',
                        color_continuous_scale='RdBu',
                        title="Top 10 Feature Contributions"
                    )
                    fig_bar.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Export Results
                st.markdown("---")
                st.markdown("### 💾 Export Results")
                
                col_export1, col_export2 = st.columns(2)
                
                prediction_record = {
                    'timestamp': pd.Timestamp.now(),
                    'age': age,
                    'bp_systolic': bp_systolic,
                    'bp_diastolic': bp_diastolic,
                    'cholesterol': cholesterol,
                    'gender': gender,
                    'bmi': results['clinical_features'].get('bmi', 0),
                    'probability': prob,
                    'risk_category': category,
                    'heart_rate': results['ecg_features'].get('heart_rate', 0),
                    'hrv_sdnn': results['ecg_features'].get('hrv_sdnn', 0),
                    'recommendation': results['recommendation']
                }
                
                df_export = pd.DataFrame([prediction_record])
                csv = df_export.to_csv(index=False)
                
                with col_export1:
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"heart_risk_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col_export2:
                    if st.button("💾 Save to Database", use_container_width=True):
                        save_prediction(prediction_record)
                        st.success("✅ Prediction saved to predictions.csv")
                
                # Clean up temporary file
                if ecg_option == "Upload File" and os.path.exists(ecg_path):
                    try:
                        os.unlink(ecg_path)
                    except:
                        pass
        
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.exception(e)
    
    elif submitted and not ecg_path:
        st.error("⚠️ Please upload an ECG file or select sample data to proceed.")
    
    # About Tab
    with tab3:
        st.markdown("""
        ## ℹ️ About This System
        
        ### AI-Driven Predictive Healthcare System
        
        This **Decision Support System (DSS)** leverages advanced machine learning algorithms to analyze 
        clinical parameters and ECG signals, providing comprehensive heart disease risk assessment.
        
        ### ✨ Key Features
        
        - **🔬 ECG Signal Processing**: Advanced R-peak detection using Pan-Tompkins algorithm
        - **📊 Heart Rate Variability Analysis**: Comprehensive HRV metrics extraction
        - **🤖 Machine Learning Models**: Logistic Regression and Random Forest classifiers
        - **📈 Risk Categorization**: Low, Medium, and High risk classifications
        - **💡 Actionable Recommendations**: Evidence-based clinical guidance
        - **📱 Interactive Visualizations**: Real-time ECG signal analysis
        
        ### 📋 Supported ECG Formats
        
        - **CSV Files**: Single column of numerical values, one per line
        - **WFDB Format**: Standard PhysioNet format (.dat/.hea files)
        - **Text Files**: Space or comma-separated numerical values
        
        ### ⚠️ Important Disclaimer
        
        This system is designed for **research and educational purposes** only. It should not replace 
        professional medical consultation, diagnosis, or treatment. Always consult qualified healthcare 
        professionals for medical decisions.
        
        ### 🔬 Technical Details
        
        - **Framework**: Streamlit, Python
        - **ML Libraries**: scikit-learn, NumPy, Pandas
        - **Signal Processing**: SciPy, WFDB
        - **Visualization**: Plotly, Matplotlib
        
        ### 📞 Support
        
        For technical support or questions, please refer to the project documentation or contact 
        the development team.
        """)
        
        # Model Information
        with st.expander("🤖 Model Information"):
            st.markdown("""
            **Current Model:** Logistic Regression (Baseline)
            
            **Training Data:** Synthetic ECG dataset with clinical parameters
            
            **Performance Metrics:**
            - Accuracy: 100%
            - ROC-AUC: 1.0000
            
            **Features Used:**
            - 7 Clinical Features (Age, BP, Cholesterol, Gender, BMI, Smoking)
            - 8 ECG Features (HR, HRV, R-R intervals, etc.)
            - Total: 15 features
            """)


if __name__ == "__main__":
    main()
