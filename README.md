# ❤️ AI Heart Disease Risk Assessment System

A professional, AI-driven Decision Support System (DSS) for heart disease risk assessment using ECG signal analysis and clinical parameters.

## ✨ Features

- 🔬 **Advanced ECG Processing**: Pan-Tompkins R-peak detection algorithm
- 📊 **Heart Rate Variability Analysis**: Comprehensive HRV metrics
- 🤖 **Machine Learning Models**: Logistic Regression & Random Forest
- 📈 **Risk Categorization**: Low/Medium/High risk classification
- 💡 **Actionable Recommendations**: Evidence-based clinical guidance
- 🎨 **Professional UI**: Modern, interactive Streamlit interface
- 📱 **Interactive Visualizations**: Real-time ECG signal analysis with Plotly

## 🚀 Quick Start

### 1. Setup Environment

**Windows:**
```bash
.\setup_venv.bat
```

**Linux/Mac:**
```bash
bash setup_venv.sh
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model (Optional - Uses Synthetic Data)

**Windows:**
```bash
.\train_model.bat
```

**Or manually:**
```bash
python generate_training_dataset.py --num-samples 500
python train_real_data.py
```

### 4. Launch Application

**Windows:**
```bash
.\start_streamlit.bat
```

**Linux/Mac:**
```bash
bash start_streamlit.sh
```

Then open your browser at `http://localhost:8501`

## 📁 Project Structure

```
DSS_Project/
├── app.py                      # Streamlit UI application
├── main.py                     # Core processing logic
├── train_real_data.py          # Training script
├── static/                     # Frontend assets
│   ├── css/main.css           # Professional styles
│   └── js/main.js             # Interactive features
├── sample_data/                # Sample ECG files
├── training_data/              # Training data (structure)
│   ├── ecg_files/            # Place ECG CSV files here
│   └── clinical_data/         # Place clinical data CSV here
├── models/                     # Trained models (generated)
└── requirements.txt           # Python dependencies
```

## 📋 Usage

1. **Enter Clinical Parameters**: Age, BP, Cholesterol, BMI, etc.
2. **Upload ECG Data**: CSV or WFDB format, or use sample data
3. **Run Assessment**: Click "Run Risk Assessment"
4. **Review Results**: Interactive visualizations and detailed analysis
5. **Export Results**: Download predictions as CSV

## 🔬 Supported ECG Formats

- **CSV**: Single column of numerical values
- **WFDB**: PhysioNet format (.dat/.hea files)
- **TXT**: Plain text with numerical values

## 🛠️ Technologies

- **Python 3.10+**
- **Streamlit** - Web UI framework
- **scikit-learn** - Machine learning
- **Plotly** - Interactive visualizations
- **NumPy, Pandas** - Data processing
- **SciPy, WFDB** - Signal processing

## 📝 Training with Your Data

See `training_data/README_TRAINING_DATA.md` for detailed instructions on training with real ECG data.

## ⚠️ Disclaimer

This system is for **research and educational purposes only**. It should not replace professional medical consultation, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

## 📄 License

[Add your license here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues or questions, please open an issue on GitHub.

---

**Built with ❤️ for better healthcare decision support**
