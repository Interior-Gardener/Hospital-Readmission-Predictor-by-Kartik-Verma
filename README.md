# Hospital Readmission Predictor 🏥

A beautiful, AI-powered web application that predicts hospital readmission risk for diabetic patients using machine learning.

![Hospital Readmission Predictor](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost-orange?style=for-the-badge)

## ✨ Features

- **🎯 AI-Powered Predictions**: Advanced XGBoost model with 85.2% accuracy
- **📊 Interactive Visualizations**: Beautiful charts and risk assessment gauges
- **📈 Analytics Dashboard**: Track predictions and analyze patterns
- **🧠 Model Interpretability**: SHAP analysis for explainable AI
- **📱 Responsive Design**: Works perfectly on desktop and mobile
- **📋 Patient History**: Track multiple predictions over time
- **📥 Export Functionality**: Download prediction reports
- **🔒 Privacy-First**: No data stored permanently

## 🚀 Quick Start

### Option 1: Run Locally

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hospital_readmission_predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if needed)
   ```bash
   python train_model.py
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

### Option 2: Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy from your forked repository

## 📊 Model Performance

- **Accuracy**: 85.2%
- **Precision**: 82.1%
- **Recall**: 79.4%
- **F1-Score**: 80.7%
- **AUC-ROC**: 88.9%

## 🏥 Clinical Features

The model analyzes 13 key clinical factors:

- **Demographics**: Age, Gender
- **Admission Details**: Type, Source, Discharge disposition
- **Clinical Metrics**: Hospital stay duration, Lab procedures
- **Medical History**: Prior visits, Number of diagnoses
- **Lab Results**: Glucose levels, A1C results
- **Treatment**: Medication changes, Diabetes medication

## 🖼️ Screenshots

### Home Page - Patient Input
Beautiful, intuitive form for entering patient information with helpful tooltips and organized sections.

### Prediction Results
Comprehensive risk assessment with:
- Risk gauge visualization
- Detailed metrics
- Clinical recommendations
- Model explanations

### Analytics Dashboard
Track predictions over time with:
- Risk distribution charts
- Timeline analysis
- Statistical summaries

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS
- **Backend**: Python 3.8+
- **ML Model**: XGBoost Classifier
- **Visualization**: Plotly, Matplotlib
- **Model Interpretation**: SHAP
- **Data Processing**: Pandas, NumPy

## 📁 Project Structure

```
hospital_readmission_predictor/
├── app.py                      # Main Streamlit application
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── diabetic_data.csv          # Training dataset
├── readmission_model.pkl      # Trained model
├── model_columns.pkl          # Model feature columns
└── shap_summary_plot.png      # SHAP feature importance plot
```

## 🔒 Privacy & Security

- **No Personal Data Storage**: Patient information is not permanently stored
- **Local Processing**: All predictions happen locally
- **HIPAA Considerations**: Designed with healthcare privacy in mind
- **Secure Connections**: HTTPS encryption for deployed versions

## ⚕️ Medical Disclaimer

**Important**: This tool is designed to assist healthcare professionals and should not replace clinical judgment. Always consult with qualified healthcare providers for medical decisions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset from UCI Machine Learning Repository
- Built with Streamlit framework
- Machine learning powered by XGBoost
- Model interpretability by SHAP

## 📞 Support

If you have any questions or need help with deployment, please open an issue on GitHub.

---

**Made with ❤️ for healthcare professionals**