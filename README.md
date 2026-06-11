# 🚀 InsightForge AI

> **AI-Powered Exploratory Data Analysis Platform**  
> Transform raw data into business insights with natural language queries, automated cleaning, and professional reports.

[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF424D.svg?style=flat&logo=datastreamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python)](https://python.org)
[![Gemini API](https://img.shields.io/badge/Gemini-API-4285F4.svg?style=flat&logo=google)](https://ai.google.dev)
[![Plotly](https://img.shields.io/badge/Plotly-5.20-29B9D3.svg?style=flat&logo=plotly)](https://plotly.com)

---

## 🎯 What is InsightForge AI?

**InsightForge AI** is a full-stack data analysis platform that empowers anyone to:

- 🗣️ **Chat with your data** using natural language (e.g., "Which category generated the most revenue?")
- 🧹 **Automatically clean datasets** with 10+ strategies (missing values, outliers, encoding, scaling)
- 📊 **Generate interactive visualizations** (histograms, boxplots, heatmaps, scatter plots)
- 💡 **Get AI-powered business insights** in seconds
- 📄 **Export professional PDF reports** with quality scores and recommendations
- 🎯 **Calculate dataset health scores** (0-100) with actionable improvements

**Think of it as:** Power BI + ChatGPT + DataRobot **in a single platform.**

---

## ✨ Key Features

### 🤖 **AI Chat with Dataset**
Ask questions in natural language and get instant insights:
- "Which category has the highest sales?"
- "Show me columns with missing values"
- "What are the top 5 outliers?"
- "Give me business recommendations"

### 🧹 **Automated Data Cleaning Studio**
Clean messy data with one click:
- **Missing Values:** Remove rows/columns, impute mean/median/mode
- **Duplicates:** Detect and remove automatically
- **Outliers:** IQR & Z-score detection, cap or remove
- **Encoding:** Label encode & one-hot encode categorical variables
- **Scaling:** StandardScaler & MinMaxScaler

### 📊 **Advanced Analytics**
- Dataset summary (rows, columns, types)
- Missing value analysis with visualizations
- Correlation heatmaps
- Statistical summaries (mean, median, std, quartiles)
- Categorical distribution analysis
- Outlier detection

### 💡 **AI Smart Recommendations**
Auto-generated insights based on data patterns:
- Skewness detection → "Apply log transformation"
- High correlations → "Use for predictive modeling"
- Outlier alerts → "Cap extreme values"
- Missing value strategies → "Impute with median"

### 📈 **Dataset Quality Score**
Instant health score (0-100) with:
- ✅ Strengths (no missing values, no duplicates)
- ⚠️ Weaknesses (outliers, inconsistencies)
- 🔹 Recommendations (imputation, outlier handling)

### 📄 **Professional PDF Reports**
Generate downloadable reports with:
- Cover page with dataset name & timestamp
- Executive summary
- Dataset overview (rows, columns, metrics)
- Quality score gauge
- AI business insights
- Smart recommendations
- Column information table
- Visualizations (histograms, heatmaps)

### 🎨 **Interactive Visualizations**
- **Histograms:** Distribution analysis
- **Boxplots:** Outlier detection
- **Correlation Heatmaps:** Feature relationships
- **Scatter Plots:** Variable correlations
- **Bar Charts:** Categorical distributions

---

## 🏗️ Architecture
InsightForgeAI/
│
├── app.py # Main Streamlit application
│
├── ai/ # AI & ML Modules
│ ├── gemini_client.py # Gemini API integration
│ ├── chat_engine.py # Natural language chat
│ ├── recommendations.py # AI-powered insights
│
├── analytics/ # Data Analytics
│ ├── quality_score.py # Dataset health scoring
│
├── reports/ # Report Generation
│ ├── pdf_generator.py # PDF report exporter
│
├── utils/ # Utility Functions
│ ├── data_utils.py # Dataset processing
│ ├── visualization.py # Plotly chart builders
│ ├── cleaning.py # Data cleaning engine
│
├── test_data/ # Sample datasets
│ ├── sales.csv
│ ├── employees.csv
│
├── .env # Environment variables (API keys)
├── requirements.txt # Dependencies
└── README.md # This file


---

## 🚀 Quick Start

### **1. Install Dependencies**

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Activate (Mac/Linux)
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### **2. Set Up API Key**

Create a `.env` file:

```env
GEMINI_API_KEY=your-actual-gemini-api-key-here
```

**Get your API key:** [Google AI Studio](https://ai.google.dev/)

### **3. Run the App**

```bash
streamlit run app.py
```

**Open in browser:** `http://localhost:8501`

---

## 🎨 Usage Guide

### **Mode 1: Quick EDA**
Fast dataset overview:
- Upload CSV/Excel/Parquet
- See instant metrics (rows, columns, missing, duplicates)
- View column information
- Check missing value analysis
- Preview data

### **Mode 2: Deep Analysis**
Full exploratory data analysis:
- Select numeric columns
- View histograms & boxplots
- See statistical summaries
- Analyze correlations
- Create scatter plots
- Explore categorical distributions

### **Mode 3: AI Insights**
AI-powered business intelligence:
- Generate business insights
- Create executive summary
- **Export professional PDF report**

### **Mode 4: AI Data Chat**
Natural language querying:
- Chat with your dataset
- Ask questions like:
  - "Which category generated the most revenue?"
  - "Show me columns with highest missing values"
  - "What trends do you see?"
- Get instant AI responses

### **Mode 5: Data Cleaning**
Automated data cleaning:
- Fix missing values (mean/median/mode)
- Remove duplicates
- Detect & handle outliers (IQR/Z-score)
- Encode categorical variables
- Scale numeric features
- Export cleaned dataset (CSV/Excel)

---

## 💻 Tech Stack

| Category | Technology |
|----------|------------|
| **Frontend** | Streamlit 1.32 |
| **Backend** | Python 3.12 |
| **Data** | Pandas 2.2, NumPy 1.26 |
| **Visualization** | Plotly 5.20 |
| **AI** | Gemini API 0.5 |
| **Reports** | ReportLab 4.1 |
| **Excel** | OpenPyXL 3.1 |
| **Environment** | Python-dotenv 1.0 |

---

## 🎯 Why InsightForge AI?

### **Traditional Data Analysis:**
1. Upload data to Excel
2. Manually calculate metrics
3. Create charts one by one
4. Write report in Word
5. Send to stakeholders (takes hours)

### **With InsightForge AI:**
1. Upload CSV (1 click)
2. Get instant insights (AI)
3. Visualizations auto-generated
4. Export PDF report (1 click)
5. Send to stakeholders (takes seconds)



**10x faster. 100% more professional.**

---

## 🧪 Testing

```bash
# Test all modules
python -c "
from utils.data_utils import *
from utils.visualization import *
from utils.cleaning import *
from ai.gemini_client import *
from ai.chat_engine import *
from ai.recommendations import *
from analytics.quality_score import *
print('✅ All modules loaded successfully!')
"
```

---

## 📈 Performance

- **Dataset Upload:** < 2 seconds (100K rows)
- **AI Insights:** 10-20 seconds (Gemini API)
- **Chart Generation:** < 1 second
- **PDF Report:** < 5 seconds
- **Memory Usage:** Optimized with caching

---

## 🚦 Roadmap

### **Phase 1 (Completed)**
- ✅ Core EDA features
- ✅ AI chat integration
- ✅ Data cleaning studio
- ✅ PDF report generator
- ✅ Quality score calculator

### **Phase 2 (Next)**
- 🔄 AutoML Studio (automatic model training)
- 🔄 Natural language visualization builder
- 🔄 Time series forecasting
- 🔄 Export center (CSV, Excel, JSON, PNG)

### **Phase 3 (Future)**
- 🚀 Multi-dataset comparison
- 🚀 Scheduled reports
- 🚀 API endpoints
- 🚀 Team collaboration

---

## 📄 License

This project is open source and available for educational and commercial use.

---

## 👨‍💻 Author

**Rishika**  
Aspiring Data Scientist and AI Engineer  
📧 [rishika.jan2007@gmail.com]  
💼 [https://www.linkedin.com/in/rishika-rai-65a22037a/]  
🚀 [https://github.com/Rishika-AQRS]

---

## 🌟 Demo

**Try it live:** [insightforge-ai.streamlit.app](https://your-app-url.streamlit.app)

---

## 🙏 Acknowledgments

- **Streamlit** - For the amazing web framework
- **Google Gemini** - For powerful AI capabilities
- **Plotly** - For beautiful visualizations
- **ReportLab** - For PDF generation

---

