# 🛡️ SIFGuard
**From Reports to Rescue — Detecting Fatal Precursors Before Incidents Happen**

[![Live App](https://img.shields.io/badge/Live-App-FF6B00?style=for-the-badge&logo=streamlit)](https://sifguard-a36gngxpezx5wzk4ghevhm.streamlit.app/)

Welcome to **SIFGuard**! This is an AI-powered safety analytics platform designed for Oil India Limited (SIH Problem Statement #26165). SIFGuard identifies Fatal Precursors hidden within unstructured safety reports by extracting linguistic nuances and mapping them to standard safety frameworks.

## 🚀 Live Demo
Access the live deployed application here:
**[https://sifguard-a36gngxpezx5wzk4ghevhm.streamlit.app/](https://sifguard-a36gngxpezx5wzk4ghevhm.streamlit.app/)**

## 🧠 The AI Pipeline
1. **Transliteration & Normalization**: Converts Hinglish/Assamese vocabulary into English semantic equivalents.
2. **Zero-Shot NLP Inference**: Uses `facebook/bart-large-mnli` to evaluate the probability of a Serious Incident or Fatality `P(SIF)`.
3. **Pattern Matching Engine**: Scans text for violations against the 9 IOGP Life-Saving Rules.
4. **Barrier Failure Detection**: Identifies critical safety barriers (e.g., Permits, Isolation, PPE) that failed.
5. **FPHI Scoring Model**: Combines P(SIF) with rule and barrier failures to generate a composite Fatal Precursor Hazard Index (0-100).

## 💻 Tech Stack
- **Python** (Pandas, Numpy)
- **Streamlit** (Web Interface)
- **HuggingFace Transformers / Torch** (NLP Inference)
- **Plotly & PyDeck** (Data Visualization & 3D Mapping)

## 🛠️ Run Locally
```bash
# Clone the repository
git clone https://github.com/Trupal-Shende/SIFGuard.git
cd SIFGuard

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---
Built with ❤️ by **Hacktivate Syndicate** for Smart Automation (Ministry of Petroleum & Natural Gas).
