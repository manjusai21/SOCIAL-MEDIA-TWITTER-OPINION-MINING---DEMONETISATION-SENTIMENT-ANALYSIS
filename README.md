# Demonetization Sentiment Analysis & Narrative Mining

**Twitter Opinion Mining Project** – Identifying key narratives and sentiments around India’s 2016 Demonetization.

---

## 📌 Project Overview

This project analyzes public opinion on Twitter regarding the 2016 Indian Demonetization policy.  
It discovers coherent **narratives** (clusters of similar opinions) and assigns an overall **sentiment** (Positive / Negative / Neutral) to each narrative.

### Key Objectives
1. Clean and preprocess ~15,000 tweets related to `#Demonetization`
2. Perform sentiment analysis using TextBlob
3. Generate tweet embeddings using Word2Vec
4. Cluster tweets into meaningful narratives using K-Means
5. Extract characteristic phrases for each narrative
6. Label each narrative with overall sentiment

---

## 🗂️ Project Structure

```
demonetization-sentiment-analysis/
├── demonetization_sentiment_analysis.py   # Main analysis script
├── demonetization-tweets.csv              # Dataset (~15k tweets)
├── requirements.txt                       # Python dependencies
├── README.md
├── LICENSE
├── .gitignore
│
├── narrative.xlsx                         # Output: Narratives + cluster details
└── narrative_sentiment_count.png          # Output: Sentiment distribution chart
```

---

## 🛠️ Tech Stack

| Technology       | Purpose                          |
|------------------|----------------------------------|
| Python 3.9+      | Core language                    |
| pandas           | Data manipulation                |
| NLTK             | Tokenization & stop-word removal |
| TextBlob         | Sentiment polarity               |
| Gensim           | Word2Vec embeddings              |
| scikit-learn     | K-Means clustering & evaluation  |
| seaborn / matplotlib | Visualization                 |
| openpyxl         | Excel report generation          |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/demonetization-sentiment-analysis.git
cd demonetization-sentiment-analysis

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
python3 demonetization_sentiment_analysis.py
```

### Generated Outputs
- `narrative.xlsx` → Detailed narratives + top tweets per cluster
- `narrative_sentiment_count.png` → Bar chart of sentiment distribution

---

## 📊 Sample Results

| Cluster | Narrative Example       | Sentiment  |
|---------|-------------------------|------------|
| 0       | black money             | Negative   |
| 0       | common man              | Negative   |
| 0       | indian economy          | Positive   |
| 4       | huge support            | Positive   |
| 4       | across nation           | Positive   |
| 5       | overall impact          | Neutral    |
| 5       | education system        | Neutral    |

---

## 📁 Dataset

- **Source**: Public Twitter data collected around 23 November 2016
- **Size**: 14,940 tweets
- **Original source**: Also available on [Kaggle](https://www.kaggle.com/datasets/arathee2/demonetization-in-india-twitter-data)

---

## 📝 Notes

- The original notebook contained multiple errors (outdated APIs, broken regex, missing packages, deprecated methods).
- This repository contains a **fully corrected, modern, and production-ready** rewrite of the analysis pipeline.
- SSL certificate handling is included for smooth NLTK data download on macOS.

---

## 📄 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Original dataset and problem inspiration from the data science community (Kaggle / GitHub).
- Historical context from the Government of India announcement (8 November 2016).
