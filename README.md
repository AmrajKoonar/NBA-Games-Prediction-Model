# 🏀 NBA Games Prediction Model

A Python machine learning program built to predict the outcomes of NBA games using historical data, box scores, and team standings. Created by **Amraj Koonar**, this project combines web scraping, data analysis, and predictive modeling to explore sports analytics.

---

## 🎯 Features

- 🧠 **ML-Powered Outcome Prediction**: Predicts NBA game results using classification models from Scikit-learn.
- 🌐 **Web Scraping Engine**: Collects game data (box scores, standings) from [basketball-reference.com](https://www.basketball-reference.com/).
- 📊 **Dataset Engineering**: Organizes scraped data into structured CSV files for modeling and evaluation.
- 📁 **Dual Data Sources**: Automatically stores data in two categories — `Scores` and `Standings`.
- 📈 **Model Evaluation**: Analyzes prediction performance using metrics like accuracy, confusion matrices, and visualizations.
- 🔄 **Repeatable Pipeline**: Can be rerun as new games are played, enabling model retraining with up-to-date data.

---

## 🧠 Tech Stack

- **Language**: Python
- **Libraries**: Scikit-learn, Pandas, Matplotlib, Requests, BeautifulSoup
- **Tools**: Jupyter Notebook or script execution
- **Data Source**: [Basketball-Reference](https://www.basketball-reference.com/)

---

## 🛠️ Setup & Installation

To run this project locally:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/nba-prediction-model.git
cd nba-prediction-model
```

### 2. (Optional) Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Script
```bash
python main.py
```

Make sure your `.py` files or notebooks include valid scraping URLs and are authorized to fetch external content.

---

## 🗃️ Sample Data Sources

- [NBA 2016 Game Results](https://www.basketball-reference.com/leagues/NBA_2016_games.html)
- [Box Score Example](https://www.basketball-reference.com/boxscores/201510270CHI.html)
- [Standings + Team Stats](https://www.basketball-reference.com/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) and is for educational and non-commercial use only.
