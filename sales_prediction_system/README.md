# 📈 Sales Prediction System

A full-stack web application that predicts product sales based on advertising budgets across TV, Radio, and Newspaper channels using **Linear Regression**.

Built with **Django** (backend) + **scikit-learn** (ML) + **vanilla HTML/CSS** (frontend).

---

## 👥 Project Team

| Name | Role |
|------|------|
| Susmita Biswas | ML Developer |
| Jotirmoy Mondal | ML Developer & Full Stack Developer |
| Alinda Paul | Data Analyst |
| Suchi Nondi | Data Analyst |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/sales-prediction-system.git
cd sales-prediction-system
```

### 2. Create virtual environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the ML model
```bash
python ml_model/train_model.py
```
This downloads the Advertising dataset, trains a Linear Regression model, and saves `ml_model/model.pkl`.

### 4. Run Django migrations
```bash
python manage.py migrate
```

### 5. Start the development server
```bash
python manage.py runserver
```

Open **http://127.0.0.1:8000** in your browser.

---

## 🗂 Project Structure

```
sales-prediction-system/
│
├── manage.py
├── requirements.txt
├── README.md
│
├── sales_prediction_project/       # Django project config
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── sales_app/                      # Main Django app
│   ├── views.py                    # Prediction logic
│   ├── models.py
│   ├── urls.py
│   └── templates/
│       └── sales_app/
│           └── index.html          # Frontend UI
│
└── ml_model/
    ├── train_model.py              # Training script
    └── model.pkl                   # Saved model (auto-generated)
```

---

## 🧠 Machine Learning

- **Dataset**: [Advertising.csv](https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv) — 200 records
- **Algorithm**: Multiple Linear Regression
- **Features**: TV budget ($K), Radio budget ($K), Newspaper budget ($K)
- **Target**: Sales (units in thousands)
- **Train/Test Split**: 80% / 20% (random_state=42)

### Model Performance (approximate)
| Metric | Value |
|--------|-------|
| MAE | ~1.27 |
| MSE | ~2.91 |
| RMSE | ~1.71 |

---

## 📸 Features

- Enter TV, Radio, and Newspaper advertising budgets
- Instantly get predicted sales volume
- Color-coded performance tier (Low / Moderate / High)
- Responsive design — works on desktop and mobile
- Falls back to hardcoded coefficients if `model.pkl` is missing

---

## 📄 License

Academic project — free to use for educational purposes.
