# MOTOAI — Flask Deployment Guide

## Project Structure
```
project/
├── app.py              # Flask backend + /predict ML endpoint
├── model.py            # ML trainer (run once to generate model.pkl)
├── cars.csv            # Training data
├── requirements.txt    # Python dependencies
├── Procfile            # For Heroku/Render deployment
├── templates/
│   └── motoai.html     # Frontend (served by Flask)
└── static/             # Place any static assets here (CSS/JS/images)
```

## Setup & Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the ML model (generates model.pkl)
python model.py

# 3. Start the Flask server
python app.py
```

Visit: http://localhost:5000

## Deploy to Render / Heroku

1. Push this folder to a Git repo
2. Set build command: `pip install -r requirements.txt && python model.py`
3. Set start command: `gunicorn app:app` (or use the Procfile)
4. The app reads `PORT` from environment automatically

## Environment Variables
- `PORT` — server port (default: 5000)
