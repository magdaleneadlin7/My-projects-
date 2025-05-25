from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import os
import pandas as pd
from model import train_and_predict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import json
import yfinance as yf
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit
USER_DATA_FILE = 'users.json'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['csv']

def load_users():
    if not os.path.exists(USER_DATA_FILE):
        return {}
    with open(USER_DATA_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users, f)

def fetch_live_nifty_data(period='1mo', interval='1d'):
    """
    Fetch recent NIFTY 50 historical data using yfinance.
    Returns a DataFrame with 'Close' prices.
    """
    ticker = '^NSEI'  # Yahoo Finance ticker for NIFTY 50
    data = yf.download(ticker, period=period, interval=interval)
    if data.empty:
        raise ValueError("Failed to fetch live data for NIFTY 50.")
    return data[['Close']]

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('predict'))
    return redirect(url_for('signin'))

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        users = load_users()
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('predict'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('signin'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        users = load_users()
        if username in users:
            flash('Username already exists')
            return redirect(url_for('signup'))
        users[username] = password
        save_users(users)
        flash('Signup successful! Please sign in.')
        return redirect(url_for('signin'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('signin'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('signin'))

    prediction = None
    accuracy = None
    graph_url = None
    error = None

    if request.method == 'POST':
        # Check if CSV file uploaded
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                try:
                    df = pd.read_csv(filepath)
                    prediction, accuracy, graph_url = train_and_predict(df)
                except Exception as e:
                    error = f"Error processing file: {str(e)}"
            else:
                error = "Invalid file type. Please upload a CSV file."
        else:
            # Manual input from form fields
            try:
                # Expecting comma separated closing prices for last N days
                prices_str = request.form.get('prices')
                if not prices_str:
                    error = "Please provide historical prices or upload a CSV file."
                else:
                    prices = [float(p.strip()) for p in prices_str.split(',')]
                    df = pd.DataFrame({'Close': prices})
                    prediction, accuracy, graph_url = train_and_predict(df)
            except Exception as e:
                error = f"Error processing manual input: {str(e)}"

    return render_template('predict.html', prediction=prediction, accuracy=accuracy, graph_url=graph_url, error=error)

@app.route('/predict-live')
def predict_live():
    if 'username' not in session:
        return redirect(url_for('signin'))
    try:
        df = fetch_live_nifty_data()
        # Ensure df is a DataFrame with single 'Close' column and no multi-index
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Close']].copy()
        prediction, accuracy, graph_url = train_and_predict(df)
        return render_template('predict.html', prediction=prediction, accuracy=accuracy, graph_url=graph_url, error=None)
    except Exception as e:
        error = f"Error fetching live data: {str(e)}"
        return render_template('predict.html', prediction=None, accuracy=None, graph_url=None, error=error)

if __name__ == '__main__':
    app.run(debug=True)

