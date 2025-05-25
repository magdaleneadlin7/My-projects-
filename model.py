import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import io
import base64

def train_and_predict(df):
    """
    Train a Random Forest model on the historical 'Close' prices and predict the next day price.
    Returns:
        prediction (float): predicted next day price
        accuracy (float): model accuracy as 1 - MAE/mean price
        graph_url (str): base64 encoded PNG graph of recent trends and prediction
    """
    # Check if 'Close' column exists
    if 'Close' not in df.columns:
        raise ValueError("CSV or input data must contain 'Close' column or manual input must be closing prices.")

    # Prepare data for supervised learning: use previous day's price to predict next day's price
    df = df[['Close']].dropna().reset_index(drop=True)
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()

    X = df[['Close']]
    y = df['Target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate accuracy metric
    mae = mean_absolute_error(y_test, y_pred)
    mean_price = y_test.mean()
    accuracy = max(0, 1 - mae / mean_price)  # accuracy between 0 and 1

    # Predict next day price using last available close price
    last_close = df['Close'].iloc[-1]
    next_day_pred = model.predict([[last_close]])[0]

    # Plot recent trends and prediction
    plt.figure(figsize=(10,5))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.plot(y_test.index, y_pred, label='Predicted Price (Test)')
    plt.scatter(df.index[-1] + 1, next_day_pred, color='red', label='Next Day Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('NIFTY 50 Close Price and Prediction')
    plt.legend()
    plt.tight_layout()

    # Save plot to PNG image in memory
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    return round(next_day_pred, 2), round(accuracy * 100, 2), graph_url
