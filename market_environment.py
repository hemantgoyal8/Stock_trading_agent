import numpy as np
import pandas as pd
from sklearn import preprocessing
import keras
print(keras.__version__)



class Market:
    def __init__(self, window_size, stock_name):
        # Load and preprocess data
        self.data = self.__get_stock_data(stock_name)
        self.window_size = window_size

        # Preprocess states
        self.states = self.__get_all_window_prices_diff(self.data, window_size)

        # Tracking variables
        self.index = -1
        self.last_data_index = len(self.data) - window_size - 1  # Adjusted to prevent index out of bounds

        # Additional tracking for more sophisticated reward
        self.initial_portfolio_value = self.data.iloc[0]["Close"]
        self.current_portfolio_value = self.initial_portfolio_value

    def __get_stock_data(self, key):
        file_path = "data/" + key + ".csv"
        lines = pd.read_csv(file_path, sep=',')

        # Additional data validation
        print("Data Overview:")
        print(lines.head())
        print("\nData Statistics:")
        print(lines.describe())
        print(f"\nTotal Data Points: {len(lines)}")
        print(f"Date Range: {lines['Date'].min()} to {lines['Date'].max()}")

        return lines

    def __get_window(self, data_df, t, n):
        d = t - n + 1
        data1 = data_df["Close"].values
        data2 = data_df["Volume"].values
        block1 = data1[d:t + 1] if d >= 0 else np.append(-d * [data1[0]], data1[0:t + 1])  # pad with t0
        block2 = data2[d:t + 1] if d >= 0 else np.append(-d * [data2[0]], data2[0:t + 1])
        res = []
        for i in range(n - 1):
            res.append(block1[i + 1] - block1[i])
        for i in range(n - 1):
            res.append(block2[i + 1] - block2[i])
        return np.array([res])

    def normalize_data(self, in_df):
        x = in_df.values  # get the numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled, columns=in_df.columns)
        return df

    def __get_all_window_prices_diff(self, data, n):
        l = len(data)
        processed_data = []

        sel_col = ["Close", "Volume"]
        scaled_data = self.normalize_data(data[sel_col])

        for t in range(l):
            state = self.__get_window(scaled_data, t, n + 1)
            processed_data.append(state)
        return processed_data

    def reset(self):
        # Reset index and portfolio value
        self.index = -1
        self.current_portfolio_value = self.initial_portfolio_value
        return self.states[0], self.data.iloc[0]["Close"]

    def get_next_state_reward(self, action, bought_price=None):
        self.index += 1

        if self.index >= self.last_data_index:
            return self.states[-1], self.data.iloc[-1]["Close"], 0, True

        next_state = self.states[self.index + 1]
        next_price_data = self.data.iloc[self.index + 1]["Close"]
        price_data = self.data.iloc[self.index]["Close"]

        # More explicit reward calculation
        reward = 0

        if action == 0:  # Hold
            # Small penalty for doing nothing
            reward = -0.01

        elif action == 1:  # Buy
            # Reward for buying at a potentially good price
            if next_price_data > price_data:
                reward = 1
            else:
                reward = -1

        elif action == 2:  # Sell
            if bought_price is not None:
                # Reward based on profit percentage
                profit_percentage = (next_price_data - bought_price) / bought_price
                reward = max(0, profit_percentage * 10)  # Scale up the reward
            else:
                reward = -1

        # Check if episode is done
        done = self.index == self.last_data_index - 1

        return next_state, next_price_data, reward, done