from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque


class Agent:
    def __init__(self, window_size, is_eval=False, model_name=""):
        self.__inventory = []
        self.__total_profit = 0
        self.action_history = []

        self.state_size = window_size * 2  # Adjusted to match the input dimension
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=2000)  # Increased memory size
        self.model_name = model_name
        self.is_eval = is_eval

        # Hyperparameters tuned for more conservative trading
        self.gamma = 0.99  # Increased discount factor
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99  # Slower decay
        self.learning_rate = 0.005  # Slightly increased learning rate

        # Reward scaling and trading parameters
        self.profit_threshold = 0.005  # 0.5% profit threshold
        self.max_inventory_size = 3  # Limit number of simultaneous stock holdings

        # Explicitly create a new model or load an existing one
        self.model = self.create_model()
        if is_eval and model_name:
            try:
                self.model = load_model(f"models/{model_name}")
                print(f"Loaded existing model: {model_name}")
            except Exception as e:
                print(f"Could not load model, using a new one. Error: {e}")

    def create_model(self):
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation="relu"),
            Dense(64, activation="relu"),  # Added more neurons
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="huber_loss", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def reset(self):
        self.__inventory = []
        self.__total_profit = 0
        self.action_history = []

    def act(self, state, price_data):
        # Ensure state is the correct shape
        state = state.reshape(1, -1)

        # Always predict in evaluation mode
        options = self.model.predict(state, verbose=0)[0]

        # Add more aggressive action selection logic
        action = np.argmax(options)

        # Debug print
        print(f"Action Predictions: {options}")
        print(f"Chosen Action Index: {action}")

        bought_price = None
        if action == 1:  # buy
            # Always try to buy if under max inventory
            if len(self.__inventory) < self.max_inventory_size:
                self.buy(price_data)
                bought_price = price_data
                print(f"Buying at price: {price_data}")

        elif action == 2 and self.has_inventory():  # sell
            # Try to sell if we can make a profit
            bought_price = self.sell(price_data)
            if bought_price is not None:
                print(f"Selling, bought at: {bought_price}, sold at: {price_data}")

        self.action_history.append(action)
        return action, bought_price

    def experience_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Randomly sample from memory
        mini_batch = random.sample(list(self.memory), batch_size)

        states = np.array([exp[0].reshape(-1) for exp in mini_batch])
        next_states = np.array([exp[3].reshape(-1) for exp in mini_batch])

        # Predict Q-values for current and next states
        current_qs_list = self.model.predict(states, verbose=0)
        future_qs_list = self.model.predict(next_states, verbose=0)

        X = []
        y = []

        for index, (state, action, reward, next_state, done) in enumerate(mini_batch):
            if done:
                new_q = reward
            else:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma * max_future_q

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(state.reshape(-1))
            y.append(current_qs)

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Fit the model
        self.model.fit(X, y, batch_size=batch_size, verbose=0, epochs=1)

        # Decay exploration rate with a more gradual approach
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def buy(self, price_data):
        self.__inventory.append(price_data)

    def sell(self, price_data):
        # Find the oldest purchase with potential profit
        for i, bought_price in enumerate(self.__inventory):
            # Only sell if we can make a profit above threshold
            if price_data > bought_price * (1 + self.profit_threshold):
                del self.__inventory[i]
                profit = price_data - bought_price
                self.__total_profit += profit
                return bought_price

        return None

    def get_total_profit(self):
        return self.format_price(self.__total_profit)

    def has_inventory(self):
        return len(self.__inventory) > 0

    def format_price(self, n):
        return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))