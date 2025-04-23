from keras.models import load_model

from agent import Agent
from market_environment import Market


import matplotlib.pyplot as plt


def main():
    try:
        stock_name = "GSPC_2011-03"
        model_name = "model_ep10"

        # Load the model
        model = load_model(f"models/{model_name}")

        # Dynamically determine window size
        window_size = model.layers[0].input_shape[1] // 2
        print(f"Detected window size: {window_size}")

        agent = Agent(window_size, is_eval=True, model_name=model_name)
        market = Market(window_size, stock_name)

        state, price_data = market.reset()

        # Debugging variables
        detailed_actions = []
        prices_at_actions = []

        for t in range(market.last_data_index):
            # Reshape state correctly
            state = state.reshape(1, -1)

            # Debug: Print model predictions
            predictions = agent.model.predict(state, verbose=0)[0]
            print(f"Step {t}: Action Predictions: {predictions}")

            action, bought_price = agent.act(state, price_data)

            # Log detailed action information
            detailed_actions.append(action)
            prices_at_actions.append(price_data)

            # Print action details
            action_names = {0: "Hold", 1: "Buy", 2: "Sell"}
            print(f"Step {t}: Action: {action_names[action]}, Price: {price_data}, Bought Price: {bought_price}")

            # Get next state
            next_state, next_price_data, reward, done = market.get_next_state_reward(action, bought_price)

            state = next_state
            price_data = next_price_data

            if done:
                break

        # Detailed profit analysis
        print("--------------------------------")
        total_profit = agent.get_total_profit()
        print(f"{stock_name} Total Profit: {total_profit}")

        # Analyze actions
        print("\nAction Analysis:")
        print("Action Distribution:")
        from collections import Counter
        action_counts = Counter(detailed_actions)
        for action, count in action_counts.items():
            print(f"{['Hold', 'Buy', 'Sell'][action]}: {count} times")

        print("\nPrice Progression:")
        for i in range(min(10, len(prices_at_actions))):
            print(f"Step {i}: {prices_at_actions[i]}")

        # Detailed inventory check
        print("\nFinal Inventory:")
        print(f"Inventory Size: {len(agent._Agent__inventory)}")

        # Plot actions
        plot_action_profit(market.data, detailed_actions, total_profit)

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

def plot_action_profit(data, action_data, profit):
    plt.figure(figsize=(12, 6))  # Add figure size for better readability
    plt.plot(range(len(data)), data['Close'])  # Assuming data is a DataFrame, plot Close prices
    plt.xlabel("Date Index")
    plt.ylabel("Price")

    # Corrected buy and sell plotting
    buy_points = []
    sell_points = []

    for d in range(len(action_data) - 1):  # Ensure index is within bounds
        if action_data[d] == 1:  # buy
            buy_points.append((d, data.iloc[d]['Close']))
        elif action_data[d] == 2:  # sell
            sell_points.append((d, data.iloc[d]['Close']))

    # Plot buy and sell points
    if buy_points:
        buy_x, buy_y = zip(*buy_points)
        plt.scatter(buy_x, buy_y, color='green', marker='*', label='Buy')

    if sell_points:
        sell_x, sell_y = zip(*sell_points)
        plt.scatter(sell_x, sell_y, color='red', marker='+', label='Sell')

    plt.title(f"Total Profit: {profit}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("buy_sell.png")
    plt.show()

if __name__=="__main__":
    main()