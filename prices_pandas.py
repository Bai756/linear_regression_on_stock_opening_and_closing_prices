import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
points = pd.read_csv("/Users/qiaoe27/programs/linear regression/linear_regression_on_stock_opening_and_closing_prices/prices-split-adjusted.csv")

points = points[['symbol', 'open', 'close']]

# Filter the DataFrame for a specific company, e.g., 'AAPL'
company_name = 'AMZN'
company_data = points[points['symbol'] == company_name]

print(company_data)
print(points.iloc[0, 0])
print(points.iloc[0, 1])

# plot points
plt.scatter(company_data['open'], company_data['close'])

x_values = company_data['open']
y_values = .9 * company_data['open'] + 5
plt.plot(x_values, y_values, color='red')

plt.xlabel('Open Price')
plt.ylabel('Close Price')

plt.show()

