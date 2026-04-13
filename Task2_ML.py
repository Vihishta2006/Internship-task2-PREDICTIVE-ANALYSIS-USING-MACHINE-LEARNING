import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Load dataset
data = pd.read_csv("datasets/house_price.csv")

# Show first 5 rows
print(data.head())

#Select Features
X = data[['area','bedrooms','bathrooms']]
y = data['price']

#Step 5: Split Data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#print("X_train:\n", X_train)
#print("X_test:\n", X_test)
#print("y_train:\n", y_train)
#print("y_test:\n", y_test)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

#Train model
model = LinearRegression()
model.fit(X_train, y_train)

#Predict
predictions = model.predict(X_test)

#Evaluate
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)

#Graph
plt.scatter(y_test, predictions)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

#Line plot 
plt.plot(y_test.values, label="Actual", marker='o')
plt.plot(predictions, label="Predicted", marker='x')
plt.legend()
plt.title("Actual vs Predicted Comparison")
plt.xlabel("Data Points")
plt.ylabel("Price")
plt.show()

#Residual Plot
residuals = y_test - predictions

plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

#Histogram of Errors
plt.hist(residuals, bins=5)
plt.title("Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

features = X.columns
coefficients = model.coef_

plt.bar(features, coefficients)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Impact on Price")
plt.show()

#Correlation Heatmap
import seaborn as sns

sns.heatmap(data.corr(), annot=True)
plt.title("Correlation Between Features")
plt.show()