import pandas as pd
from func import my_train_test_split, my_scaler_fit, my_scaler_transform, MyLinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./data/day.csv")
df = df.drop(columns=["instant", "dteday", "casual", "registered", "mnth", "temp"])

x = df.drop("cnt", axis= 1)
y = df["cnt"]

x_train, x_test, y_train, y_test = my_train_test_split(
    x,
    y,
    random_state= 42,
    test_size= 0.2
)

mean, std = my_scaler_fit(x_train)

x_train = my_scaler_transform(x_train, mean, std)
x_test = my_scaler_transform(x_test, mean, std)

model = MyLinearRegression(epochs= 10000, lr= 1e-4)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)


plt.figure(figsize=(10, 5))
plt.plot(model.loss_history)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training Loss")
plt.grid()

plt.show()