import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Define Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset" / "salary_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load Dataset
# --------------------------------------------------

df = pd.read_csv(DATASET_PATH)


# --------------------------------------------------
# 3. Separate Features and Target
# --------------------------------------------------

X = df[["Years_of_Experience"]]
y = df["Salary"]


# --------------------------------------------------
# 4. Split Dataset into Training and Testing Sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 5. Create and Train Linear Regression Model
# --------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)


# --------------------------------------------------
# 6. Display Model Equation
# --------------------------------------------------

coefficient = model.coef_[0]
intercept = model.intercept_

equation = (
    f"Salary = {coefficient:.2f} × "
    f"Years_of_Experience + {intercept:.2f}"
)

print("Linear Regression Model")
print("----------------------------")
print(f"Coefficient : {coefficient:.2f}")
print(f"Intercept   : {intercept:.2f}")
print(f"Equation    : {equation}")


# --------------------------------------------------
# 7. Make Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 8. Evaluate Model
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print("\nModel Evaluation")
print("----------------------------")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# --------------------------------------------------
# 9. Display Actual vs Predicted Values
# --------------------------------------------------

comparison = pd.DataFrame({
    "Actual Salary": y_test.values,
    "Predicted Salary": y_pred
})

print("\nActual vs Predicted Salary")
print("----------------------------")
print(comparison)


# --------------------------------------------------
# 10. Save Evaluation Results
# --------------------------------------------------

results_path = OUTPUT_DIR / "evaluation_results.txt"

with open(results_path, "w", encoding="utf-8") as file:

    file.write("Linear Regression Model Evaluation\n")
    file.write("==================================\n\n")

    file.write("Dataset:\n")
    file.write("Salary Prediction Dataset\n\n")

    file.write("Feature:\n")
    file.write("Years_of_Experience\n\n")

    file.write("Target:\n")
    file.write("Salary\n\n")

    file.write("Model:\n")
    file.write("Linear Regression\n\n")

    file.write("Train/Test Split:\n")
    file.write("80% Training\n")
    file.write("20% Testing\n\n")

    file.write("Model Equation:\n")
    file.write(f"{equation}\n\n")

    file.write("Evaluation Metrics:\n")
    file.write("-------------------\n\n")

    file.write(f"MAE  : {mae:.2f}\n")
    file.write(f"MSE  : {mse:.2f}\n")
    file.write(f"RMSE : {rmse:.2f}\n")
    file.write(f"R²   : {r2:.4f}\n\n")

    file.write("Interpretation:\n")
    file.write("----------------\n")

    file.write(
        f"The Linear Regression model achieved an R² score of "
        f"{r2:.4f}, indicating that the model explains approximately "
        f"{r2 * 100:.2f}% of the variance in salary on the test dataset.\n\n"
    )

    file.write(
        f"The MAE of {mae:.2f} indicates that the model's predictions "
        f"differ from the actual salary values by approximately "
        f"₹{mae:.2f} on average.\n"
    )

print(f"\nEvaluation results saved to: {results_path}")


# --------------------------------------------------
# 11. Create Regression Visualization
# --------------------------------------------------

X_test_sorted = X_test.sort_values(by="Years_of_Experience")
y_pred_sorted = model.predict(X_test_sorted)

plt.figure(figsize=(8, 5))

plt.scatter(
    X_test,
    y_test,
    label="Actual Salary"
)

plt.plot(
    X_test_sorted,
    y_pred_sorted,
    label="Regression Line"
)

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression - Salary Prediction")

plt.legend()

plot_path = OUTPUT_DIR / "regression_plot.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Regression plot saved to: {plot_path}")