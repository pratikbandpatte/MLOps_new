import pandas as pd


data = {
    "customer_id": list(range(1, 31)),
    "age": [
        25, 34, 45, 29, 52, 41, 23, 38, 31, 60,
        27, 36, 49, 33, 55, 42, 24, 39, 30, 58,
        26, 35, 47, 32, 53, 40, 22, 37, 28, 57
    ],
    "monthly_charges": [
        29.9, 59.5, 89.2, 45.0, 99.5, 75.3, 25.5, 65.2, 50.0, 110.0,
        35.0, 70.5, 95.0, 48.0, 105.0, 80.0, 28.0, 67.0, 52.0, 108.0,
        32.0, 72.0, 92.0, 46.0, 102.0, 78.0, 27.0, 64.0, 49.0, 107.0
    ],
    "tenure_months": [
        3, 24, 48, 8, 60, 36, 2, 30, 12, 72,
        5, 20, 55, 10, 65, 40, 1, 28, 14, 70,
        4, 22, 50, 9, 62, 38, 2, 26, 11, 68
    ],
    "support_calls": [
        5, 2, 1, 4, 0, 1, 6, 2, 3, 0,
        5, 2, 1, 4, 0, 1, 6, 2, 3, 0,
        5, 2, 1, 4, 0, 1, 6, 2, 3, 0
    ],
    "contract_type": [
        "Month-to-month", "One year", "Two year", "Month-to-month", "Two year",
        "One year", "Month-to-month", "One year", "Month-to-month", "Two year",
        "Month-to-month", "One year", "Two year", "Month-to-month", "Two year",
        "One year", "Month-to-month", "One year", "Month-to-month", "Two year",
        "Month-to-month", "One year", "Two year", "Month-to-month", "Two year",
        "One year", "Month-to-month", "One year", "Month-to-month", "Two year"
    ],
    "internet_service": [
        "Fiber", "DSL", "Fiber", "DSL", "Fiber",
        "DSL", "Fiber", "DSL", "Fiber", "Fiber",
        "DSL", "Fiber", "Fiber", "DSL", "Fiber",
        "DSL", "Fiber", "DSL", "Fiber", "Fiber",
        "DSL", "Fiber", "Fiber", "DSL", "Fiber",
        "DSL", "Fiber", "DSL", "Fiber", "Fiber"
    ],
    "payment_method": [
        "Electronic check", "Credit card", "Bank transfer", "Electronic check", "Credit card",
        "Bank transfer", "Electronic check", "Credit card", "Electronic check", "Bank transfer",
        "Electronic check", "Credit card", "Bank transfer", "Electronic check", "Credit card",
        "Bank transfer", "Electronic check", "Credit card", "Electronic check", "Bank transfer",
        "Electronic check", "Credit card", "Bank transfer", "Electronic check", "Credit card",
        "Bank transfer", "Electronic check", "Credit card", "Electronic check", "Bank transfer"
    ],
    "churn": [
        1, 0, 0, 1, 0,
        0, 1, 0, 1, 0,
        1, 0, 0, 1, 0,
        0, 1, 0, 1, 0,
        1, 0, 0, 1, 0,
        0, 1, 0, 1, 0
    ]
}


df = pd.DataFrame(data)

df.to_csv("data/customer_churn.csv", index=False)

print("Customer churn dataset created successfully.")
print("Shape:", df.shape)
print(df.head())