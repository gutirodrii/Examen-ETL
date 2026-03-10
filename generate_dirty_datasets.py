from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 42


def generate_employee_dataset(rows: int = 100) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    first_names = [
        "Alice",
        "Bob",
        "Carlos",
        "Diana",
        "Elena",
        "Farid",
        "Grace",
        "Hector",
        "Isabel",
        "Jamal",
    ]
    last_names = [
        "Smith",
        "Johnson",
        "Garcia",
        "Brown",
        "Lopez",
        "Wilson",
        "Taylor",
        "Martinez",
        "Anderson",
        "Thomas",
    ]
    departments = ["Sales", "HR", "IT", "Finance", "Operations", "Marketing", None]
    genders = ["Male", "Female", "M", "F", "male", "female", "Unknown", "Fem", None]

    dob_base = pd.date_range("1970-01-01", "2000-12-31", periods=rows)
    join_base = pd.date_range("2010-01-01", "2024-12-31", periods=rows)

    data = {
        "Employee ID": [f"EMP{1000 + i}" for i in range(rows)],
        "Name": [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(rows)],
        "Department": [random.choice(departments[:-1]) for _ in range(rows)],
        "Date of Birth": [date.strftime("%Y-%m-%d") for date in dob_base],
        "Gender": [random.choice(genders[:-1]) for _ in range(rows)],
        "Salary": np.random.randint(28000, 120000, size=rows).astype(float),
        "Date of Joining": [date.strftime("%Y-%m-%d") for date in join_base],
        "Performance Score": np.random.randint(1, 6, size=rows).astype(float),
    }
    df = pd.DataFrame(data)

    # Introduce missing values in more than 20% of cells across several columns.
    missing_specs = {
        "Department": 30,
        "Name": 25,
        "Gender": 30,
        "Performance Score": 25,
        "Date of Birth": 25,
        "Salary": 15,
        "Date of Joining": 15,
    }
    for column, count in missing_specs.items():
        idx = np.random.choice(df.index, size=count, replace=False)
        df.loc[idx, column] = np.nan

    # Inconsistent date formats.
    dob_idx = np.random.choice(df.index, size=25, replace=False)
    join_idx = np.random.choice(df.index, size=25, replace=False)
    dob_formats = ["%d/%m/%Y", "%m-%d-%Y", "%Y/%m/%d", "%d-%b-%Y"]
    join_formats = ["%m/%d/%Y", "%d-%m-%Y", "%Y.%m.%d", "%b %d, %Y"]
    for idx in dob_idx:
        if pd.notna(df.loc[idx, "Date of Birth"]):
            parsed = pd.to_datetime(df.loc[idx, "Date of Birth"], errors="coerce")
            df.loc[idx, "Date of Birth"] = parsed.strftime(random.choice(dob_formats))
    for idx in join_idx:
        parsed = pd.to_datetime(df.loc[idx, "Date of Joining"], errors="coerce")
        if pd.notna(parsed):
            df.loc[idx, "Date of Joining"] = parsed.strftime(random.choice(join_formats))

    # Erroneous salary values.
    bad_salary_idx = np.random.choice(df.index, size=12, replace=False)
    bad_salaries = [-5000, -100, 0, 9999999, 2500000, -42000]
    for idx in bad_salary_idx:
        df.loc[idx, "Salary"] = random.choice(bad_salaries)

    # Extra inconsistent gender values.
    gender_idx = np.random.choice(df.index, size=18, replace=False)
    inconsistent_genders = ["fem", "MALE ", " woman", "NonBinary", "X", "Fem."]
    for idx in gender_idx:
        if pd.notna(df.loc[idx, "Gender"]):
            df.loc[idx, "Gender"] = random.choice(inconsistent_genders)

    return df


def generate_sales_dataset(rows: int = 120) -> pd.DataFrame:
    random.seed(RANDOM_SEED + 1)
    np.random.seed(RANDOM_SEED + 1)

    categories = ["Electronics", "Furniture", "Clothing", "Groceries", "Sports", "Beauty"]
    payment_methods = ["Credit Card", "credit card", "Cash", "cash ", "Debit Card", "Mobile Pay"]
    sale_base = pd.date_range("2023-01-01", "2025-02-28", periods=rows)

    data = {
        "Transaction ID": [f"TXN{5000 + i}" for i in range(rows)],
        "Product Category": [random.choice(categories) for _ in range(rows)],
        "Quantity Sold": np.random.randint(1, 15, size=rows).astype(float),
        "Sale Amount": np.round(np.random.uniform(15, 2500, size=rows), 2),
        "Payment Method": [random.choice(payment_methods) for _ in range(rows)],
        "Sale Date": [date.strftime("%Y-%m-%d") for date in sale_base],
    }
    df = pd.DataFrame(data)

    # At least 20% missing values in product category and additional gaps elsewhere.
    category_missing_idx = np.random.choice(df.index, size=30, replace=False)
    df.loc[category_missing_idx, "Product Category"] = np.nan

    amount_missing_idx = np.random.choice(df.index, size=15, replace=False)
    payment_missing_idx = np.random.choice(df.index, size=12, replace=False)
    df.loc[amount_missing_idx, "Sale Amount"] = np.nan
    df.loc[payment_missing_idx, "Payment Method"] = np.nan

    # Negative quantity values.
    negative_qty_idx = np.random.choice(df.index, size=14, replace=False)
    negative_values = [-1, -2, -5, -10]
    for idx in negative_qty_idx:
        df.loc[idx, "Quantity Sold"] = random.choice(negative_values)

    # Inconsistent date formats.
    sale_date_idx = np.random.choice(df.index, size=35, replace=False)
    sale_formats = ["%d/%m/%Y", "%m-%d-%Y", "%Y/%m/%d", "%d %b %Y", "%Y.%m.%d"]
    for idx in sale_date_idx:
        parsed = pd.to_datetime(df.loc[idx, "Sale Date"], errors="coerce")
        df.loc[idx, "Sale Date"] = parsed.strftime(random.choice(sale_formats))

    return df


def main() -> None:
    employee_df = generate_employee_dataset()
    sales_df = generate_sales_dataset()

    employee_path = OUTPUT_DIR / "employee_information_dirty.csv"
    sales_path = OUTPUT_DIR / "sales_data_dirty.csv"

    employee_df.to_csv(employee_path, index=False)
    sales_df.to_csv(sales_path, index=False)

    print(f"Created {employee_path.name} with shape {employee_df.shape}")
    print(f"Created {sales_path.name} with shape {sales_df.shape}")
    print(f"Employee missing cells: {int(employee_df.isna().sum().sum())}")
    print(f"Sales missing cells: {int(sales_df.isna().sum().sum())}")


if __name__ == "__main__":
    main()
