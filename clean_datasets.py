from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer


BASE_DIR = Path(__file__).resolve().parent
EMPLOYEE_INPUT = BASE_DIR / "employee_information_dirty.csv"
SALES_INPUT = BASE_DIR / "sales_data_dirty.csv"
EMPLOYEE_OUTPUT = BASE_DIR / "employee_information_cleaned.csv"
SALES_OUTPUT = BASE_DIR / "sales_data_cleaned.csv"
CLEANING_REPORT = BASE_DIR / "data_cleaning_report.txt"


def parse_mixed_date(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()
    if not text:
        return pd.NaT

    for dayfirst in (False, True):
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)
        if pd.notna(parsed):
            return parsed.normalize()
    return pd.NaT


def standardize_employee_gender(series: pd.Series) -> pd.Series:
    mapping = {
        "m": "Male",
        "male": "Male",
        "male ": "Male",
        "f": "Female",
        "female": "Female",
        "fem": "Female",
        "fem.": "Female",
    }

    def normalize(value: object) -> object:
        if pd.isna(value):
            return np.nan
        cleaned = str(value).strip().lower()
        return mapping.get(cleaned, np.nan)

    return series.map(normalize)


def standardize_title_text(series: pd.Series) -> pd.Series:
    def normalize(value: object) -> object:
        if pd.isna(value):
            return np.nan
        cleaned = " ".join(str(value).strip().split())
        if not cleaned:
            return np.nan
        return cleaned.title()

    return series.map(normalize)


def standardize_sales_categories(series: pd.Series) -> pd.Series:
    valid_categories = {
        "electronics": "Electronics",
        "furniture": "Furniture",
        "clothing": "Clothing",
        "groceries": "Groceries",
        "sports": "Sports",
        "beauty": "Beauty",
    }

    def normalize(value: object) -> object:
        if pd.isna(value):
            return np.nan
        cleaned = str(value).strip().lower()
        return valid_categories.get(cleaned, np.nan)

    return series.map(normalize)


def standardize_payment_method(series: pd.Series) -> pd.Series:
    mapping = {
        "credit card": "Credit Card",
        "cash": "Cash",
        "debit card": "Debit Card",
        "mobile pay": "Mobile Pay",
    }

    def normalize(value: object) -> object:
        if pd.isna(value):
            return np.nan
        cleaned = " ".join(str(value).strip().lower().split())
        return mapping.get(cleaned, np.nan)

    return series.map(normalize)


def impute_dates(series: pd.Series, strategy: str = "median") -> pd.Series:
    parsed = series.map(parse_mixed_date)

    if strategy == "ffill":
        parsed = parsed.ffill().bfill()
        return parsed

    numeric_dates = parsed.map(lambda x: x.value if pd.notna(x) else np.nan).to_numpy(dtype="float64").reshape(-1, 1)
    imputed_numeric = SimpleImputer(strategy="median").fit_transform(numeric_dates).ravel()
    return pd.to_datetime(imputed_numeric)


def clean_employee_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    cleaned = df.copy()

    cleaned["Name"] = standardize_title_text(cleaned["Name"])
    cleaned["Department"] = standardize_title_text(cleaned["Department"])
    cleaned["Gender"] = standardize_employee_gender(cleaned["Gender"])

    cleaned["Date of Birth"] = impute_dates(cleaned["Date of Birth"], strategy="median")
    cleaned["Date of Joining"] = impute_dates(cleaned["Date of Joining"], strategy="ffill")

    salary = pd.to_numeric(cleaned["Salary"], errors="coerce")
    performance = pd.to_numeric(cleaned["Performance Score"], errors="coerce")
    salary[(salary <= 0) | (salary > 500000)] = np.nan
    performance[(performance < 1) | (performance > 5)] = np.nan
    cleaned["Salary"] = salary
    cleaned["Performance Score"] = performance

    categorical_columns = ["Name", "Department", "Gender"]
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    cleaned[categorical_columns] = categorical_imputer.fit_transform(cleaned[categorical_columns])

    numeric_columns = ["Salary", "Performance Score"]
    knn_imputer = KNNImputer(n_neighbors=5)
    cleaned[numeric_columns] = knn_imputer.fit_transform(cleaned[numeric_columns])

    cleaned["Date of Birth"] = pd.to_datetime(cleaned["Date of Birth"]).dt.strftime("%Y-%m-%d")
    cleaned["Date of Joining"] = pd.to_datetime(cleaned["Date of Joining"]).dt.strftime("%Y-%m-%d")

    summary = {
        "employee_missing_before": int(df.isna().sum().sum()),
        "employee_missing_after": int(cleaned.isna().sum().sum()),
        "employee_invalid_salary_before": int(((pd.to_numeric(df["Salary"], errors="coerce") <= 0) | (pd.to_numeric(df["Salary"], errors="coerce") > 500000)).sum()),
        "employee_invalid_salary_after": int(((pd.to_numeric(cleaned["Salary"], errors="coerce") <= 0) | (pd.to_numeric(cleaned["Salary"], errors="coerce") > 500000)).sum()),
    }
    return cleaned, summary


def clean_sales_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    cleaned = df.copy()

    cleaned["Product Category"] = standardize_sales_categories(cleaned["Product Category"])
    cleaned["Payment Method"] = standardize_payment_method(cleaned["Payment Method"])
    cleaned["Sale Date"] = impute_dates(cleaned["Sale Date"], strategy="ffill")

    quantity = pd.to_numeric(cleaned["Quantity Sold"], errors="coerce")
    sale_amount = pd.to_numeric(cleaned["Sale Amount"], errors="coerce")
    quantity[quantity <= 0] = np.nan
    sale_amount[sale_amount <= 0] = np.nan
    cleaned["Quantity Sold"] = quantity
    cleaned["Sale Amount"] = sale_amount

    categorical_columns = ["Product Category", "Payment Method"]
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    cleaned[categorical_columns] = categorical_imputer.fit_transform(cleaned[categorical_columns])

    numeric_columns = ["Quantity Sold", "Sale Amount"]
    knn_imputer = KNNImputer(n_neighbors=5)
    cleaned[numeric_columns] = knn_imputer.fit_transform(cleaned[numeric_columns])

    cleaned["Sale Date"] = pd.to_datetime(cleaned["Sale Date"]).dt.strftime("%Y-%m-%d")
    cleaned["Quantity Sold"] = cleaned["Quantity Sold"].round().astype(int)
    cleaned["Sale Amount"] = cleaned["Sale Amount"].round(2)

    summary = {
        "sales_missing_before": int(df.isna().sum().sum()),
        "sales_missing_after": int(cleaned.isna().sum().sum()),
        "sales_negative_qty_before": int((pd.to_numeric(df["Quantity Sold"], errors="coerce") < 0).sum()),
        "sales_negative_qty_after": int((pd.to_numeric(cleaned["Quantity Sold"], errors="coerce") < 0).sum()),
    }
    return cleaned, summary


def write_report(employee_summary: dict[str, int], sales_summary: dict[str, int]) -> None:
    lines = [
        "Data Cleaning Report",
        "====================",
        "",
        "Employee Information",
        f"Missing cells before cleaning: {employee_summary['employee_missing_before']}",
        f"Missing cells after cleaning: {employee_summary['employee_missing_after']}",
        f"Invalid salary rows before cleaning: {employee_summary['employee_invalid_salary_before']}",
        f"Invalid salary rows after cleaning: {employee_summary['employee_invalid_salary_after']}",
        "",
        "Methods applied:",
        "- KNN Imputer for Salary and Performance Score",
        "- Most Frequent Imputer for Name, Department, and Gender",
        "- Median date imputation for Date of Birth after parsing mixed formats",
        "- Forward-fill and backward-fill for Date of Joining after parsing mixed formats",
        "- Text standardization with trimming and title casing",
        "",
        "Sales Data",
        f"Missing cells before cleaning: {sales_summary['sales_missing_before']}",
        f"Missing cells after cleaning: {sales_summary['sales_missing_after']}",
        f"Negative quantity rows before cleaning: {sales_summary['sales_negative_qty_before']}",
        f"Negative quantity rows after cleaning: {sales_summary['sales_negative_qty_after']}",
        "",
        "Methods applied:",
        "- KNN Imputer for Quantity Sold and Sale Amount",
        "- Most Frequent Imputer for Product Category and Payment Method",
        "- Forward-fill and backward-fill for Sale Date after parsing mixed formats",
        "- Category and payment method standardization",
    ]
    CLEANING_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    employee_df = pd.read_csv(EMPLOYEE_INPUT)
    sales_df = pd.read_csv(SALES_INPUT)

    employee_cleaned, employee_summary = clean_employee_data(employee_df)
    sales_cleaned, sales_summary = clean_sales_data(sales_df)

    employee_cleaned.to_csv(EMPLOYEE_OUTPUT, index=False)
    sales_cleaned.to_csv(SALES_OUTPUT, index=False)
    write_report(employee_summary, sales_summary)

    print(f"Created cleaned dataset: {EMPLOYEE_OUTPUT.name}")
    print(f"Created cleaned dataset: {SALES_OUTPUT.name}")
    print(f"Created report: {CLEANING_REPORT.name}")


if __name__ == "__main__":
    main()
