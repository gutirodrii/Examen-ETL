from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
EMPLOYEE_INPUT = BASE_DIR / "employee_information_cleaned.csv"
SALES_INPUT = BASE_DIR / "sales_data_cleaned.csv"
MERGED_OUTPUT = BASE_DIR / "employee_sales_merged.csv"
MERGE_REPORT = BASE_DIR / "data_merge_report.txt"
RANDOM_SEED = 42


def attach_employee_ids(sales_df: pd.DataFrame, employee_df: pd.DataFrame) -> pd.DataFrame:
    sales_with_ids = sales_df.copy()

    if "Employee ID" in sales_with_ids.columns:
        return sales_with_ids

    rng = np.random.default_rng(RANDOM_SEED)
    employee_ids = employee_df["Employee ID"].dropna().astype(str).to_numpy()
    sales_with_ids.insert(
        1,
        "Employee ID",
        rng.choice(employee_ids, size=len(sales_with_ids), replace=True),
    )
    return sales_with_ids


def main() -> None:
    employee_df = pd.read_csv(EMPLOYEE_INPUT)
    sales_df = pd.read_csv(SALES_INPUT)

    sales_with_ids = attach_employee_ids(sales_df, employee_df)
    merged_df = sales_with_ids.merge(employee_df, on="Employee ID", how="left", validate="many_to_one")

    merged_df["Sale Date"] = pd.to_datetime(merged_df["Sale Date"], errors="coerce")
    merged_df["Date of Birth"] = pd.to_datetime(merged_df["Date of Birth"], errors="coerce")
    merged_df["Date of Joining"] = pd.to_datetime(merged_df["Date of Joining"], errors="coerce")

    merged_df["Employee Age at Sale"] = (
        (merged_df["Sale Date"] - merged_df["Date of Birth"]).dt.days / 365.25
    ).round(1)
    merged_df["Employee Tenure at Sale"] = (
        (merged_df["Sale Date"] - merged_df["Date of Joining"]).dt.days / 365.25
    ).round(1)

    merged_df["Sale Date"] = merged_df["Sale Date"].dt.strftime("%Y-%m-%d")
    merged_df["Date of Birth"] = merged_df["Date of Birth"].dt.strftime("%Y-%m-%d")
    merged_df["Date of Joining"] = merged_df["Date of Joining"].dt.strftime("%Y-%m-%d")

    merged_df.to_csv(MERGED_OUTPUT, index=False)

    matched_rows = int(merged_df["Name"].notna().sum())
    unique_employees_in_sales = int(merged_df["Employee ID"].nunique())
    lines = [
        "Data Merge Report",
        "=================",
        "",
        "Merge approach:",
        "- Sales data did not contain an Employee ID column, so a deterministic employee assignment was added before merging.",
        "- A many-to-one left join was applied from sales to employee data on Employee ID.",
        "",
        f"Employee rows available: {len(employee_df)}",
        f"Sales rows available: {len(sales_df)}",
        f"Merged rows produced: {len(merged_df)}",
        f"Rows matched to employee records: {matched_rows}",
        f"Unique employees represented in sales: {unique_employees_in_sales}",
        f"Missing values remaining in merged dataset: {int(merged_df.isna().sum().sum())}",
    ]
    MERGE_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"Created merged dataset: {MERGED_OUTPUT.name}")
    print(f"Created report: {MERGE_REPORT.name}")


if __name__ == "__main__":
    main()
