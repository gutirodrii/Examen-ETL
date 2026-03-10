from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
EMPLOYEE_PATH = BASE_DIR / "employee_information_dirty.csv"
SALES_PATH = BASE_DIR / "sales_data_dirty.csv"
REPORT_PATH = BASE_DIR / "data_profiling_report.txt"
EMPLOYEE_HEATMAP_PATH = BASE_DIR / "employee_missing_heatmap.png"
SALES_HEATMAP_PATH = BASE_DIR / "sales_missing_heatmap.png"


def build_dataset_profile(name: str, df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    lines.append(f"{name}")
    lines.append("=" * len(name))
    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append("")
    lines.append("Columns and data types:")
    lines.append(df.dtypes.to_string())
    lines.append("")

    lines.append("Missing values by column:")
    missing = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percent": (df.isna().mean() * 100).round(2),
        }
    )
    lines.append(missing.to_string())
    lines.append("")

    duplicate_rows = int(df.duplicated().sum())
    lines.append(f"Duplicate rows: {duplicate_rows}")
    lines.append("")

    numeric_summary = df.describe(include="number").transpose()
    if not numeric_summary.empty:
        lines.append("Numeric summary:")
        lines.append(numeric_summary.round(2).to_string())
        lines.append("")

    categorical_summary = df.describe(include="object").transpose()
    if not categorical_summary.empty:
        lines.append("Categorical summary:")
        lines.append(categorical_summary.to_string())
        lines.append("")

    if name == "Employee Information":
        salary = pd.to_numeric(df["Salary"], errors="coerce")
        invalid_salaries = df[(salary < 0) | (salary > 500000) | (salary == 0)]
        lines.append("Invalid salary rows (negative, zero, or unrealistic > 500000):")
        lines.append(str(len(invalid_salaries)))
        lines.append("")

        parsed_birth = pd.to_datetime(df["Date of Birth"], errors="coerce", dayfirst=True)
        parsed_join = pd.to_datetime(df["Date of Joining"], errors="coerce", dayfirst=True)
        birth_mismatches = int(df["Date of Birth"].notna().sum() - parsed_birth.notna().sum())
        join_mismatches = int(df["Date of Joining"].notna().sum() - parsed_join.notna().sum())
        lines.append(f"Date of Birth parsing failures: {birth_mismatches}")
        lines.append(f"Date of Joining parsing failures: {join_mismatches}")
        lines.append("")

        gender_values = sorted(df["Gender"].dropna().astype(str).unique().tolist())
        standardized_gender = {"male", "female", "m", "f"}
        invalid_gender_values = [value for value in gender_values if value.strip().lower() not in standardized_gender]
        lines.append("Observed gender values:")
        lines.append(", ".join(gender_values) if gender_values else "None")
        lines.append("")
        lines.append("Inconsistent gender values:")
        lines.append(", ".join(invalid_gender_values) if invalid_gender_values else "None")
        lines.append("")

    if name == "Sales Data":
        quantity = pd.to_numeric(df["Quantity Sold"], errors="coerce")
        sale_amount = pd.to_numeric(df["Sale Amount"], errors="coerce")
        negative_qty = int((quantity < 0).sum())
        negative_sales = int((sale_amount < 0).sum())
        parsed_sale_dates = pd.to_datetime(df["Sale Date"], errors="coerce", dayfirst=True)
        sale_date_failures = int(df["Sale Date"].notna().sum() - parsed_sale_dates.notna().sum())

        lines.append(f"Negative Quantity Sold rows: {negative_qty}")
        lines.append(f"Negative Sale Amount rows: {negative_sales}")
        lines.append(f"Sale Date parsing failures: {sale_date_failures}")
        lines.append("")

        payment_values = sorted(df["Payment Method"].dropna().astype(str).unique().tolist())
        lines.append("Observed payment method values:")
        lines.append(", ".join(payment_values) if payment_values else "None")
        lines.append("")

    return lines


def save_missing_heatmap(df: pd.DataFrame, title: str, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    employee_df = pd.read_csv(EMPLOYEE_PATH)
    sales_df = pd.read_csv(SALES_PATH)

    report_lines: list[str] = []
    report_lines.extend(build_dataset_profile("Employee Information", employee_df))
    report_lines.append("")
    report_lines.extend(build_dataset_profile("Sales Data", sales_df))
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    save_missing_heatmap(
        employee_df,
        "Employee Information Missing Values Heatmap",
        EMPLOYEE_HEATMAP_PATH,
    )
    save_missing_heatmap(
        sales_df,
        "Sales Data Missing Values Heatmap",
        SALES_HEATMAP_PATH,
    )

    print(f"Created report: {REPORT_PATH.name}")
    print(f"Created heatmap: {EMPLOYEE_HEATMAP_PATH.name}")
    print(f"Created heatmap: {SALES_HEATMAP_PATH.name}")


if __name__ == "__main__":
    main()
