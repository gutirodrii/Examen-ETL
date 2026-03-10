from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
MERGED_INPUT = BASE_DIR / "employee_sales_merged.csv"

SALES_BY_DEPARTMENT_PLOT = BASE_DIR / "bi_sales_by_department.png"
SALES_BY_CATEGORY_PLOT = BASE_DIR / "bi_sales_by_product_category.png"
SALARY_PERFORMANCE_PLOT = BASE_DIR / "bi_salary_vs_performance.png"
PAYMENT_METHOD_PLOT = BASE_DIR / "bi_payment_method_vs_sale_amount.png"
BI_REPORT = BASE_DIR / "business_intelligence_report.txt"


def save_sales_by_department(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("Department", dropna=False)["Sale Amount"]
        .agg(total_sales="sum", average_sales="mean")
        .sort_values("total_sales", ascending=False)
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=summary, x="Department", y="total_sales", hue="Department", legend=False, ax=axes[0], palette="Blues_d")
    axes[0].set_title("Total Sales by Department")
    axes[0].set_xlabel("Department")
    axes[0].set_ylabel("Total Sales")
    axes[0].tick_params(axis="x", rotation=30)

    sns.barplot(data=summary, x="Department", y="average_sales", hue="Department", legend=False, ax=axes[1], palette="Greens_d")
    axes[1].set_title("Average Sale Amount by Department")
    axes[1].set_xlabel("Department")
    axes[1].set_ylabel("Average Sale Amount")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(SALES_BY_DEPARTMENT_PLOT, dpi=200)
    plt.close()
    return summary


def save_sales_by_category(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("Product Category", dropna=False)["Sale Amount"]
        .agg(total_sales="sum")
        .sort_values("total_sales", ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(10, 7))
    colors = sns.color_palette("Set2", n_colors=len(summary))
    plt.pie(summary["total_sales"], labels=summary["Product Category"], autopct="%1.1f%%", startangle=140, colors=colors)
    plt.title("Sales Contribution by Product Category")
    plt.tight_layout()
    plt.savefig(SALES_BY_CATEGORY_PLOT, dpi=200)
    plt.close()
    return summary


def save_salary_vs_performance(df: pd.DataFrame) -> float:
    employee_points = (
        df[["Employee ID", "Salary", "Performance Score", "Department"]]
        .drop_duplicates(subset=["Employee ID"])
        .copy()
    )

    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=employee_points,
        x="Salary",
        y="Performance Score",
        scatter_kws={"s": 70, "alpha": 0.8},
        line_kws={"color": "darkred"},
    )
    plt.title("Salary vs Performance Score")
    plt.xlabel("Salary")
    plt.ylabel("Performance Score")
    plt.tight_layout()
    plt.savefig(SALARY_PERFORMANCE_PLOT, dpi=200)
    plt.close()

    return float(employee_points["Salary"].corr(employee_points["Performance Score"]))


def save_payment_method_analysis(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("Payment Method", dropna=False)["Sale Amount"]
        .agg(total_sales="sum", average_sales="mean")
        .sort_values("total_sales", ascending=False)
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=summary, x="Payment Method", y="total_sales", hue="Payment Method", legend=False, ax=axes[0], palette="Oranges_d")
    axes[0].set_title("Total Sales by Payment Method")
    axes[0].set_xlabel("Payment Method")
    axes[0].set_ylabel("Total Sales")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=summary, x="Payment Method", y="average_sales", hue="Payment Method", legend=False, ax=axes[1], palette="Purples_d")
    axes[1].set_title("Average Sale Amount by Payment Method")
    axes[1].set_xlabel("Payment Method")
    axes[1].set_ylabel("Average Sale Amount")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(PAYMENT_METHOD_PLOT, dpi=200)
    plt.close()
    return summary


def write_report(
    department_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    salary_performance_corr: float,
    payment_summary: pd.DataFrame,
) -> None:
    top_department_total = department_summary.iloc[0]
    top_department_avg = department_summary.sort_values("average_sales", ascending=False).iloc[0]
    top_category = category_summary.iloc[0]
    top_payment_total = payment_summary.iloc[0]
    top_payment_avg = payment_summary.sort_values("average_sales", ascending=False).iloc[0]

    lines = [
        "Business Intelligence Report",
        "============================",
        "",
        "Sales by Department",
        f"- Highest total sales: {top_department_total['Department']} ({top_department_total['total_sales']:.2f})",
        f"- Highest average sale amount: {top_department_avg['Department']} ({top_department_avg['average_sales']:.2f})",
        "",
        "Sales by Product Category",
        f"- Top product category by total sales: {top_category['Product Category']} ({top_category['total_sales']:.2f})",
        "",
        "Salary vs. Performance",
        f"- Pearson correlation between salary and performance score: {salary_performance_corr:.3f}",
        "",
        "Payment Method vs. Sale Amount",
        f"- Highest total sales payment method: {top_payment_total['Payment Method']} ({top_payment_total['total_sales']:.2f})",
        f"- Highest average sale amount payment method: {top_payment_avg['Payment Method']} ({top_payment_avg['average_sales']:.2f})",
        "",
        "Detailed tables",
        "",
        "Department summary:",
        department_summary.round(2).to_string(index=False),
        "",
        "Category summary:",
        category_summary.round(2).to_string(index=False),
        "",
        "Payment summary:",
        payment_summary.round(2).to_string(index=False),
    ]
    BI_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    sns.set_theme(style="whitegrid")
    merged_df = pd.read_csv(MERGED_INPUT)

    department_summary = save_sales_by_department(merged_df)
    category_summary = save_sales_by_category(merged_df)
    salary_performance_corr = save_salary_vs_performance(merged_df)
    payment_summary = save_payment_method_analysis(merged_df)

    write_report(
        department_summary=department_summary,
        category_summary=category_summary,
        salary_performance_corr=salary_performance_corr,
        payment_summary=payment_summary,
    )

    print(f"Created plot: {SALES_BY_DEPARTMENT_PLOT.name}")
    print(f"Created plot: {SALES_BY_CATEGORY_PLOT.name}")
    print(f"Created plot: {SALARY_PERFORMANCE_PLOT.name}")
    print(f"Created plot: {PAYMENT_METHOD_PLOT.name}")
    print(f"Created report: {BI_REPORT.name}")


if __name__ == "__main__":
    main()
