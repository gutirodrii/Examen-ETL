from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / name)


def build_dataset_profile(name: str, df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append(name)
    lines.append("=" * len(name))
    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append("")
    lines.append("Columns and data types:")
    lines.append(df.dtypes.to_string())
    lines.append("")
    lines.append("Missing values by column:")
    lines.append(
        pd.DataFrame(
            {
                "missing_count": df.isna().sum(),
                "missing_percent": (df.isna().mean() * 100).round(2),
            }
        ).to_string()
    )
    lines.append("")
    lines.append(f"Duplicate rows: {int(df.duplicated().sum())}")
    lines.append("")

    numeric_summary = df.describe(include="number").transpose()
    if not numeric_summary.empty:
        lines.append("Numeric summary:")
        lines.append(numeric_summary.round(2).to_string())
        lines.append("")

    if name == "Employee Information":
        salary = pd.to_numeric(df["Salary"], errors="coerce")
        invalid_salaries = int(((salary < 0) | (salary > 500000) | (salary == 0)).sum())
        lines.append(f"Invalid salary rows: {invalid_salaries}")
        lines.append("")

    if name == "Sales Data":
        quantity = pd.to_numeric(df["Quantity Sold"], errors="coerce")
        lines.append(f"Negative Quantity Sold rows: {int((quantity < 0).sum())}")
        lines.append("")

    return "\n".join(lines)


def build_filtered_dataset(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    filtered["Sale Date"] = pd.to_datetime(filtered["Sale Date"], errors="coerce")

    st.sidebar.header("Filtros")
    departments = sorted(filtered["Department"].dropna().unique().tolist())
    categories = sorted(filtered["Product Category"].dropna().unique().tolist())
    payment_methods = sorted(filtered["Payment Method"].dropna().unique().tolist())

    selected_departments = st.sidebar.multiselect("Department", departments, default=departments)
    selected_categories = st.sidebar.multiselect("Product Category", categories, default=categories)
    selected_payment_methods = st.sidebar.multiselect("Payment Method", payment_methods, default=payment_methods)

    min_date = filtered["Sale Date"].min().date()
    max_date = filtered["Sale Date"].max().date()
    selected_dates = st.sidebar.date_input(
        "Rango de fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if len(selected_dates) == 2:
        start_date, end_date = selected_dates
        filtered = filtered[filtered["Sale Date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]

    filtered = filtered[
        filtered["Department"].isin(selected_departments)
        & filtered["Product Category"].isin(selected_categories)
        & filtered["Payment Method"].isin(selected_payment_methods)
    ]
    return filtered


def render_overview(
    filtered_df: pd.DataFrame,
    employee_dirty: pd.DataFrame,
    sales_dirty: pd.DataFrame,
    employee_cdc: pd.DataFrame,
    sales_cdc: pd.DataFrame,
) -> None:
    st.subheader("Resumen Ejecutivo")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ventas filtradas", f"{len(filtered_df):,}")
    col2.metric("Total Sales Amount", f"{filtered_df['Sale Amount'].sum():,.2f}")
    col3.metric("Average Sale Amount", f"{filtered_df['Sale Amount'].mean():,.2f}")
    col4.metric("Employees in Scope", f"{filtered_df['Employee ID'].nunique():,}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Employee Rows Lost if Drop Nulls", f"{len(employee_dirty) - len(employee_dirty.dropna()):,}")
    col6.metric("Sales Rows Lost if Drop Nulls", f"{len(sales_dirty) - len(sales_dirty.dropna()):,}")
    col7.metric(
        "Modified Employee Records",
        f"{employee_cdc.loc[employee_cdc['change_type'].eq('modified'), 'record_key'].nunique():,}",
    )
    col8.metric(
        "Modified Sales Records",
        f"{sales_cdc.loc[sales_cdc['change_type'].eq('modified'), 'record_key'].nunique():,}",
    )


def render_profiling(employee_dirty: pd.DataFrame, sales_dirty: pd.DataFrame) -> None:
    st.subheader("Data Profiling")

    col1, col2 = st.columns(2)
    with col1:
        st.text_area("Employee profiling", build_dataset_profile("Employee Information", employee_dirty), height=420)
    with col2:
        st.text_area("Sales profiling", build_dataset_profile("Sales Data", sales_dirty), height=420)

    heatmap_employee = px.imshow(
        employee_dirty.isna().astype(int),
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Employee Missing Values Heatmap",
    )
    heatmap_sales = px.imshow(
        sales_dirty.isna().astype(int),
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Sales Missing Values Heatmap",
    )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(heatmap_employee, use_container_width=True)
    with col4:
        st.plotly_chart(heatmap_sales, use_container_width=True)


def render_cleaning(employee_clean: pd.DataFrame, sales_clean: pd.DataFrame) -> None:
    st.subheader("Data Cleaning")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Employee missing after cleaning", int(employee_clean.isna().sum().sum()))
    col2.metric("Sales missing after cleaning", int(sales_clean.isna().sum().sum()))
    col3.metric(
        "Invalid salary after cleaning",
        int(
            (
                (pd.to_numeric(employee_clean["Salary"], errors="coerce") <= 0)
                | (pd.to_numeric(employee_clean["Salary"], errors="coerce") > 500000)
            ).sum()
        ),
    )
    col4.metric(
        "Negative quantity after cleaning",
        int((pd.to_numeric(sales_clean["Quantity Sold"], errors="coerce") < 0).sum()),
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**cleaned_employee_data.csv**")
        st.dataframe(employee_clean.head(15), use_container_width=True)
    with right:
        st.markdown("**cleaned_sales_data.csv**")
        st.dataframe(sales_clean.head(15), use_container_width=True)


def render_merge_and_cdc(merged_df: pd.DataFrame, employee_cdc: pd.DataFrame, sales_cdc: pd.DataFrame) -> None:
    st.subheader("Merge y Change Data Capture")

    col1, col2, col3 = st.columns(3)
    col1.metric("Merged Rows", f"{len(merged_df):,}")
    col2.metric("Merged Columns", f"{merged_df.shape[1]:,}")
    col3.metric("Remaining Nulls", f"{int(merged_df.isna().sum().sum()):,}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Employee CDC New", int((employee_cdc["change_type"] == "added").sum()))
    col5.metric("Employee CDC Deleted", int((employee_cdc["change_type"] == "deleted").sum()))
    col6.metric("Employee CDC Modified", employee_cdc.loc[employee_cdc["change_type"].eq("modified"), "record_key"].nunique())

    col7, col8, col9 = st.columns(3)
    col7.metric("Sales CDC New", int((sales_cdc["change_type"] == "added").sum()))
    col8.metric("Sales CDC Deleted", int((sales_cdc["change_type"] == "deleted").sum()))
    col9.metric("Sales CDC Modified", sales_cdc.loc[sales_cdc["change_type"].eq("modified"), "record_key"].nunique())

    tab1, tab2 = st.tabs(["Employee CDC", "Sales CDC"])
    with tab1:
        st.dataframe(employee_cdc.head(100), use_container_width=True)
    with tab2:
        st.dataframe(sales_cdc.head(100), use_container_width=True)


def render_bi(filtered_df: pd.DataFrame) -> None:
    st.subheader("Business Intelligence")

    department_summary = (
        filtered_df.groupby("Department", as_index=False)["Sale Amount"]
        .agg(total_sales="sum", average_sales="mean")
        .sort_values("total_sales", ascending=False)
    )
    category_summary = (
        filtered_df.groupby("Product Category", as_index=False)["Sale Amount"]
        .sum()
        .sort_values("Sale Amount", ascending=False)
    )
    payment_summary = (
        filtered_df.groupby("Payment Method", as_index=False)["Sale Amount"]
        .agg(total_sales="sum", average_sales="mean")
        .sort_values("total_sales", ascending=False)
    )
    employee_points = (
        filtered_df[["Employee ID", "Department", "Salary", "Performance Score"]]
        .drop_duplicates(subset=["Employee ID"])
        .copy()
    )

    salary_corr = float(employee_points["Salary"].corr(employee_points["Performance Score"]))

    fig_department_total = px.bar(
        department_summary,
        x="Department",
        y="total_sales",
        color="Department",
        title="Sales by Department: Total Sales",
    )
    fig_department_avg = px.bar(
        department_summary,
        x="Department",
        y="average_sales",
        color="Department",
        title="Sales by Department: Average Sale Amount",
    )
    fig_category = px.pie(
        category_summary,
        names="Product Category",
        values="Sale Amount",
        title="Sales by Product Category",
        hole=0.35,
    )
    fig_salary_perf = px.scatter(
        employee_points,
        x="Salary",
        y="Performance Score",
        color="Department",
        hover_data=["Employee ID"],
        title=f"Salary vs. Performance (corr={salary_corr:.3f})",
        trendline="ols",
    )
    fig_payment_total = px.bar(
        payment_summary,
        x="Payment Method",
        y="total_sales",
        color="Payment Method",
        title="Payment Method vs. Sale Amount: Total",
    )
    fig_payment_avg = px.bar(
        payment_summary,
        x="Payment Method",
        y="average_sales",
        color="Payment Method",
        title="Payment Method vs. Sale Amount: Average",
    )

    st.plotly_chart(fig_department_total, use_container_width=True)
    st.plotly_chart(fig_department_avg, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_category, use_container_width=True)
    with col2:
        st.plotly_chart(fig_salary_perf, use_container_width=True)

    st.plotly_chart(fig_payment_total, use_container_width=True)
    st.plotly_chart(fig_payment_avg, use_container_width=True)

    top_department = department_summary.iloc[0]
    top_category = category_summary.iloc[0]
    top_payment = payment_summary.iloc[0]
    st.markdown(
        f"""
        **Insights actuales**

        - Highest total sales department: `{top_department['Department']}` with `{top_department['total_sales']:.2f}`
        - Top product category: `{top_category['Product Category']}` with `{top_category['Sale Amount']:.2f}`
        - Payment method with highest total sales: `{top_payment['Payment Method']}` with `{top_payment['total_sales']:.2f}`
        - Salary vs Performance correlation: `{salary_corr:.3f}`
        """
    )


def render_downloads() -> None:
    st.subheader("Descargas")

    files = [
        "exam.ipynb",
        "employee_information_dirty.csv",
        "sales_data_dirty.csv",
        "cleaned_employee_data.csv",
        "cleaned_sales_data.csv",
        "merged_analysis_data.csv",
        "employee_cdc_changes.csv",
        "sales_cdc_changes.csv",
    ]

    for file_name in files:
        data = (BASE_DIR / file_name).read_bytes()
        st.download_button(
            label=f"Download {file_name}",
            data=data,
            file_name=file_name,
            mime="application/octet-stream",
        )


def main() -> None:
    st.set_page_config(page_title="ETL Project Dashboard", page_icon="📊", layout="wide")

    st.title("ETL Project Dashboard")
    st.caption("Notebook-driven ETL workflow with profiling, cleaning, CDC, and BI")

    employee_dirty = load_csv("employee_information_dirty.csv")
    sales_dirty = load_csv("sales_data_dirty.csv")
    employee_clean = load_csv("cleaned_employee_data.csv")
    sales_clean = load_csv("cleaned_sales_data.csv")
    merged_df = load_csv("merged_analysis_data.csv")
    employee_cdc = load_csv("employee_cdc_changes.csv")
    sales_cdc = load_csv("sales_cdc_changes.csv")

    filtered_df = build_filtered_dataset(merged_df)

    overview_tab, profiling_tab, cleaning_tab, merge_tab, bi_tab, downloads_tab = st.tabs(
        ["Overview", "Profiling", "Cleaning", "Merge & CDC", "BI", "Downloads"]
    )

    with overview_tab:
        render_overview(filtered_df, employee_dirty, sales_dirty, employee_cdc, sales_cdc)
    with profiling_tab:
        render_profiling(employee_dirty, sales_dirty)
    with cleaning_tab:
        render_cleaning(employee_clean, sales_clean)
    with merge_tab:
        render_merge_and_cdc(merged_df, employee_cdc, sales_cdc)
    with bi_tab:
        render_bi(filtered_df)
    with downloads_tab:
        render_downloads()


if __name__ == "__main__":
    main()
