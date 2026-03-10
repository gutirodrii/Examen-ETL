from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / name)


@st.cache_data
def load_text(name: str) -> str:
    return (BASE_DIR / name).read_text(encoding="utf-8")


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


def render_overview(filtered_df: pd.DataFrame, employee_dirty: pd.DataFrame, sales_dirty: pd.DataFrame) -> None:
    st.subheader("Resumen Ejecutivo")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ventas filtradas", f"{len(filtered_df):,}")
    col2.metric("Total Sales Amount", f"{filtered_df['Sale Amount'].sum():,.2f}")
    col3.metric("Average Sale Amount", f"{filtered_df['Sale Amount'].mean():,.2f}")
    col4.metric("Employees in Scope", f"{filtered_df['Employee ID'].nunique():,}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Employee Rows Lost if Drop Nulls", f"{len(employee_dirty) - len(employee_dirty.dropna()):,}")
    col6.metric("Sales Rows Lost if Drop Nulls", f"{len(sales_dirty) - len(sales_dirty.dropna()):,}")
    col7.metric("Modified Employee Records", "99")
    col8.metric("Modified Sales Records", "120")

    st.markdown(
        """
        Esta dashboard consolida todo el flujo del proyecto:
        generación de datos sintéticos con errores, profiling, limpieza, merge, CDC y análisis de negocio.
        """
    )


def render_profiling() -> None:
    st.subheader("Data Profiling")

    st.text_area(
        "Reporte de profiling",
        load_text("data_profiling_report.txt"),
        height=420,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(str(BASE_DIR / "employee_missing_heatmap.png"), caption="Missing Values Heatmap: Employee Information")
    with col2:
        st.image(str(BASE_DIR / "sales_missing_heatmap.png"), caption="Missing Values Heatmap: Sales Data")


def render_cleaning(employee_clean: pd.DataFrame, sales_clean: pd.DataFrame) -> None:
    st.subheader("Data Cleaning")

    st.text_area(
        "Reporte de limpieza",
        load_text("data_cleaning_report.txt"),
        height=260,
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Employee Cleaned Preview**")
        st.dataframe(employee_clean.head(15), use_container_width=True)
    with right:
        st.markdown("**Sales Cleaned Preview**")
        st.dataframe(sales_clean.head(15), use_container_width=True)


def render_merge_and_cdc(merged_df: pd.DataFrame, employee_cdc: pd.DataFrame, sales_cdc: pd.DataFrame) -> None:
    st.subheader("Merge y Change Data Capture")

    col1, col2, col3 = st.columns(3)
    col1.metric("Merged Rows", f"{len(merged_df):,}")
    col2.metric("Merged Columns", f"{merged_df.shape[1]:,}")
    col3.metric("Remaining Nulls", f"{int(merged_df.isna().sum().sum()):,}")

    st.text_area("Reporte de merge", load_text("data_merge_report.txt"), height=180)
    st.text_area("Reporte CDC", load_text("cdc_report.txt"), height=220)

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
        title="Salary vs. Performance",
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

    st.text_area(
        "Reporte BI",
        load_text("business_intelligence_report.txt"),
        height=320,
    )


def render_downloads() -> None:
    st.subheader("Descargas")

    files = [
        "employee_information_dirty.csv",
        "sales_data_dirty.csv",
        "employee_information_cleaned.csv",
        "sales_data_cleaned.csv",
        "employee_sales_merged.csv",
        "employee_cdc_changes.csv",
        "sales_cdc_changes.csv",
        "data_profiling_report.txt",
        "data_cleaning_report.txt",
        "data_merge_report.txt",
        "cdc_report.txt",
        "business_intelligence_report.txt",
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
    st.set_page_config(
        page_title="ETL Project Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("ETL Project Dashboard")
    st.caption("Synthetic Data Generation, Cleaning, CDC and Business Intelligence")

    employee_dirty = load_csv("employee_information_dirty.csv")
    sales_dirty = load_csv("sales_data_dirty.csv")
    employee_clean = load_csv("employee_information_cleaned.csv")
    sales_clean = load_csv("sales_data_cleaned.csv")
    merged_df = load_csv("employee_sales_merged.csv")
    employee_cdc = load_csv("employee_cdc_changes.csv")
    sales_cdc = load_csv("sales_cdc_changes.csv")

    filtered_df = build_filtered_dataset(merged_df)

    overview_tab, profiling_tab, cleaning_tab, merge_tab, bi_tab, downloads_tab = st.tabs(
        ["Overview", "Profiling", "Cleaning", "Merge & CDC", "BI", "Downloads"]
    )

    with overview_tab:
        render_overview(filtered_df, employee_dirty, sales_dirty)
    with profiling_tab:
        render_profiling()
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
