from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

EMPLOYEE_BEFORE = BASE_DIR / "employee_information_dirty.csv"
EMPLOYEE_AFTER = BASE_DIR / "employee_information_cleaned.csv"
SALES_BEFORE = BASE_DIR / "sales_data_dirty.csv"
SALES_AFTER = BASE_DIR / "sales_data_cleaned.csv"

EMPLOYEE_CHANGES = BASE_DIR / "employee_cdc_changes.csv"
SALES_CHANGES = BASE_DIR / "sales_cdc_changes.csv"
CDC_REPORT = BASE_DIR / "cdc_report.txt"


def normalize_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].map(
            lambda value: "<MISSING>" if pd.isna(value) else str(value).strip()
        )
    return normalized


def compute_cdc(before_df: pd.DataFrame, after_df: pd.DataFrame, key_column: str) -> pd.DataFrame:
    before = normalize_for_compare(before_df).set_index(key_column, drop=False)
    after = normalize_for_compare(after_df).set_index(key_column, drop=False)

    before_keys = set(before.index)
    after_keys = set(after.index)
    common_keys = sorted(before_keys & after_keys)
    added_keys = sorted(after_keys - before_keys)
    deleted_keys = sorted(before_keys - after_keys)

    change_rows: list[dict[str, str]] = []

    for record_key in common_keys:
        before_row = before.loc[record_key]
        after_row = after.loc[record_key]
        for column in before.columns:
            if column == key_column:
                continue
            before_value = before_row[column]
            after_value = after_row[column]
            if before_value != after_value:
                change_rows.append(
                    {
                        "record_key": record_key,
                        "change_type": "modified",
                        "column_name": column,
                        "old_value": before_value,
                        "new_value": after_value,
                    }
                )

    for record_key in added_keys:
        added_row = after.loc[record_key]
        change_rows.append(
            {
                "record_key": record_key,
                "change_type": "added",
                "column_name": "<ROW>",
                "old_value": "<NOT_PRESENT>",
                "new_value": added_row.to_json(),
            }
        )

    for record_key in deleted_keys:
        deleted_row = before.loc[record_key]
        change_rows.append(
            {
                "record_key": record_key,
                "change_type": "deleted",
                "column_name": "<ROW>",
                "old_value": deleted_row.to_json(),
                "new_value": "<NOT_PRESENT>",
            }
        )

    return pd.DataFrame(change_rows)


def build_summary(name: str, before_df: pd.DataFrame, after_df: pd.DataFrame, changes_df: pd.DataFrame) -> list[str]:
    modified_records = changes_df.loc[changes_df["change_type"] == "modified", "record_key"].nunique()
    added_records = int((changes_df["change_type"] == "added").sum())
    deleted_records = int((changes_df["change_type"] == "deleted").sum())
    modified_fields = int((changes_df["change_type"] == "modified").sum())

    return [
        name,
        "=" * len(name),
        f"Rows before cleaning: {len(before_df)}",
        f"Rows after cleaning: {len(after_df)}",
        f"Modified records: {modified_records}",
        f"Modified fields: {modified_fields}",
        f"Added records: {added_records}",
        f"Deleted records: {deleted_records}",
        "",
    ]


def main() -> None:
    employee_before_df = pd.read_csv(EMPLOYEE_BEFORE)
    employee_after_df = pd.read_csv(EMPLOYEE_AFTER)
    sales_before_df = pd.read_csv(SALES_BEFORE)
    sales_after_df = pd.read_csv(SALES_AFTER)

    employee_changes = compute_cdc(employee_before_df, employee_after_df, "Employee ID")
    sales_changes = compute_cdc(sales_before_df, sales_after_df, "Transaction ID")

    employee_changes.to_csv(EMPLOYEE_CHANGES, index=False)
    sales_changes.to_csv(SALES_CHANGES, index=False)

    lines: list[str] = ["CDC Report", "==========", ""]
    lines.extend(build_summary("Employee Information", employee_before_df, employee_after_df, employee_changes))
    lines.extend(build_summary("Sales Data", sales_before_df, sales_after_df, sales_changes))
    lines.append("Notes:")
    lines.append("- Cleaning changed field values but did not add or remove business records in this workflow.")
    lines.append("- Added/deleted counts will remain zero unless future source extracts change row membership.")
    CDC_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"Created CDC change log: {EMPLOYEE_CHANGES.name}")
    print(f"Created CDC change log: {SALES_CHANGES.name}")
    print(f"Created report: {CDC_REPORT.name}")


if __name__ == "__main__":
    main()
