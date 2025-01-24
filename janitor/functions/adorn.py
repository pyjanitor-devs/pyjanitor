from typing import Optional

import pandas as pd


def tabyl(
    df: pd.DataFrame,
    col1: str,
    col2: Optional[str] = None,
    col3: Optional[str] = None,
    show_counts: bool = True,
    show_percentages: bool = False,
    percentage_axis: Optional[str] = None,  # 'row', 'col', or 'all'
) -> pd.DataFrame:
    """
    Create a summary table similar to R's `tabyl`.

    Args:
        df: Input DataFrame.
        col1: Name of the first column for grouping (required).
        col2: Name of the second column for grouping (optional).
        col3: Name of the third column for grouping (optional).
        show_counts: Whether to show raw counts in the table.
        show_percentages: Whether to show percentages in the table.
        percentage_axis: Axis for percentages ('row', 'col', or 'all').
        Only applies if `show_percentages` is True.

    Returns:
        A DataFrame representing the summary table.
    """
    if col1 not in df.columns:
        raise ValueError(f"Column '{col1}' is not in the DataFrame.")
    if col2 and col2 not in df.columns:
        raise ValueError(f"Column '{col2}' is not in the DataFrame.")
    if col3 and col3 not in df.columns:
        raise ValueError(f"Column '{col3}' is not in the DataFrame.")

    # Step 1: Group and count
    group_cols = [col1]
    if col2:
        group_cols.append(col2)
    if col3:
        group_cols.append(col3)

    grouped = df.groupby(group_cols).size().reset_index(name="count")

    # Step 2: Pivot for 3D (col1, col2, col3)
    if col2 and col3:
        pivot = grouped.pivot_table(
            index=col1,
            columns=[col2, col3],  # Creating 2-level columns for col2 and col3
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
    elif col2:
        pivot = grouped.pivot_table(
            index=col1,
            columns=col2,
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
    else:
        pivot = grouped.set_index(col1)["count"].to_frame()

    if show_percentages:
        pivot = pivot.astype(
            float
        )  # Convert to float before calculating percentages

        if percentage_axis == "row":
            percentages = pivot.div(pivot.sum(axis=1), axis=0)
        elif percentage_axis == "col":
            percentages = pivot.div(pivot.sum(axis=0), axis=1)
        elif percentage_axis == "all":
            total = pivot.values.sum()
            percentages = pivot / total
        else:
            raise ValueError(
                "`percentage_axis` must be one of 'row', 'col', or 'all'."
            )

        percentages = percentages.applymap(lambda x: f"{x:.2%}")

        if show_counts:
            pivot = pivot.astype(str) + " (" + percentages + ")"
        else:
            pivot = percentages

    return pivot.reset_index()
