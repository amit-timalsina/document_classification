import pandas as pd


def df_to_text(df: pd.DataFrame) -> str:
    """Convert dataframe to plain text conserving blocks and lines."""
    text = ""
    for row in df.itertuples():
        text += str(row.text)
        if row.space_type == 0:
            continue
        if row.space_type in [1, 2]:
            text += " "
        elif row.space_type in [3, 4]:
            text += "\n"
        else:
            text += "\n\n"
    return text.strip()
