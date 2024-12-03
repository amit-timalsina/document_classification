import pandas as pd


def df_to_text(df: pd.DataFrame) -> str:
    """Convert dataframe to plain text conserving blocks and lines."""
    # TODO (Amit): Isn't this also a type of formatter? The only problem is it takes a dataframe as
    # input.
    text = ""
    for row in df.itertuples():
        text += str(row.text)
        if row.space_type == 0:
            continue
        if row.space_type in [1, 2]:
            text += " "
        elif row.space_type in [3, 4]:
            text += "\n"
        elif row.space_type is None:
            text += " "
        else:
            text += "\n\n"
    return text.strip()
