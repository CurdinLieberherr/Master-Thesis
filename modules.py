import zipfile
import re
import os
import pandas as pd

def read_orbis_excel(filepath, sheet_name="Ergebnisse", **kwargs):
    """
    Fixes the applyNumFmt CellStyle error from Orbis/BvD exports
    by patching the styles.xml inside the xlsx file before reading.
    """

    #only fix original files
    if filepath.endswith('_fixed.xlsx'):
        fixed_path = filepath
    
    else: 
        fixed_path = filepath.replace(".xlsx", "_fixed.xlsx")

        # Read all files from original xlsx
        with zipfile.ZipFile(filepath, "r") as zin:
            files = {}
            for name in zin.namelist():
                files[name] = zin.read(name)

        # Patch styles.xml - remove problematic apply* attributes
        styles = files["xl/styles.xml"].decode("utf-8")
        styles_fixed = re.sub(
            r'\s+apply(NumFmt|Font|Fill|Border|Alignment|Protection)=\"[^\"]*\"',
            "",
            styles
        )
        files["xl/styles.xml"] = styles_fixed.encode("utf-8")

        # Write fixed xlsx
        with zipfile.ZipFile(fixed_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for name, data in files.items():
                zout.writestr(name, data)

        #delete old file 
        os.remove(filepath)

    # Read and return dataframe
    df = pd.read_excel(fixed_path, sheet_name=sheet_name, **kwargs)
    return df

def melt_orbis_df(df, id_cols):
    # melt all year columns at once
    year_cols = [c for c in df.columns if re.search(r"\d{4}", c)]

    df_long = df[id_cols + year_cols].melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="col_raw",
        value_name="value"
    )

    # extract year and variable name
    df_long["year"] = df_long["col_raw"].str.extract(r"(\d{4})").astype(int)
    df_long["variable"] = df_long["col_raw"].str.replace(r"\s*\d{4}$", "", regex=True).str.strip()

    # pivot back to wide with one column per variable
    df_long = df_long.drop(columns="col_raw").pivot_table(
        index=id_cols + ["year"],
        columns="variable",
        values="value",
        aggfunc="first"
    ).reset_index()

    return df_long