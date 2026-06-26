import pandas as pd
import numpy as np
import re
import os
import zipfile
from pathlib import Path


def prepare_orbis_excel(folder):
    directory = Path(folder)
    files = list(map(str, directory.glob("*.xlsx")))

    # Usage
    df = pd.DataFrame()
    for file in files:
        dffile = read_orbis_excel(
            file,
            sheet_name="Ergebnisse", dtype = str
        )
        df = pd.concat([df, dffile])


    #replace n.v. with na
    df = df.replace('n.v.', np.nan)

    DESC_COLS = ['Unternehmensname Latin alphabet',
        'NACE Rev. 2 Core Code (4 Ziffern)']

    #take num cols and drop 2014 and betriebsertrag
    NUM_COLS = [col for col in df.columns if re.search(r'\b\d{4}\b', str(col)) and 'EUR' in col] + [col for col in df.columns if 'Anzahl der Mitarbeiter' in col]

    df = df[DESC_COLS + NUM_COLS]

    #make num cols to numeric
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    #make data long for any variable col
    df_long = melt_orbis_df(df, DESC_COLS)
    df_long

    #rename columns for later use
    df_long = df_long.rename(columns={
        
    })

    namedict = {
        'Unternehmensname Latin alphabet' : 'FirmName',
        'NACE Rev. 2 Core Code (4 Ziffern)':   'sector',
        'Anlagevermögen tsd EUR' : 'assets',
        'Umsatz tsd EUR' : 'revenue',
        'Materialkosten verkaufter Güter tsd EUR': 'materials',
        'Anzahl der Mitarbeiter': 'nEmployees',
        'Mitarbeiterkosten tsd EUR': 'wagebill',
        "Durchschnittlicher Mitarbeiterkosten tsd EUR": "wage",
        "Langfristige Finanzschulden tsd EUR": 'debt'
    }

    df_long = df_long.rename(columns=namedict)

    return df_long

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
