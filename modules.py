import zipfile
import re
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pycountry
import matplotlib.pyplot as plt


    
class RealRate():
    def __init__(self, country: str):
        self.country = country
        self.df = get_real_interest_rate(country)
    
    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.df['year'], self.df['realinterestrate'], marker='o', linestyle='-', label = 'real interest rate')
        plt.plot(self.df['year'], self.df['inflation'], linestyle='--', color='black', label='inflation')
        plt.plot(self.df['year'], self.df['nominalinterestrate'], linestyle=':', color='red', label='nominal interest rate')


        plt.xlabel("Year")
        plt.ylabel("Real Interest Rate")
        plt.title(f"{self.country} Real Interest Rate")
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/realinterestrate.png', dpi=300)

        plt.show()



def get_real_interest_rate(country:str):
    #capital prices eurostat
    file = "Data/Nominal Interest Rate Eurostats.xlsx"
    noi = pd.read_excel(file, sheet_name='Sheet 1', header=10)

    NUMCOLS = [str(year) for year in range(1949, 2026)]
    noi.columns = ['country'] + [
        col for year in NUMCOLS for col in (year, f'{year}_flag')
    ][:-1]

    #drop flag columns
    noi = noi[ [col for col in noi.columns if '_flag' not in col] ]

    #set to numeric
    for col in NUMCOLS:
        noi[col] = pd.to_numeric(noi[col], errors='coerce')


    noi = noi.melt('country', value_vars=NUMCOLS, value_name='nominalinterestrate', var_name='year')
    noi['nominalinterestrate'] = noi['nominalinterestrate'] / 100

    noi = noi[noi['country'] == country]

    #inflation
    file = "Data/Inflation Eurostats.xlsx"
    inf = pd.read_excel(file, sheet_name='Sheet 1', header=10)

    NUMCOLS = [str(year) for year in range(1996, 2026)]
    inf.columns = ['country'] + [
        col for year in NUMCOLS for col in (year, f'{year}_flag')
    ][:-1]
    #drop flag columns
    inf = inf[ [col for col in inf.columns if '_flag' not in col] ]
    #set to numeric
    for col in NUMCOLS:
        inf[col] = pd.to_numeric(inf[col], errors='coerce')
    inf = inf.melt('country', value_vars=NUMCOLS, value_name='inflation', var_name='year')
    inf['inflation'] = inf['inflation'] / 100
    inf = inf[inf['country'] == country]


    rirate = noi.merge(inf, on=['country', 'year'], how='inner')
    rirate['realinterestrate'] = rirate['nominalinterestrate'] - rirate['inflation']

    rirate['year'] = rirate['year'].astype(int)

    return rirate


def capital_prices(country: str):
    #capital prices eurostat
    file = "/Users/curdinlieberherr/Library/Mobile Documents/com~apple~CloudDocs/Uni/26 FS/Thesis/Data Work/Data/EU Capital Prices Eurostat 2021 = 100.xlsx"
    capi = pd.read_excel(file, sheet_name='Sheet 1', header=10)
    capi['country'] = capi.iloc[:,0]
    NUMCOLS = [str(year) for year in range(1992, 2026)]
    capi = capi[['country'] + NUMCOLS]

    #set to numeric
    for col in NUMCOLS:
        capi[col] = pd.to_numeric(capi[col], errors='coerce')
    capi['2015'] = 100

    capi = capi.melt('country', value_vars=['2015']+NUMCOLS, value_name='capitalprice', var_name='year')
    capi['capitalprice'] = capi['capitalprice'] / 100

    capi = capi[capi['country'] == country]

    return capi.sort_values('year', ascending = True).reset_index(drop=True)


def price_indexes(country: str):
    #read data and set header names
    file = '/Users/curdinlieberherr/Library/Mobile Documents/com~apple~CloudDocs/Uni/26 FS/Thesis/Data Work/Data/EU Producer Prices Country Sector 1992 - 2025 2021=100.xlsx'
    df = pd.read_excel(file, sheet_name='Sheet 1', header=9)
    df = df.iloc[2:, :]
    df['country'] = df.iloc[:,0]
    df['sector2d'] = df.iloc[:,1].str[1:]
    NUMCOLS = [str(year) for year in range(1992, 2026)]
    df = df[['country', 'sector2d'] + NUMCOLS]
    for col in NUMCOLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    #only keep spain 
    df = df[df['country'] == country]

    #convert to long format
    df = df.melt('sector2d', value_vars=NUMCOLS, value_name='priceind', var_name='year')

    #divide index by 100 to get decimal
    df['priceind'] = df['priceind'] / 100

    return df.sort_values('year', ascending = True).reset_index(drop=True)

def get_country_name(iso_code):
    country = pycountry.countries.get(alpha_2=iso_code.upper())
    return country.name if country else None


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


def get_eurostats_turnover(country: str = None):
    to1 = pd.read_excel("Data/EU Manufacturing Turnover 1992-2002.xlsx", sheet_name='Sheet 1', header=8)
    to2 = pd.read_excel("Data/EU Manufacturing Turnover 2003 - 2004.xlsx", sheet_name='Sheet 1', header=8)
    to3 = pd.read_excel("Data/EU Manufacturing Turnover 2005-2020.xlsx", sheet_name='Sheet 1', header=8)
    to4= pd.read_excel("Data/EU Net Turnover 2021-2024.xlsx", sheet_name='Sheet 1', header=9)

    data = pd.DataFrame(columns=['country'])
    numcols = []

    for df, yrange in [(to1, range(1992,2003)), (to2, range(2003,2005)), (to3, range(2005,2021)), (to4, range(2021,2025))]:
        cols = [str(year) for year in yrange]
        numcols = numcols + cols
        df['country'] = df.iloc[:,0]
        df = df[['country'] + cols]

        data = data.merge(df, on='country', how='outer')

    #melt frame
    data = data.melt('country', value_vars=numcols, value_name='turnover', var_name='year')

    if country:
        data = data[data['country'] == country]

    return data.sort_values('year', ascending = True).reset_index(drop=True)


import duckdb
import pandas as pd

def read_parquet(country_iso: str) -> pd.DataFrame:
    #financials = "/Users/curdinlieberherr/Documents/Schule/HSG/Semester/12.FS26/thesis/industry_global_financials_and_ratios_eur"
    financials = "/Users/curdinlieberherr/Documents/Schule/HSG/Semester/12.FS26/thesis/financials_history_quarterly_industry_global_financials_and_ratios_eur"
    addresses = "/Users/curdinlieberherr/Documents/Schule/HSG/Semester/12.FS26/thesis/all_addresses"
    sectors = "/Users/curdinlieberherr/Documents/Schule/HSG/Semester/12.FS26/thesis/industry_classifications"

    sql = f"""
    WITH addr_filtered AS (
        SELECT DISTINCT bvd_id_number
        FROM read_parquet('{addresses}/*.parquet')
        WHERE country_iso_code = '{country_iso}'
    ),
    sector_filtered AS (
        SELECT DISTINCT bvd_id_number, 
            REPLACE(nace_rev_2_core_code_4_digits_, '.', '') AS nace_code
        FROM read_parquet('{sectors}/*.parquet')
        WHERE TRY_CAST(nace_rev_2_core_code_4_digits_ AS DOUBLE) BETWEEN 10.0 AND 33.0
    ),
    filtered_ids AS (
        SELECT a.bvd_id_number, s.nace_code
        FROM addr_filtered AS a
        JOIN sector_filtered AS s USING (bvd_id_number)
    ),
    fin_filtered AS (
        SELECT fin.*, ids.nace_code,
            ROW_NUMBER() OVER (
                PARTITION BY fin.bvd_id_number, YEAR(closing_date)
                ORDER BY CASE consolidation_code
                    WHEN 'U2' THEN 1
                    WHEN 'U1' THEN 2
                    WHEN 'C2' THEN 3
                    WHEN 'C1' THEN 4
                    WHEN 'LF' THEN 5
                    ELSE 6
                END
            ) AS rn
        FROM read_parquet('{financials}/*.parquet') AS fin
        JOIN filtered_ids ids USING (bvd_id_number)
        WHERE YEAR(closing_date) >= 2015 AND MONTH(closing_date) = 12 AND DAY(closing_date) = 31
    )
    SELECT * EXCLUDE (rn)
    FROM fin_filtered
    WHERE rn = 1
    """

    df = duckdb.query(sql).df()

    df['year'] = df['closing_date'].dt.year.astype(str)

    namedict = {
    'bvd_id_number' : 'FirmName',
    'nace_code':   'sector',
    'current_assets' : 'assets',
    'sales' : 'revenue',
    'material_costs': 'materials',
    'number_of_employees': 'nEmployees',
    'costs_of_employees': 'wagebill'
    }

    df = df.rename(columns=namedict)

    #turn values in to tsd
    for col in ['assets', 'revenue', 'materials', 'wagebill']:
        df[col] = df[col] / 1000

    return df[['FirmName', 'assets', 'revenue', 'wagebill', 'nEmployees', 'materials', 'sector', 'year']]