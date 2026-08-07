import pandas as pd
import warnings
warnings.filterwarnings('ignore')


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