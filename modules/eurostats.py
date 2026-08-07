import pandas as pd
import warnings
warnings.filterwarnings('ignore')



def get_eurostats_data(country: str = None):

    files = [("Data/Eurostat Comparison 1992 - 2002.xlsx", (1992,2002), 8,
                {'Sheet 1': 'nfirms', 'Sheet 2': 'turnover', 'Sheet 3': 'valueadded', 'Sheet 5': 'wages', 'Sheet 6': 'nemployees'}),
            ("Data/Eurostats Comparison 2002-2004.xlsx", (2002,2004+1),8,
                {'Sheet 1': 'nfirms', 'Sheet 2': 'turnover', 'Sheet 4': 'valueadded', 'Sheet 5': 'wages', 'Sheet 6': 'nemployees'}),
            ("Data/Eurostat Comparison 2005-2020.xlsx", (2005,2020+1),8,
                {'Sheet 1': 'nfirms', 'Sheet 2': 'turnover', 'Sheet 3': 'valueadded', 'Sheet 5': 'wages', 'Sheet 6': 'nemployees'}),
            ("Data/Eurostat Comparison 2021-2024.xlsx", (2021,2024+1),9,
                {'Sheet 1': 'nfirms', 'Sheet 2': 'nemployees', 'Sheet 3': 'valueadded', 'Sheet 4': 'wages', 'Sheet 5': 'turnover'})]

    data = pd.DataFrame()

    for file, yrange, header, mapdict in files:
        btdf = pd.DataFrame(columns = ['country', 'year'])
        for sheet, var in mapdict.items():
            df = pd.read_excel(file, sheet_name = sheet, header=header)
            cols = [str(year) for year in range(*yrange)]
            df['country'] = df.iloc[:,0]
            df = df[['country'] + cols]
            df = df.melt('country', value_vars=cols, value_name=var, var_name='year')
            unit = (1 if var in ['nfirms', 'nemployees'] else 1e6)
            df[var] = pd.to_numeric(df[var], errors = 'coerce') * unit

            btdf = btdf.merge(df, on=['country', 'year'], how = 'outer')
        
        data = pd.concat([data, btdf])

    if country:
        data = data[data['country'] == country]

    return data.sort_values('year', ascending = True).reset_index(drop=True)