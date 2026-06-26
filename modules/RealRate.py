import pandas as pd
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
