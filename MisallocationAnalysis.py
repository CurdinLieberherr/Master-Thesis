import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pycountry
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Literal
from modules import eurostats, orbis_parquet, prices, RealRate, WithinFirmMoments

NACE_REV_2_SECTORS = {
    "A": {"description": "Agriculture, forestry and fishing", "range": (1, 3)},
    "B": {"description": "Mining and quarrying", "range": (5, 9)},
    "C": {"description": "Manufacturing", "range": (10, 33)},
    "D": {"description": "Electricity, gas, steam and air conditioning supply", "range": (35, 35)},
    "E": {"description": "Water supply; sewerage, waste management and remediation activities", "range": (36, 39)},
    "F": {"description": "Construction", "range": (41, 43)},
    "G": {"description": "Wholesale and retail trade; repair of motor vehicles and motorcycles", "range": (45, 47)},
    "H": {"description": "Transportation and storage", "range": (49, 53)},
    "I": {"description": "Accommodation and food service activities", "range": (55, 56)},
    "J": {"description": "Information and communication", "range": (58, 63)},
    "K": {"description": "Financial and insurance activities", "range": (64, 66)},
    "L": {"description": "Real estate activities", "range": (68, 68)},
    "M": {"description": "Professional, scientific and technical activities", "range": (69, 75)},
    "N": {"description": "Administrative and support service activities", "range": (77, 82)},
    "O": {"description": "Public administration and defence; compulsory social security", "range": (84, 84)},
    "P": {"description": "Education", "range": (85, 85)},
    "Q": {"description": "Human health and social work activities", "range": (86, 88)},
    "R": {"description": "Arts, entertainment and recreation", "range": (90, 93)},
    "S": {"description": "Other service activities", "range": (94, 96)},
    "T": {"description": "Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use", "range": (97, 98)},
    "U": {"description": "Activities of extraterritorial organisations and bodies", "range": (99, 99)},
    "X": {"description": "All sectors", "range": (1,99)}
}

NaceSection = Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U"]

class MisallocationAnalysis():
    def __init__(self, country: str, start:int = 2015, end:int=2024, sector:NaceSection = "C"):
        self.start = start
        self.end = end
        self.country = country
        country_iso = get_country_iso(country)
        self.selected_sector = sector
        sectormin, sectormax = NACE_REV_2_SECTORS[sector]['range'] 
        self.fin = orbis_parquet.read_from_partitioned_file(country_iso, start=start, end=end, sectormin=sectormin, sectormax=sectormax)
        self.eurostats = eurostats.get_eurostats_data(self.country)
        self.prices = prices.price_indexes(self.country)
        self.capi = prices.capital_prices(self.country)
        self.realrate = RealRate.RealRate(self.country)
        self.df = self._main_df()
        self.win_df = self._win_df()
        self.sector_weights = self._calculate_sector_weights()
        self.time_invariant_weights = self._calculate_sector_weights(True)
        self.dispt, self.disp = self._calculate_dispersion()
        self.dispt_sector = self._dispersion_per_sector()
        self.tfpdf = self._estimate_productivity()
        self.cap_moments = self._capital_moments()
        self.estimate_all_moments()

    def _main_df(self) -> pd.DataFrame:

         #rename the data
        namedict = {
            'bvd_id_number' : 'FirmName',
            'nace_code':   'sector',
            'current_assets' : 'assets',
            'sales' : 'revenue',
            'material_costs': 'materials',
            'number_of_employees': 'nEmployees',
            'costs_of_employees': 'wagebill',
            'long_term_debt': 'debt',
            'year': 'year'
        }

        df = self.fin.copy()

        #clean liabilites i assume that at leas one info about liabilites have to be in the data
        cols = ['non_current_liabilities', 'other_non_current_liabilities', 
        'current_liabilities', 'other_current_liabilities']
        if all(col in df.columns for col in cols + ['total_assets']):
            df = df.dropna(subset=cols, how='all')
            df['liabilites'] = df[cols].sum(axis=1)
            df['netfirmvalue'] = df['total_assets'] - df['liabilites']
            namedict['netfirmvalue'] = 'netfirmvalue'
            namedict['long_term_debt'] = 'debt'

        df['year'] = df['closing_date'].dt.year.astype(str)
        df = df.rename(columns=namedict)
        df = df[namedict.values()]

        #merge priceindexes 
        df['sector2d'] = df['sector'].astype(str).str[:2]
        df['year'] = df['year'].astype(str)
        df = df.merge(self.prices, on=['sector2d', 'year'], how='left')
        df = df.merge(self.capi, on='year', how='left')
        df['year'] = df['year'].astype(int)
        df = df.drop('sector2d', axis=1)

        #drop empty price index
        df = df[df['priceind'].notna()]
        df = df[df['capitalprice'].notna()]


        #deflate assets and wagebill to get k and l with pricindex
        df['k'] = df['assets'] / df['capitalprice']   
        df['nvad'] = df['revenue'] - df['materials']

        #deflate values
        df['y'] = df['nvad'] / df['priceind']
        df['l'] = df['wagebill'] / df['priceind']
        df['w'] = (df['wagebill'] / df['nEmployees']) / df['priceind']
        df['a'] = df['netfirmvalue'] / df['priceind']
        df['b'] = df['debt'] / df['capitalprice']
      
        #drop non positive values and prices
        df = df[df['k'] > 0]
        df = df[df['w'] > 0]
        df = df[df['nvad'] > 0]
        df = df[df['revenue'] > 0]
        df = df[df['materials'] > 0]
        df = df[(df['debt'] >= 0) | (df['debt'].isna())]
        df = df[df['netfirmvalue'] > 0]
        df = df[df['a'] > 0]
        df[df['b'] >= 0]

        #add real interest rate
        df = df.merge(self.realrate.df[['year', 'realinterestrate']], on='year', how='left')

        return df

    def _win_df(self) -> pd.DataFrame:
        df = self.df.copy()
        vars = ['k', 'w', 'nvad', 'revenue', 'materials', 'debt', 'netfirmvalue', 'wagebill', 'a', 'b']
        for var in vars:
            df[var] = winsorise(df[var])

        return df

    #calculate revenue distribution per firm size
    def revenue_per_firm_size(self):
        df = self.df.copy()
        df['size_category'] = np.where(
            df['nEmployees'] < 20, '0-19 employees', pd.NA
        )
        df['size_category'] = np.where(
                    (df['nEmployees'] > 19) & (df['nEmployees'] < 250), '20-249 employees', df['size_category']
        )
        df['size_category'] = np.where(
                            df['nEmployees'] > 250, '250+ employees', df['size_category']
        )
        grouped = df.groupby('size_category')['revenue'].sum()
        grouped = grouped / df['revenue'].sum()
        return grouped.round
    
    def _calculate_sector_weights(self, time_invariant = False):
        # total VA per sector per year
        vast = self.df.groupby(['sector', 'year'])['nvad'].sum().reset_index()

        # total VA across all sectors, per year
        vat = self.df.groupby('year')['nvad'].sum().reset_index().rename(columns={'nvad': 'nvadt'})

        # merge total onto sector-level data
        vast = pd.merge(vast, vat, how='left', on='year')

        # weight = sector's VA share of that year's total VA
        vast['sectorweight'] = vast['nvad'] / vast['nvadt']

        # sanity check: weights must sum to 1 within EACH year
        year_sums = vast.groupby('year')['sectorweight'].sum()
        if not np.allclose(year_sums, 1):
            bad_years = year_sums[~np.isclose(year_sums, 1)]
            raise Exception(f'Weights do not sum to 1 for these years:\n{bad_years}')

        if time_invariant:
            # --- build time-invariant sector weights ---
            # average each sector's weight across all years to get a single fixed weight per sector
            time_invariant_weights = (
                vast
                .groupby('sector')['sectorweight']
                .mean()
                .reset_index()
            )
            return time_invariant_weights

        return vast[['sector', 'year', 'sectorweight']]

    def compare_df_eurostats(self) -> pd.DataFrame:
        #get infos of companies per year
        counts = self.df.groupby('year').agg({
            'FirmName': 'count', 'revenue': 'sum', 'wagebill': 'sum', 'materials': 'sum', 'nEmployees': 'sum'}).reset_index()
        counts['year'] = counts['year'].astype(str)

        #merge with eurostats turnover and compare
        counts = counts.merge(self.eurostats, how='left', on='year')

        counts['Turnover'] = (counts['revenue'] / counts['turnover'])
        counts['Wages'] = (counts['wagebill'] / counts['wages'])
        counts['Value Added'] = (counts['revenue'] - counts['materials']) / counts['valueadded']
        counts['Firms'] = counts['FirmName'] / counts['nfirms']
        counts['Employees'] = counts['nEmployees'] / counts['nemployees']
        counts['Year'] = counts['year']

        return counts[['Year', 'Turnover','Wages', 'Value Added', 'Firms', 'Employees']].round(2)
    
    def _calculate_dispersion(self, df:pd.DataFrame=None):
        alpha = 0.35
        markup = 1

        if df is None:
            df = self.win_df

        # see gopinath p. 1926
        #calculate mrpk
        df['MRPK'] = (alpha/markup) * (df['nvad']/ df['k'])
        #calculate mrpl
        df['MRPL'] = ((1-alpha)/markup)* (df['nvad']/ df['l'])
        #calculate firms total factor productivity
        df['TFPR'] = df['revenue'] / ( (df['k']** alpha) * (df['l']**(1-alpha)) )


        #calcualte dispersion from log MRPK and MRPL
        df['log_MRPK'] = np.log(df['MRPK'])
        df['log_MRPL'] = np.log(df['MRPL'])
        df['log_TFPR'] = np.log(df['TFPR'])

        #group by sector, year and take std
        disp = df.groupby(['sector', 'year']).agg(
                            disp_MRPK=('log_MRPK', 'std'),
                            disp_MRPL=('log_MRPL', 'std'),
                            disp_TFPR=('log_TFPR', 'std')).reset_index()

        #merge weights with dispersion on sector and sum
        disp = pd.merge(disp, self.time_invariant_weights, on='sector', how='left')
        disp['w_disp_MRPK'] = disp['disp_MRPK'] * disp['sectorweight']
        disp['w_disp_MRPL'] = disp['disp_MRPL'] * disp['sectorweight']
        disp['w_disp_TFPR'] = disp['disp_TFPR'] * disp['sectorweight']
        #get dispersion weighted on sectors per year
        dispt = disp.groupby('year').agg({'w_disp_MRPK': 'sum', 'w_disp_MRPL': 'sum', 'w_disp_TFPR': 'sum'})
        dispt = dispt.sort_index(ascending=True)
        
        return dispt, disp
    
    def _estimate_disp_large_firms(self, top:float = 0.05):
        df = self.win_df.copy()

        df['top5pct'] = df.groupby(['sector','year'])['k'].transform(
        lambda x: x >= x.quantile(1-top)
        )
        df = df[df['top5pct']]

        #group by sector, year and take std
        disp = df.groupby(['sector', 'year']).agg(
                            disp_MRPK=('log_MRPK', 'std'),
                            disp_MRPL=('log_MRPL', 'std'),
                            disp_TFPR=('log_TFPR', 'std')).reset_index()

        #merge weights with dispersion on sector and sum
        disp = pd.merge(disp, self.time_invariant_weights, on='sector', how='left')
        disp['w_disp_MRPK'] = disp['disp_MRPK'] * disp['sectorweight']
        disp['w_disp_MRPL'] = disp['disp_MRPL'] * disp['sectorweight']
        disp['w_disp_TFPR'] = disp['disp_TFPR'] * disp['sectorweight']
        #get dispersion weighted on sectors per year
        dispt = disp.groupby('year').agg({'w_disp_MRPK': 'sum', 'w_disp_MRPL': 'sum', 'w_disp_TFPR': 'sum'})
        dispt = dispt.sort_index(ascending=True)
        
        return dispt

    def _dispersion_per_sector(self):
        df = self.disp.copy()
        df['sector'] = df['sector'].str[:2]
        grouped = df.groupby(['sector', 'year'])['w_disp_MRPK'].sum().reset_index().sort_values(['sector', 'year'],ascending=True)

        grouped['w_disp_rel'] = grouped.groupby('sector')['w_disp_MRPK'].transform(
            lambda x: x / x.iloc[0]
        )

        sector_variation = (
            grouped.groupby('sector')['w_disp_rel']
            .agg(
                std='var',
                abs_change=lambda x: x.iloc[-1] - x.iloc[0]
            )
            .sort_values('std', ascending=False).reset_index()
        )

        #merge sectorname
        names = get_sector_description()
        sector_variation = sector_variation.merge(names, left_on='sector', right_on='Codes')

        return sector_variation
    
    def plot_top_dispersion_sector(self, n:int=5):
        df = self.disp.copy()
        df['sector'] = df['sector'].str[:2]
        grouped = df.groupby(['sector', 'year'])['w_disp_MRPK'].sum().reset_index().sort_values(['sector', 'year'],ascending=True)

        grouped['w_disp_rel'] = grouped.groupby('sector')['w_disp_MRPK'].transform(
            lambda x: x / x.iloc[0]
        )

        top = self.dispt_sector['sector'][:n].to_list()
        grouped = grouped[grouped['sector'].isin(top)]

        names = get_sector_description()
        grouped = grouped.merge(names, left_on='sector', right_on='Codes')

        fig, ax = plt.subplots()
        for sector, group in grouped.groupby('Labels'):
            ax.plot(group['year'], group['w_disp_rel'], label=sector)

        ax.legend()
        plt.show()

        return fig


    def plot_main_sectors_dispersion(self, n:int=5):
        if self.selected_sector != 'X':
            raise Exception('For this plot you need to select sector = X when initializing misallocation analysis.')
        df = self.disp.copy()
        sector_code_lookup = build_code_lookup(NACE_REV_2_SECTORS)
        df['main_sector'] = df['sector'].str[:2].astype(int).map(sector_code_lookup)
        grouped = df.groupby(['main_sector', 'year'])['w_disp_MRPK'].sum().reset_index().sort_values(['main_sector', 'year'],ascending=True)
        grouped['w_disp_rel'] = grouped.groupby('main_sector')['w_disp_MRPK'].transform(
            lambda x: x / x.iloc[0]
        )
        
        sector_names = {key: info["description"] for key, info in NACE_REV_2_SECTORS.items()}
        grouped['sector_name'] = grouped['main_sector'].map(sector_names)

        fig, ax = plt.subplots()
        for sector, group in grouped.groupby('sector_name'):
            ax.plot(group['year'], group['w_disp_rel'], label=f"{group['main_sector'].min()}-{sector}")

        ax.legend()
        plt.show()

        return fig

    def plot_dispersion(self, variables: Literal['MRPK', 'MRPL', 'both'] = 'MRPK', figsize=(8, 6), show=True):
        plotdf = self.dispt.copy()
        plotdf = plotdf / plotdf.iloc[0] -1
        plotdf = plotdf.reset_index()

        # create plot
        fig, ax1 = plt.subplots(figsize=figsize)

        if variables in ['MRPK', 'both']:
            ax1.plot(plotdf["year"], plotdf["w_disp_MRPK"], label="MRPK", color="#1f77b4", linewidth=2)
        if variables in ['MRPL', 'both']:
            ax1.plot(plotdf["year"], plotdf["w_disp_MRPL"], label="MRPL", color="#d62728", linewidth=2)

        # dynamic title based on which variable(s) are plotted
        title_map = {
            'MRPK': 'log MRPK Dispersion',
            'MRPL': 'log MRPL Dispersion',
            'both': 'log MRPK and log MRPL Dispersion'
        }
        ax1.set_title(f"{self.country} - {title_map[variables]}")

        ax1.set_xlabel("Year")
        ax1.set_ylabel(f"Disp {title_map[variables]} (Growth, {plotdf['year'][0]} = 0)")
        ax1.legend()

        plt.tight_layout()
        if show:
            plt.show()

        return fig

    def plot_dispersion_productivity(self, variables: Literal['MRPK', 'MRPL', 'both'] = 'MRPK', figsize=(6, 4)):
        plotdf = self.tfpdf.copy()
        plotdf = plotdf / plotdf.iloc[0] -1
        plotdf = plotdf.reset_index()

        fig = self.plot_dispersion(variables=variables, figsize=figsize, show=False)
        ax = fig.axes[0]  # grab the first (or only) axes in the figure
        ax.plot(plotdf["year"], plotdf["log_tfp"], label="TFP", color="green", linewidth=2)
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--', dashes=(5, 10), alpha = 0.5)
        ax.legend()
        current_title = ax.get_title()
        ax.set_title(f"{current_title} and log TFP Growth")
        ax.set_ylabel(f'Growth, {plotdf['year'][0]} = 0')
        plt.show()

        return fig

    
    def plot_histogram_firm_size(self, figsize = (8,6)):
        df = self.df[['year', 'k']].copy()

        years = sorted(df['year'].unique())
        colors = cm.viridis(np.linspace(0, 1, len(years)))  # or plt.cm.plasma, cm.turbo, cm.coolwarm

        fig, ax = plt.subplots(figsize=figsize)
        for year, color in zip(years, colors):
            subset = df.loc[df['year'] == year, 'k']
            ax.hist(np.log(subset), bins=40, alpha=0.5, label=str(year), density=True, color=color)

        ax.set_xlabel(r'$\log(k)$')
        ax.legend()
        plt.show()

        return fig


    def plot_dispersion_small_large_firms(self, top = 0.05, figsize = (8,6)):
        all = self.dispt.copy()
        all = all.reset_index()
        all['change'] = all['w_disp_MRPK'] / all.loc[0, 'w_disp_MRPK']

        large = self._estimate_disp_large_firms(top)
        large = large.reset_index()
        large['change'] = large['w_disp_MRPK'] / large.loc[0, 'w_disp_MRPK']

        # create plot
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()  # second y-axis sharing the same x-axis

        # absolute levels on ax1 (solid lines)
        l1, = ax1.plot(all["year"], all["w_disp_MRPK"], label="all firms (level)",
                    color="#1f77b4", linewidth=2, linestyle='-')
        l2, = ax1.plot(large["year"], large["w_disp_MRPK"], label="large firms (level)",
                    color="#d62728", linewidth=2, linestyle='-')

        # normalized change on ax2 (dashed lines)
        l3, = ax2.plot(all["year"], all["change"], label="all firms (change)",
                    color="#1f77b4", linewidth=2, linestyle='--', alpha=0.7)
        l4, = ax2.plot(large["year"], large["change"], label="large firms (change)",
                    color="#d62728", linewidth=2, linestyle='--', alpha=0.7)

        ax1.set_xlabel("Year")
        ax1.set_ylabel("SD log MRPK (level)")
        ax2.set_ylabel(f"SD log MRPK (change, {all['year'][0]} = 1)")
        ax1.set_title(f"{self.country} - Dispersion of MRPK Small vs. Large Firms ({top})")

        # combine legends from both axes into one
        lines = [l1, l2, l3, l4]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='best')

        plt.tight_layout()
        plt.show()

        return fig



    def _estimate_productivity(self):
        #calculate firm productivity z
        EPSILON = 3
        alpha = 0.35

        #calculate sectorrevenue and merge to df
        sectorrevenue = self.win_df.groupby(['year', 'sector']).agg(nvad_s = ('nvad', 'sum')).reset_index()
        self.win_df = self.win_df.merge(sectorrevenue, on=['year', 'sector'], how='left')

        #calculate firm productivity
        self.win_df['Z'] = (
            (self.win_df['nvad_s']**(-1/(EPSILON-1)) / self.win_df['priceind'])
        ) * (
            (self.win_df['nvad']**(EPSILON/(EPSILON-1))) 
            / ((self.win_df['k']**alpha) * (self.win_df['l']**(1-alpha)))
        )
        self.win_df['Z_pow'] = self.win_df['Z'] ** (EPSILON - 1)

    
        # df groupby sector and year
        sdf = self.win_df.groupby(['year', 'sector'])

        # calculate log tfpe and tfp per year and sector by the formulas above
        log_tfpe_st = ((1/(EPSILON-1)) 
                    * (np.log(sdf['FirmName'].count()) + np.log(sdf['Z_pow'].mean()))
                    )
        log_tfp_st = np.log(sdf['y'].sum()) - (alpha * np.log(sdf['k'].sum())) - ((1-alpha) * np.log(sdf['l'].sum()))

        tfp_st_df = pd.concat([log_tfpe_st, log_tfp_st], axis=1)
        tfp_st_df.columns = ['log_tfpe_st', 'log_tfp_st']
        tfp_st_df = tfp_st_df.reset_index()
        tfp_st_df = tfp_st_df.merge(self.time_invariant_weights, on='sector', how='left')
        tfp_st_df['wlogtfpe'] = tfp_st_df['log_tfpe_st'] * tfp_st_df['sectorweight']
        tfp_st_df['wlogtfp'] = tfp_st_df['log_tfp_st'] * tfp_st_df['sectorweight']

        tfp_df = tfp_st_df.groupby('year').agg(
            log_tfp=('wlogtfp', 'sum'),
            log_tfpe=('wlogtfpe', 'sum')
        )
        #add a 1% growth tfp
        tfp_df['log_tfpg'] = tfp_df['log_tfp'].iloc[0] + np.log(1.01) * np.arange(len(tfp_df))

        return tfp_df
        

    def plot_productivity(self, figsize = (8,6)):
        plotdf = self.tfpdf.copy()
        plotdf = plotdf / plotdf.iloc[0] - 1
        plotdf = plotdf.reset_index()

        #create plot 
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=figsize)

        ax1.plot(plotdf["year"], plotdf["log_tfp"], label="TFP", color="#1f77b4", linewidth=2)
        ax1.plot(plotdf["year"], plotdf["log_tfpe"], label="TFPe", color="#d62728", linewidth=2)
        ax1.plot(plotdf["year"], plotdf["log_tfpg"], label="1% growth", color="black", linewidth=2)
        ax1.set_title(f"TFP vs. TFPe vs. 1% growth. {plotdf['year'][0]} = 0")
        ax1.set_ylabel("log TFP growth")
        ax1.legend()

        ax2.plot(plotdf['year'],  plotdf["log_tfp"] -  plotdf["log_tfpe"], label='TFP vs TFPe')
        ax2.set_title(f"{self.country} - Evolution of TFP Growth relative to Efficient Level Growth")
        ax2.set_ylabel('log(TFP) - log(TFPe)')
        ax2.legend()


        plt.tight_layout()
        plt.show()

        return fig

    def plot_mrpk_tfp_realrate(self, figsize=(6, 4)):
        # get dispersion dataframe
        plotdf = self.dispt.copy()
        plotdf = plotdf.merge(self.tfpdf, left_index=True, right_index=True)
        # calculate relative change
        plotdf = (plotdf / plotdf.iloc[0]) - 1
        plotdf = plotdf.reset_index()
        # add realinterestrate
        plotdf = plotdf.merge(self.realrate.df[['year', 'realinterestrate']].drop_duplicates('year'), on='year', how='left')
        # calculate change of real interest rate
        plotdf['realinterestrate'] = np.where(plotdf['realinterestrate'].isna(), np.nan, plotdf['realinterestrate'] - plotdf['realinterestrate'].dropna().iloc[0])

        fig, ax1 = plt.subplots(figsize=figsize)

        # --- Left axis: MRPK and TFP ---
        l1, = ax1.plot(plotdf["year"], plotdf["w_disp_MRPK"], label="MRPK Dispersion", color="#1f77b4", linewidth=2)
        l2, = ax1.plot(plotdf["year"], plotdf["log_tfp"], label="TFP", color="green", linewidth=2, linestyle=":")

        ax1.axhline(y=0, color='black', linewidth=0.8, linestyle='--', dashes=(5, 10), alpha = 0.5)
        ax1.set_xlabel("Year")
        ax1.set_ylabel(f"Normalized ({plotdf['year'].iloc[0]} = 0)")

        # --- Right axis: Real Interest Rate ---
        ax2 = ax1.twinx()
        ax2.axhline(y=0, color="#ff7f0e", linewidth=0.8, linestyle='--', dashes=(5, 10), alpha = 0.5)
        l3, = ax2.plot(plotdf["year"], plotdf["realinterestrate"], label="Real Interest Rate", color="#ff7f0e", linewidth=2, linestyle="--")
        ax2.set_ylabel("Δ Real Interest Rate (pp)", color="#ff7f0e")
        ax2.tick_params(axis='y', labelcolor="#ff7f0e")

        # --- Combined legend ---
        lines = [l1, l2, l3]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="lower left")

        ax1.set_title(f"{self.country} - MRPK Dispersion vs. Real Interest Rate vs. TFP.")

        plt.tight_layout()
        plt.show()

        return fig
    
    def _capital_moments(self):
        cap = self.win_df.copy()

        cap['log_k'] = np.log(cap['k'])
        cap['log_Z'] = np.log(cap['Z'])

        # Cross-sectional correlation and std — pooled across all firms per year
        cap_year = cap.groupby('year').apply(lambda g: pd.Series({
            'corr_Z_k': g['log_k'].corr(g['log_Z']),
            'std_k':    g['log_k'].std(),
            'std_Z':    g['log_Z'].std(),
        })).reset_index()

        return cap_year

    
    def plot_capital_tfp_moments(self, figsize = (10,4)):
        
        cap = self.cap_moments

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        ax1.plot(cap['year'], cap['std_k'])
        ax1.set_ylabel('Standard Deviation of log(k)')
        ax2.plot(cap['year'], cap['corr_Z_k'])
        ax2.set_ylabel('Correlation of log(k) with log(Z)')

        plt.tight_layout()
        plt.show()

        return fig
    
    def plot_capital_wedges(self, figsize = (10,4)):
        DELTA = 0.1
        df = self.win_df.copy()
        df['log_tau_k'] = np.log(df['w']) + df['log_MRPK'] - np.log(df['realinterestrate'] + DELTA) - df['log_MRPL']

        # ── Assign groups: bottom 50% vs top 10% ────────────────────────
        def assign_group(x):
            p50 = x.quantile(0.50)
            p90 = x.quantile(0.90)
            groups = pd.Series('middle', index=x.index)
            groups[x <= p50] = 'Bottom 50%'
            groups[x >= p90] = 'Top 10%'
            return groups

        df['size_group'] = df.groupby('year')['k'].transform(assign_group)

        # ── Group means ──────────────────────────────────────────────────
        group_tau = (df[df['size_group'] != 'middle']
                    .groupby(['year', 'size_group'])['log_tau_k']
                    .mean()
                    .unstack())
        group_tau['gap'] = group_tau['Bottom 50%'] - group_tau['Top 10%']

        # ── Plots ────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: log tau_k by group
        for grp, color in zip(['Bottom 50%', 'Top 10%'], ['steelblue', 'firebrick']):
            axes[0].plot(group_tau.index, group_tau[grp], marker='o', label=grp, color=color)

        axes[0].set_title('Log $\\tau_k$: Bottom 50\\% vs Top 10\\%')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Mean log $\\tau_k$')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # Plot 2: Gap
        axes[1].plot(group_tau.index, group_tau['gap'], marker='o', color='darkorange')
        axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[1].set_title('Gap in log $\\tau_k$: Bottom 50\\% $-$ Top 10\\%')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Gap')
        axes[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

        return fig
    
    def estimate_distributional_moments(self):
        vars = ['Z_pow', 'k', 'l']
        df = self.win_df.copy()[['sector', 'year'] + vars]
        df[vars] = np.log(df[vars])
        for var in vars:
            df[var] = winsorise(df[var])
        df = df.groupby('sector')[vars].agg('std').reset_index()
        df = pd.merge(df, self.time_invariant_weights, on='sector', how='left')
        for var in vars:
            df[var] = df[var] * df['sectorweight']
        distmoments = df[vars].sum().round(2)
        distmoments.index = [f'Std.dev.(log {i})' for i in vars]

        def top20_share(group, cols):
            result = {}
            for col in cols:
                threshold = group[col].quantile(0.80)
                top20_sum = group.loc[group[col] >= threshold, col].sum()
                result[col] = top20_sum / group[col].sum()
            return pd.Series(result)

        cols = ['l', 'k']
        result = self.win_df.copy().groupby('year').apply(lambda g: top20_share(g, cols))[cols].mean().round(2)
        result.index = [f'Top 20% Share of {var}' for var in cols]

        self.distributional_moments = pd.concat([distmoments,result], axis=0)

    def estimate_within_firm_moments(self):
        self.within_firm_moments = WithinFirmMoments.WithinFirmMoments(self.win_df)

    def estimate_cross_sectional_moments(self):
        df = self.win_df.copy()

        #calculate borrower share
        borrower_share = (df['b'] > 0).mean()

        #calculate correlations
        vars = ['Z_pow', 'k', 'a', 'MRPK']
        for var in vars:
            df[var] = np.log(df[var])

        results = pd.Series([
            borrower_share,
            df['Z_pow'].corr(df['k']),
            df['Z_pow'].corr(df['a']),
            df['MRPK'].corr(df['Z_pow']),
            df['MRPK'].corr(df['k']),
            df['MRPK'].corr(df['a'])
        ], index=['Fraction borrowing',
                'Corr (log Z, log k)',
                'Corr (log Z, log a)',
                'Corr (log MRPK, log Z)',
                'Corr (log MRPK, log k)',
                'Corr (log MRPK, log a)']).round(2)
        
        self.cross_sectional_moments = results

    def estimate_all_moments(self):
        self.estimate_distributional_moments()
        self.estimate_within_firm_moments()
        self.estimate_cross_sectional_moments()

        all = pd.concat([self.distributional_moments,
                        self.within_firm_moments.coefs['parameter'],
                        self.cross_sectional_moments])

        self.all_moments = all

        return all
    
    def estimate_corrected_measures(self):
        df = self.win_df.copy()
        print(len(df))
        df = df.sort_values(['FirmName', 'year'])

        #create total cost for weigh
        df['total_cost'] = df['k'] + df['w'] + df['materials']
        df['cost_weight'] = df['total_cost'] / df.groupby(['sector','year'])['total_cost'].transform('sum')

        # share weights - use empirical cost shares
        df['s_K'] = df['k'] / df['total_cost']
        df['s_L'] = df['w'] / df['total_cost']
        df['s_X'] = df['materials'] / df['total_cost']

        # calculate cost weightes sector year average TFPR
        df['wlnTFPR'] = df['cost_weight'] * df['log_TFPR']
        df['meanlnTFPR'] = df.groupby(['sector', 'year'])['wlnTFPR'].transform('sum')
        df['TFPRdev'] = df['log_TFPR'] - df['meanlnTFPR']

        #create logs of inputs and lags
        df['lnK'] = np.log(df['k'])
        df['lnL'] = np.log(df['w'])
        df['lnX'] = np.log(df['materials'])
        df['lnR'] = np.log(df['revenue'])

        #create lags and consecutive logic
        df['lnKlag'] = df.groupby('FirmName')['lnK'].shift(1) 
        df['lnLlag'] = df.groupby('FirmName')['lnL'].shift(1) 
        df['lnXlag'] = df.groupby('FirmName')['lnX'].shift(1) 
        df['lnRlag'] = df.groupby('FirmName')['lnR'].shift(1)

        df['TFPRdevlag'] = df.groupby('FirmName')['TFPRdev'].shift(1) 
        df['yearlag'] = df.groupby('FirmName')['year'].shift(1)
        df['consecutive'] = (df['year'] - df['yearlag']) == 1


        #calculate growth and average mrpk deviation
        for var in ['lnK', 'lnL', 'lnX', 'lnR']:
            df[f'd{var}'] = np.where(
                df['consecutive'],
                df[var] - df[f'{var}lag'],
                np.nan
            )

        df['TFPRmeandev'] = np.where(
            df['consecutive'],
            (df['TFPRdev'] + df['TFPRdevlag']) / 2,
            np.nan
        )

        df['decile'] = pd.qcut(df['TFPRmeandev'], 10, labels=False) + 1

        #sum cost by weights
        df['dlnI'] = df['s_K']*df['dlnK'] + df['s_L']*df['dlnL'] + df['s_X']*df['dlnX']

        # demean growth rates by sector-year (removes common sector-year shocks)
        df['dR dev'] = df['dlnR'] - df.groupby(['sector', 'year'])['dlnR'].transform('mean')
        df['dI dev'] = df['dlnI'] - df.groupby(['sector', 'year'])['dlnI'].transform('mean')

        df = df[df['consecutive']]
        print(len(df))

        # --- 6. regress dR_dev on dI_dev, separately per decile, weighted by cost share ---
        beta_by_decile = {}
        se_by_decile = {}
        reg_data = df.dropna(subset=['dR dev', 'dI dev', 'decile'])

        for k, g in reg_data.groupby('decile'):
            X = sm.add_constant(g['dI dev'])
            y = g['dR dev']
            model = sm.WLS(y, X, weights=g['cost_weight']).fit()
            beta_by_decile[k] = model.params['dI dev']
            se_by_decile[k] = model.bse['dI dev']

        results = pd.DataFrame({'beta': beta_by_decile, 'se': se_by_decile}).sort_index().reset_index().rename(columns={'index': 'decile'})

        self.decile_betas = results

        ## from betas calculate corrected values
        df['decile_weight'] = df['total_cost'] / df.groupby(['decile','year'])['total_cost'].transform('sum')
        df['decile_weighted_tfpr'] = df['TFPR'] * df['decile_weight']

        #group deciles to get total ln tfpr per decile
        decile_means = df.groupby('decile').agg(
            ln_tfpr_k=('decile_weighted_tfpr', 'sum')
        ).reset_index()
        decile_means['ln_tfpr_k'] = np.log(decile_means['ln_tfpr_k'])
        #merge with results to get beta per decile
        decile_means = decile_means.merge(results, on='decile', how='left')
        decile_means['lnbeta'] = np.log(decile_means['beta'])
        #calculate covariance term and variance term of equation 49 in klenow 2021 to get sigma
        cov_term = np.cov(decile_means['ln_tfpr_k'], decile_means['lnbeta'])[0, 1]
        var_beta = decile_means['lnbeta'].var()
        sigma2 = -cov_term - var_beta
        sigma = np.sqrt(max(sigma2, 0))
        #get epsilon for whole df from sigma
        np.random.seed(0)
        df['epsilon'] = np.random.normal(loc=0.0, scale=sigma, size=len(df))
        #calculate corrected values
        df = df.merge(decile_means, on='decile', how='left')
        df['corrected MRPK'] = df['log_MRPK'] + df['lnbeta'] + df['epsilon']
        df['corrected MRPL'] = df['log_MRPL'] + df['lnbeta'] + df['epsilon']

        self.corrected_measures = df[['FirmName', 'year', 'corrected MRPK', 'corrected MRPL']]

    def plot_correct_measures(self, measure: Literal['MRPK', 'MRPL']):
        df = self.win_df.merge(self.corrected_measures, on=['FirmName', 'year'], how='right')

        VARS = [f'log_{measure}',f'corrected {measure}']
        plotdf = df[VARS + ['nvad',  'year']].copy()
        plotdf['weight'] = plotdf['nvad'] / plotdf.groupby('year')['nvad'].transform('sum')

        for var in VARS:
            plotdf[var] = plotdf[var] * plotdf['weight']

        plotdf = plotdf.groupby('year')[VARS].sum().reset_index()

        fig,ax1 = plt.subplots(figsize=(6,4))

        for var in VARS:
            ax1.plot(plotdf["year"], plotdf[var], label=var,  linewidth=2)

        plt.legend()
        plt.show()
        return fig
    
    def _recalculate_correct_dispersion(self):
        df = self.win_df.copy()
        for var in ['log_MRPK', 'log_MRPL', 'log_TFPR']:
            df[var] = winsorise(df[var])

        #group by sector, year and take std
        disp = df.groupby(['sector', 'year']).agg(
                            disp_MRPK=('log_MRPK', 'std'),
                            disp_MRPL=('log_MRPL', 'std'),
                            disp_TFPR=('log_TFPR', 'std')).reset_index()

        #merge weights with dispersion on sector and sum
        disp = pd.merge(disp, self.time_invariant_weights, on='sector', how='left')
        disp['w_disp_MRPK'] = disp['disp_MRPK'] * disp['sectorweight']
        disp['w_disp_MRPL'] = disp['disp_MRPL'] * disp['sectorweight']
        disp['w_disp_TFPR'] = disp['disp_TFPR'] * disp['sectorweight']
        #get dispersion weighted on sectors per year
        dispt = disp.groupby('year').agg({'w_disp_MRPK': 'sum', 'w_disp_MRPL': 'sum', 'w_disp_TFPR': 'sum'})
        dispt = dispt.sort_index(ascending=True)
        
        return dispt, disp
    
    
    def inplace_correct_measures(self):
        self.win_df = self.win_df.merge(self.corrected_measures, on=['FirmName', 'year'], how='right')

        self.win_df['log_MRPK'] = self.win_df['corrected MRPK']
        self.win_df['log_MRPL'] = self.win_df['corrected MRPL']
        self.win_df['MRPK'] = np.exp(self.win_df['corrected MRPK'])
        self.win_df['MRPL'] = np.exp(self.win_df['corrected MRPL'])

        alpha = 0.35
        mu = 1

        self.win_df['log_TFPR'] = (
            np.log(mu)
            + alpha * (self.win_df['log_MRPK'] - np.log(alpha))
            + (1 - alpha) * (self.win_df['log_MRPL'] - np.log(1 - alpha))
        )
        self.win_df['TFPR'] = np.exp(self.win_df['log_TFPR'])

        self.dispt, self.disp = self._recalculate_correct_dispersion()

    def exiters_regression(self):
        df = self.win_df.copy()
        # ── Classify firms ───────────────────────────────────────────────
        last_year = df['year'].max()

        # Each firm's true last observed year
        last_obs = df.groupby('FirmName')['year'].max().rename('last_year')
        df = df.join(last_obs, on='FirmName')

        # Flag only the final observation of a firm that never comes back
        df['exiter'] = (
            (df['year'] == df['last_year']) &   # this is the firm's last row
            (df['last_year'] < last_year)        # and it exits before data ends
        ).astype(int)

        #drop the last year because there cannot be any exiters
        df = df[df['year'] < last_year]

        #make sector to 2 digit level to have more observations for the fixed effects
        df['sector2d'] = df['sector'].str[:2]

        # ── For each sector, characterise exiters vs survivors ───────────
        # along productivity (log_Z) and net worth (log_a)
        df['log_a'] = np.log(df['a'])
        df['log_Z'] = np.log(df['Z_pow'])


        model = smf.logit(
            'exiter ~ log_Z + log_a + C(sector2d) + C(year)',
            data=df
        ).fit(cov_type='cluster', cov_kwds={'groups': df['FirmName']}, disp=0)

        summary_df = pd.DataFrame({
            'coef': model.params,
            'std_err': model.bse,
            'p_value': model.pvalues,
        })

        return summary_df.loc[['log_Z', 'log_a']].round(3), model

    def plot_mrpk_exiters_vs_remainers(self):
        df = self.win_df.copy()

        # Each firm's true last observed year
        last_obs = df.groupby('FirmName')['year'].max().rename('last_year')
        df = df.join(last_obs, on='FirmName')

        df['exiter'] = (
            (df['year'] == df['last_year']) &   # this is the firm's last row
            (df['last_year'] < df['year'].max())        # and it exits before data ends
        ).astype(int)

        #compare mean log mrpk of exiters versus non exiters per year
        vs = df.groupby(['sector', 'exiter', 'year'])['log_MRPK'].mean().reset_index()
        vs = vs.merge(self.time_invariant_weights, on='sector', how='left')
        vs['log_MRPK'] = vs['log_MRPK']* vs['sectorweight']
        vs = vs.groupby(['exiter', 'year'])['log_MRPK'].sum().round(2).unstack(level=0)

        fig1 = vs.plot(figsize=(6, 4))
        fig1.get_figure().savefig('plots/exitersvsstayers20002012.png', dpi=300)
        plt.title('Mean log MRPK by year — exiters vs stayers')
        plt.ylabel('Mean log MRPK')
        plt.xlabel('Year')
        plt.legend()
        plt.tight_layout()
        plt.show()

        return fig1

    def plot_dispersion_all_vs_survivors(self):
        df = self.win_df.copy()

        # Each firm's true last observed year
        last_obs = df.groupby('FirmName')['year'].max().rename('last_year')
        df = df.join(last_obs, on='FirmName')

        # Flag only the final observation of a firm that never comes back
        df['exiter'] = (
            (df['year'] == df['last_year']) &   # this is the firm's last row
            (df['last_year'] < df['year'].max())        # and it exits before data ends
        ).astype(int)

        df = df[df['exiter'] == 0]

        #drop the last year because there cannot be any exiters
        df = df[df['year'] < df['year'].max()]
        #recalculate the dispersion
        remaindispt, _ = self._calculate_dispersion(df=df)
        #reuse full dipsersion
        dispt = self.dispt.copy()
        plotdf = dispt / dispt.iloc[0]
        plotdf = plotdf.reset_index()
        remaindispt = remaindispt/remaindispt.iloc[0]
        remaindispt = remaindispt.reset_index()

        #create plot 
        fig, ax1 = plt.subplots()

        ax1.plot(plotdf["year"], plotdf["w_disp_MRPK"], label="all", color="#1f77b4", linewidth=2)
        ax1.plot(remaindispt["year"], remaindispt["w_disp_MRPK"], label="remainers", color="#d62728", linewidth=2)
        ax1.set_title(f"all vs. remainers. {plotdf['year'][0]} = 1")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Standard Deviation")
        ax1.legend()

        plt.tight_layout()
        plt.show()

        return fig

    def overview_statistics(self):
        period = f"{self.start} - {self.end}"
        avg_firmcount = self.df.groupby('year')['FirmName'].count().mean()
        avg_revenue_coverage = self.compare_df_eurostats()['Turnover'].mean() 
        top20share = self.distributional_moments.loc['Top 20% Share of k']
        sector = NACE_REV_2_SECTORS.get(self.selected_sector)['description']

        return pd.DataFrame([{'Country': self.country, 'Period': period, 'Sector': sector, 'Avg. n. Firms': avg_firmcount, 'Avg. Revenue Covered': avg_revenue_coverage, 'Top 20% Share of Capital': top20share}]).round(2)

    def descriptive_statistics(self):
        AMOUNTCOLS = ['assets','revenue','materials','wagebill','debt']
        df = self.df[AMOUNTCOLS + ['nEmployees']].copy()
        for col in AMOUNTCOLS:
            df[col] = df[col] / 1000
        df = df.describe().round(2).T
        df['Country'] = self.country
        df['Period'] = f"{self.start}-{self.end}"
        df['Sector'] = NACE_REV_2_SECTORS.get(self.selected_sector)['description']
        additional = ['Country', 'Period', 'Sector']
        other_cols = [c for c in df.columns if c not in additional]
        df = df[additional + other_cols]
        return df.reset_index(names='Variables')


    

#winsorize function to drop the stupid ones
def winsorise(s, lower=0.01, upper=0.99):
    lo, hi = s.quantile([lower, upper])
    return s.clip(lo, hi)



def get_country_iso(name):
    country = pycountry.countries.search_fuzzy(name)[0]
    return country.alpha_2


def get_sector_description() -> pd.DataFrame:
    file = 'Data/EU Producer Prices Country Sector 1992 - 2025 2021=100.xlsx'
    df = pd.read_excel(file, header=10, dtype=str)[['NACE_R2 (Codes)', 'NACE_R2 (Labels)']].drop_duplicates()
    df = df.rename(columns={'NACE_R2 (Codes)': 'Codes', 'NACE_R2 (Labels)': 'Labels'})
    df['Codes'] = df['Codes'].str[1:]
    return df

def build_code_lookup(sectors_dict, exclude_keys=("X",)):
    """Expand each letter's (min, max) range into a {2-digit code: letter} dict."""
    lookup = {}
    for key, info in sectors_dict.items():
        if key in exclude_keys:
            continue  # skip aggregates like "X" that overlap everything
        low, high = info["range"]
        for code in range(low, high + 1):
            lookup[code] = key
    return lookup
