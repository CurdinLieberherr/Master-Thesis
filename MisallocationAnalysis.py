from modules import *
class MisallocationAnalysis():
    def __init__(self, country_iso: str):
        self.country = get_country_name(country_iso)
        self.fin = read_parquet(country_iso)
        self.euto = get_eurostats_turnover(self.country)
        self.prices = price_indexes(self.country)
        self.capi = capital_prices(self.country)
        self.realrate = RealRate(self.country)
        self.df = self._main_df()
        self.sector_weights = self._calculate_sector_weights()
        self.dispt = self._calculate_dispersion()
        self.tfpdf = self._estimate_productivity()

    def _main_df(self) -> pd.DataFrame:

        df = self.fin.copy()

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

        #calculate base variables
        #calculate firm nominal value added
        df['nvad'] = df['revenue'] - df['materials']

        #deflate assets and wagebill to get k and l with pricindex
        df['k'] = df['assets'] / df['capitalprice']   
        df['y'] = df['nvad'] / df['priceind']

        if 'wagebill' in df.columns:
            df['l'] = df['wagebill'] / df['priceind']
            df['w'] = (df['wagebill'] / df['nEmployees']) / df['priceind']
        else:
            df['w'] = df['wage'] / df['priceind']
            df['b'] = df['debt'] / df['priceind']

        #drop non positive values and prices
        df = df[df['nvad'] > 0]
        df = df[df['k'] > 0]
        df = df[df['w'] > 0]

        #add real interest rate
        df = df.merge(self.realrate.df[['year', 'realinterestrate']], on='year', how='left')

        return df
    
    def _calculate_sector_weights(self):
        # calculate weights per sector and year
        vast = self.df.groupby(['sector', 'year'])['nvad'].sum().reset_index()
        vat = self.df.groupby(['year'])['nvad'].sum().reset_index().rename(columns={'nvad': 'nvadt'})
        
        # merge on years
        vast = pd.merge(vast, vat, how='left', on='year')
        
        # divide total VA per sector by total VA across all years
        sector_totals = vast.groupby('sector')['nvad'].sum()
        weights = (sector_totals / sector_totals.sum()).reset_index().rename(columns={'nvad': 'sectorweight'})

        w_sum = weights.sectorweight.sum()
        if round(w_sum, 6) != 1:
            raise Exception(f'Weights of sectors do not sum to 1! They sum to: {w_sum}')

        return weights



    def compare_df_eurostats(self) -> pd.DataFrame:
        #get infos of companies per year
        counts = self.df.groupby('year').agg({'FirmName': 'count', 'revenue': 'sum'}).reset_index()
        counts['year'] = counts['year'].astype(str)

        #merge with eurostats turnover and compare
        counts = counts.merge(self.euto, how='left', on='year')
        #set revenue to mios
        counts['revenue'] = counts['revenue'] / 1000
        #create coverage ratio
        counts['ratio'] = (counts['revenue'] / counts['turnover'])

        rename = {
            'year': 'Year',
            'revenue' : 'Revenue Orbis',
            'turnover': 'Turnover Eurostats',
            'ratio': 'Share Orbis'
        }
        counts = counts.rename(columns=rename)[rename.values()]

        return counts[['Year', 'Share Orbis']]
    
    def _calculate_dispersion(self):
        alpha = 0.35
        markup = 1

        df = self.df.copy()

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

        for var in ['log_MRPK', 'log_MRPL', 'log_TFPR']:
            df[var] = df.groupby(['sector', 'year'])[var].transform(winsorise)

        #group by sector, year and take std
        disp = df.groupby(['sector', 'year']).agg(
                            disp_MRPK=('log_MRPK', 'std'),
                            disp_MRPL=('log_MRPL', 'std'),
                            disp_TFPR=('log_TFPR', 'std')).reset_index()

        #merge weights with dispersion on sector and sum
        disp = pd.merge(disp, self.sector_weights, on='sector', how='left')
        disp['w_disp_MRPK'] = disp['disp_MRPK'] * disp['sectorweight']
        disp['w_disp_MRPL'] = disp['disp_MRPL'] * disp['sectorweight']
        disp['w_disp_TFPR'] = disp['disp_TFPR'] * disp['sectorweight']
        #get dispersion weighted on sectors per year
        dispt = disp.groupby('year').agg({'w_disp_MRPK': 'sum', 'w_disp_MRPL': 'sum', 'w_disp_TFPR': 'sum'})
        dispt = dispt.sort_index(ascending=True)
        
        return dispt
    
    def plot_dispersion(self):
        plotdf = self.dispt.copy()
        plotdf = plotdf / plotdf.iloc[0]
        plotdf = plotdf.reset_index()

        #create plot 
        fig, ax1 = plt.subplots(figsize=(8,6))

        ax1.plot(plotdf["year"], plotdf["w_disp_MRPK"], label="MRPK", color="#1f77b4", linewidth=2)
        ax1.plot(plotdf["year"], plotdf["w_disp_MRPL"], label="MRPL", color="#d62728", linewidth=2)
        ax1.set_title(f"{self.country} - Dispersion of MRPK and MRPL. {plotdf['year'][0]} = 1")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Standard Deviation")
        ax1.legend()

        plt.tight_layout()
        plt.show()

        return fig

    def _estimate_productivity(self):
        #calculate firm productivity z
        EPSILON = 3
        alpha = 0.35

        #calculate sectorrevenue and merge to df
        sectorrevenue = self.df.groupby(['year', 'sector']).agg(nvad_s = ('nvad', 'sum')).reset_index()
        self.df = self.df.merge(sectorrevenue, on=['year', 'sector'], how='left')

        #calculate firm productivity
        self.df['Z'] = ((self.df['nvad_s']**(-(1/(EPSILON-1)))) / self.df['priceind'] ) * ( (self.df['nvad']**(EPSILON/(EPSILON-1))) / ( (self.df['k']**alpha) * self.df['l']**(1-alpha)) )
        self.df['Z_pow'] = self.df['Z'] ** (EPSILON - 1)

    
        #df groupby sector and year
        sdf = self.df.groupby(['year', 'sector'])
        #calcualte log tfpe and tfp per year and sector by the formulas above
        log_tfpe_st =   ((1/(EPSILON-1)) 
                    * (np.log(sdf['FirmName'].count()) + np.log(sdf['Z_pow'].mean()) )
                    )
        log_tfp_st = np.log(sdf['y'].sum()) - alpha * np.log(sdf['k'].sum()) - (1-alpha) * np.log(sdf['l'].sum())
        #multiply with the sectorweights and sum to get tfp and tfpe per year
        tfp_st_df = pd.concat([log_tfpe_st, log_tfp_st], axis=1)
        tfp_st_df.columns = ['log_tfpe_st', 'log_tfp_st']
        tfp_st_df = tfp_st_df.reset_index()
        tfp_st_df = tfp_st_df.merge(self.sector_weights, on= 'sector', how='left')
        tfp_st_df['wlogtfpe'] = tfp_st_df['log_tfpe_st'] * tfp_st_df['sectorweight']
        tfp_st_df['wlogtfp'] = tfp_st_df['log_tfp_st'] * tfp_st_df['sectorweight']
        tfp_df = tfp_st_df.groupby('year').agg(log_tfp = ('wlogtfp', 'sum'),
                                            log_tfpe = ('wlogtfpe', 'sum'))
        #add a 1% growth tfp
        tfp_df['log_tfpg'] = tfp_df['log_tfp'].iloc[0] + np.log(1.01) * np.arange(len(tfp_df))

        return tfp_df
        

    def plot_productivity(self):
        plotdf = self.tfp_df.copy()
        plotdf = plotdf / plotdf.iloc[0] - 1
        plotdf = plotdf.reset_index()

        #create plot 
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8,6))

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

    def plot_mrpk_tfp_realrate(self):
        #get dispersion dataframe
        plotdf = self.dispt.copy()
        plotdf = plotdf.merge(self.tfpdf, left_index=True, right_index=True)
        #calculate relative change
        plotdf = (plotdf / plotdf.iloc[0]) - 1
        plotdf = plotdf.reset_index()
        #add realinterestrate
        plotdf = plotdf.merge(self.realrate.df[['year', 'realinterestrate']].drop_duplicates('year'), on='year', how='left')
        #calculate change or real interestrate
        plotdf['realinterestrate'] = plotdf['realinterestrate'] - plotdf.loc[0,'realinterestrate']


        fig, ax1 = plt.subplots(figsize=(8, 6))

        ax1.plot(plotdf["year"], plotdf["w_disp_MRPK"], label="MRPK Dispersion", color="#1f77b4", linewidth=2)
        ax1.plot(plotdf["year"], plotdf["realinterestrate"], label="Real Interest Rate", color="#ff7f0e", linewidth=2, linestyle="--")
        ax1.plot(plotdf["year"], plotdf["log_tfp"], label="TFP", color="green", linewidth=2, linestyle=":")

        ax1.axhline(y=0, color='black', linewidth=0.8, linestyle='--', dashes=(5, 10))

        ax1.set_xlabel("Year")
        ax1.set_ylabel(f"Normalized ({plotdf['year'].iloc[0]} = 0)")
        ax1.legend(loc="lower left")
        ax1.set_title(f"{self.country} - MRPK Dispersion vs. Real Interest Rate vs. TFP.")

        plt.tight_layout()
        plt.show()

        return fig

#winsorize function to drop the stupid ones
def winsorise(s, lower=0.01, upper=0.99):
    lo, hi = s.quantile([lower, upper])
    return s.clip(lo, hi)

            
