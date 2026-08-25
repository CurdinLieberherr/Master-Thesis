import pandas as pd
import matplotlib.pyplot as plt
from MisallocationAnalysis import MisallocationAnalysis
from typing import Literal

COUNTRY_STYLES = {
    'Portugal': {'color': '#D6001C', 'marker': 'P', 'linestyle': ':'},   # flag red
    'Spain':    {'color': '#F57C00', 'marker': 'o', 'linestyle': '-'},   # warm orange (Spain-adjacent, distinct from Portugal's red)
    'Italy':    {'color': '#009246', 'marker': 's', 'linestyle': '--'},  # flag green
    'Belgium':  {'color': '#F9A825', 'marker': '^', 'linestyle': '-.'},  # flag gold
    'Sweden':   {'color': '#006AA7', 'marker': '*', 'linestyle': '-'},   # flag blue
    'Germany':  {'color': '#1A1A1A', 'marker': 'p', 'linestyle': '--'},  # flag black
}

class CompareCountries:
    def __init__(self, countries: list[MisallocationAnalysis], start: int = None, end: int= None):
        self.countries = countries
        self.start = start
        self.end = end

    def compare(self, sample: Literal['permanent', 'full'] = 'permanent'):
        if sample == 'permanent':
            permanent = True
        elif sample == 'full':
            permanent = False
        else: permanent = None

        data = []
        for country in self.countries:

            #override start year if given
            if self.start and self.start > country.start:
                start = self.start
            else:
                start = country.start
            if self.end and self.end < country.end:
                end = self.end
            else:
                end = country.end

            year_range = f"{start} - {end}"
            orbis_turnover_mean = round(country.compare_df_eurostats(permanent)['Turnover'].mean() * 100, 2)
            # realrate calculation
            rates = (
                country.realrate.df
                .query(f"year >= {start} and year <= {end}")
                .sort_values('year')
                ['realinterestrate']
            )
            diffs = (rates.diff().dropna() * 100).round(2)
            rr_total_increase = diffs[diffs > 0].sum().round(2)
            rr_total_decrease = diffs[diffs < 0].sum().round(2)
            rr_net_change = (diffs.sum()).round(2)

            #mrpk calculation
            dispt = country.dispt[country.dispt['sample'] == sample]
            mrpk = dispt.loc[start:end, 'w_disp_MRPK'].sort_index()
            pct_diffs = (mrpk.pct_change().dropna() * 100).round(2)
            mrpk_total_increase = pct_diffs[pct_diffs > 0].sum().round(2)
            mrpk_total_decrease = pct_diffs[pct_diffs < 0].sum().round(2)
            mrpk_net_change = pct_diffs.sum().round(2)
            #get moments data
            networth_on_capital_growth = country.within_firm_moments.coefs.loc['Coef log_a on capital growth', 'parameter']
            significance = country.within_firm_moments.coefs.loc['Coef log_a on capital growth', 'pvalue']

            data.append([country.country,year_range, orbis_turnover_mean, 
                        rr_net_change, rr_total_increase, rr_total_decrease,
                        mrpk_net_change, mrpk_total_increase, mrpk_total_decrease,
                        networth_on_capital_growth, significance])
        
        return pd.DataFrame(data, columns=['Country', 'Time', 'Revenue Data Coverage %',
                                            'Change Real Interest Rate %', 'Increase in Real Interest Rate %', 'Decrease in Real Interest Rate %',
                                              'Change log MRPK %', 'Increase in MRPK %', 'Decrease in MRPK %', 
                                              'Coef log_a on capital growth', 'P-Value Coefficient'])
    
    def plot_compare_mrpk(self, figsize=(12, 6), sample: Literal['permanent', 'full'] = 'permanent'):
        fig, ax = plt.subplots(figsize=figsize)

        for country in self.countries:
            # Apply start/end filtering
            start = max(self.start, country.start) if self.start else country.start
            end = min(self.end, country.end) if self.end else country.end

            dispt = country.dispt[country.dispt['sample'] == sample]
            series = dispt.loc[start:end, 'w_disp_MRPK'].dropna()

            # Normalize so start year == 0 (relative growth)
            base = series.iloc[0]
            relative = (series - base) / base * 100

            color = COUNTRY_STYLES.get(country.country)['color']
            linestyle = COUNTRY_STYLES.get(country.country)['linestyle']
            ax.plot(relative.index, relative.values, label=country.country, color=color, linestyle=linestyle, linewidth=2)

        #find start years
        startyears = []
        for country in self.countries:
            if country.start not in startyears:
                startyears.append(country.start)
        #add vertical line
        for i,year in enumerate(startyears):
            ax.axvline(year, color='grey', linewidth=0.8, linestyle='--')
            ax.text(year, ax.get_ylim()[1], f'Period {i+1}', ha='center', va='bottom')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title("Relative Growth of MRPK Dispersion (Start Year = 0%)")
        ax.set_xlabel("Year")
        ax.set_ylabel("% Change from Start Year")

        # dedupe legend in case the same country is plotted more than once (e.g. two periods)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.margins(x=0.05, y=0.1)  
        plt.tight_layout()
        plt.show()

        return fig

    def compare_all_moments(self) -> pd.DataFrame:
        df = pd.DataFrame()

        for country in self.countries:
            moments = country.all_moments
            moments.name = country.country + f" {country.start}-{country.end}"
            df = pd.concat([df,moments], axis=1)

        return df

    def overview_statistics(self, permanent=False):
        df = pd.DataFrame()
        for country in self.countries:
            df = pd.concat([df, country.overview_statistics(permanent=permanent)], axis=0, ignore_index=True)

        return df

    def descriptive_statistics(self, permanent=False):
        df = pd.DataFrame()
        for country in self.countries:
            stats = country.descriptive_statistics(permanent=permanent)
            df = pd.concat([df, stats], axis=0, ignore_index=True)

        return df

        
            



