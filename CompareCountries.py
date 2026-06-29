import pandas as pd
import matplotlib.pyplot as plt
from MisallocationAnalysis import MisallocationAnalysis

class CompareCountries:
    def __init__(self, countries: list[MisallocationAnalysis], start: int = None, end: int= None):
        self.countries = countries
        self.start = start
        self.end = end

    def compare(self):

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
            orbis_turnover_mean = (country.compare_df_eurostats()['Turnover'].mean() * 100).round(2)
            real_rate_change = ((country.realrate.df.query(f"year == {end}")['realinterestrate'].values[0] - country.realrate.df.query(f"year == {start}")['realinterestrate'].values[0]) * 100).round(2)
            mrpk_change = (((country.dispt.loc[end, 'w_disp_MRPK'] - country.dispt.loc[start, 'w_disp_MRPK'] ) / country.dispt.loc[start, 'w_disp_MRPK'] ) * 100).round(2)

            data.append([country.country,year_range, orbis_turnover_mean, real_rate_change, mrpk_change ])
        
        return pd.DataFrame(data, columns=['Country', 'Time', 'Revenue Data Coverage %', 'Change Real Interest Rate %', 'Change log MRPK %'])
    
    def plot_compare_mrpk(self, figsize=(12, 6)):
        fig, ax = plt.subplots(figsize=figsize)

        for country in self.countries:
            # Apply start/end filtering
            start = max(self.start, country.start) if self.start else country.start
            end = min(self.end, country.end) if self.end else country.end

            series = country.dispt.loc[start:end, 'w_disp_MRPK'].dropna()

            # Normalize so start year == 0 (relative growth)
            base = series.iloc[0]
            relative = (series - base) / base * 100

            ax.plot(relative.index, relative.values, label=country.country, linewidth=2)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title("Relative Growth of MRPK Dispersion (Start Year = 0%)")
        ax.set_xlabel("Year")
        ax.set_ylabel("% Change from Start Year")
        ax.legend()
        plt.tight_layout()
        plt.show()

        return fig
            



