import pandas as pd
from linearmodels.panel import PanelOLS
import numpy as np

#winsorize function to drop the stupid ones
def winsorise(s, lower=0.01, upper=0.99):
    lo, hi = s.quantile([lower, upper])
    return s.clip(lo, hi)

class WithinFirmMoments():
    def __init__(self, data: pd.DataFrame):
    
        # Assuming your dataframe has columns: firm, year, sector, Z, k, a
        # Sort by firm and year so k_{t+1} aligns correctly
        df = data.copy()
        df = df.sort_values(['FirmName', 'year'])

        # Dependent variable: (k_{t+1} - k_t) / k_t
        df['k_lead'] = df.groupby('FirmName')['k'].shift(-1)
        df['dep_var'] = (df['k_lead'] - df['k']) / df['k']

        # Regressors in logs
        df['log_Z'] = np.log(df['Z_pow'])
        df['log_a'] = np.log(df['a'])
        df['log_k'] = np.log(df['k'])

        #winsorize to remove outliers
        for var in ['dep_var', 'log_Z', 'log_a', 'log_k']:
            df[var] = df.groupby('year')[var].transform(winsorise)

        # Sector-year fixed effect grouping variable
        df['sector_year'] = df['sector'].astype(str) + "_" + df['year'].astype(str)

        # Drop rows with missing values needed for regression. because ther are gap in years
        df_reg = df.dropna(subset=['dep_var', 'log_Z', 'log_a', 'log_k', 'FirmName', 'sector_year'])
        df_reg = df_reg.set_index(['FirmName', 'year'])

        exog = df_reg[['log_Z', 'log_a', 'log_k']]
        exog = sm.add_constant(exog) if False else exog  # constant absorbed by FE, skip

        mod = PanelOLS(
            dependent=df_reg['dep_var'],
            exog=exog,
            entity_effects=True,       # absorbs d_i (firm FE)
            time_effects=False,
            other_effects=df_reg['sector_year']  # absorbs d_st (sector-year FE)
        )

        res = mod.fit(cov_type='clustered', cluster_entity=True)
        result = pd.concat([res.params, res.std_errors, res.tstats, res.pvalues], axis=1).round(3)
        result.index = [f'Coefficient of {var} on firm capital growth' for var in result.index]

        self.res = res
        self.coefficients = result

    def summary(self):
        print(self.res)