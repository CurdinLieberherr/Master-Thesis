import pandas as pd
from linearmodels.panel import PanelOLS
import numpy as np
from typing import Literal

class WithinFirmMoments():
    def __init__(self, data: pd.DataFrame):
        self.capital_growth_regr, self.capital_growth_coefs = firm_capital_debt_regression(data, 'capital growth')
        self.debt_growth_regr, self.debt_growth_coefs = firm_capital_debt_regression(data, 'debt growth') 
        self.capital_growth_start_end_regr, self.capital_growth_start_end_coefs = firm_capital_debt_regression(data, 'capital growth', consecutive=False)

        self.coefs = pd.concat([self.capital_growth_coefs, self.capital_growth_start_end_coefs, self.debt_growth_coefs], axis=0)

def firm_capital_debt_regression(data: pd.DataFrame, 
        dep_variable:Literal['capital growth', 'debt growth'], 
        consecutive:bool = True):
    # Assuming your dataframe has columns: firm, year, sector, Z, k, a
    # Sort by firm and year so k_{t+1} aligns correctly
    df = data.copy()
    df = df.sort_values(['FirmName', 'year'])

    if consecutive == False:
        start, end = df['year'].min(), df['year'].max()
        #select firms that are appearing in start and end year
        firms_start = set(df.loc[df['year'] == start, 'FirmName'])
        firms_end = set(df.loc[df['year'] == end, 'FirmName'])
        # intersection
        valid_firms = firms_start & firms_end  
        df = df[df['FirmName'].isin(valid_firms)]

    if dep_variable == 'capital growth':
        # Dependent variable: (k_{t+1} - k_t) / k_t
        df['k_lead'] = df.groupby('FirmName')['k'].shift(-1)
        df['dep_var'] = (df['k_lead'] - df['k']) / df['k']

        # Regressors in logs
        df['log_Z'] = np.log(df['Z_pow'])
        df['log_a'] = np.log(df['a'])
        df['log_k'] = np.log(df['k'])
    elif dep_variable == 'debt growth':
        # Dependent variable: (k_{t+1} - k_t) / k_t
        df['b_lead'] = df.groupby('FirmName')['b'].shift(-1)
        df['dep_var'] = (df['b_lead'] - df['b']) / df['k']

        # Regressors in logs
        df['log_Z'] = np.log(df['Z_pow'])
        df['log_a'] = np.log(df['a'])
        df['log_k'] = np.log(df['k'])
    else:
        raise Exception('Enter valid dependent variable!')

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
    result.index = [f'Coef {var} on {dep_variable}{(" t_0 - T" if consecutive == False else "")}' for var in result.index]

    return res, result