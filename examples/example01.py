import numpy as np
from pandas import read_csv
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor
from InventoryOptim.inventory import InventoryOptim

import warnings
warnings.filterwarnings("ignore", category=Warning)

try:
    df = read_csv('CFS.csv', parse_dates=['ds'], infer_datetime_format=True)
except FileNotFoundError:
    df = read_csv('./examples/CFS.csv', parse_dates=['ds'], infer_datetime_format=True)
pairs= [('FC', 'FCc'), ('GH', 'GHc'), ('EF', 'EFc'), ('OP', 'OPc'), ('SH', 'SHc')]
# define regressors
param_grid_sgd = {"alpha": np.logspace(-4, 0, 5),
                  "penalty": ['none', 'l1', 'l2', 'elasticnet'],
                  "tol": np.logspace(-4, 0, 5)}
sgd_regressor = RandomizedSearchCV(SGDRegressor(), param_distributions=param_grid_sgd, n_iter=25, cv=2)

# initialize
s_date = datetime(year=2018, month=1, day=1)
instance = InventoryOptim(df, date_fld='ds', units_costs=pairs, start_date=s_date,
			num_intrvl=(0., 1.), projection_date=datetime(year=2021, month=1, day=1),
			c_limit=.95)
instance.set_unit_count_regressor(LinearRegression())
instance.set_cost_regressor(LinearRegression())
instance.fit_regressors()
instance.plot_init_system().savefig('init.png', dpi=200)
instance.budget = lambda t: 30000-22000*np.exp(-t-1.)
# constraints
instance.constraint('GH', 650, datetime(year=2021, month=1, day=1))
instance.constraint('FC', 700, datetime(year=2021, month=1, day=1))
instance.constraint('FCc', 6., datetime(year=2020, month=1, day=1))
instance.constraint('GHc', 8., datetime(year=2021, month=1, day=1))
instance.constraint('EFc', 2.5, datetime(year=2020, month=1, day=1))
instance.constraint('OPc', 3.5, datetime(year=2020, month=1, day=1))
# run
instance.adjust_system(tbo='b')
fig = instance.plot_analysis()
fig[1].savefig('results.png', dpi=200)