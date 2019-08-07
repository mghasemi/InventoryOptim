class InventoryOptim(object):
    """
    :param df: the `DataFrame` containing data point
    :param units_costs: a list of pairs :math:`(G_i, C_i)`.
    :param date_fld: `string` the name of the column keeping each row's date
    :param start_date: `None` or `datetime`the start date of the analysis;
        if `None` the minimum date found in `date_fld` is used.
    :param num_intrvl: `2-tuple` the numerical range to be used for converting
        dates to numbers
    :param projectioni_date: `datetime` the target date of the analysis
    :param c_limit: float between 0 and 1, the confidence interval
    :param min_samples: `int` minimum number of samples to perform Monte Carlo sampling
    :param error_tol: `float` error tolerance
    """

    def __init__(
            self,
            df,
            units_costs,
            date_fld="date",
            start_date=None,
            num_intrvl=(0.0, 10.0),
            projection_date=None,
            c_limit=0.95,
            min_samples=5,
            error_tol=1.0e-4,
    ):
        clmns = list(df.columns)
        if date_fld in clmns:
            self.date_fld = date_fld
        else:
            raise Exception("'%s' is not a column of the given DataFrame" % date_fld)
        dts = list(df[self.date_fld])
        self.MinDate = min(dts)
        self.MaxDate = max(dts)
        if start_date is None:
            self.start_date = self.MinDate
        else:
            if (start_date < self.MinDate) or (self.MaxDate < start_date):
                raise Exception("The given start date is out of the DataFrame's scope")
            self.start_date = start_date
        self.num_intrvl = num_intrvl
        self.min_samples = min_samples
        self.error_tol = error_tol
        self.unit_flds = []
        self.cost_flds = []
        self.unit_cost = []
        for uc in units_costs:
            if len(uc) != 2:
                raise Exception("'units_costs' must be a lista of pairs.")
            if (uc[0] in clmns) and (uc[1] in clmns):
                self.unit_flds.append(uc[0])
                self.cost_flds.append(uc[1])
                self.unit_cost += list(uc)
        self.df = df[[self.date_fld] + self.unit_cost].sort_values(by=self.date_fld)
        mp = [self.date2num(_) for _ in list(self.df[self.date_fld])]
        self.df["T"] = mp
        self.df["TotalUnits"] = self.df.apply(
            lambda x, flds=tuple(self.unit_flds): sum([x[_] for _ in flds]), axis=1
        )
        self.df["TotalCost"] = self.df.apply(
            lambda x, cst=tuple(self.cost_flds), unt=tuple(self.unit_flds): sum(
                [x[cst[_]] * x[unt[_]] for _ in range(len(cst))]
            ),
            axis=1,
        )
        self.training_size = sum([1 if _ >= 0 else 0 for _ in mp])
        if projection_date is not None:
            self.projection_date = projection_date
            self.FT = self.date2num(projection_date)
        else:
            from datetime import timedelta

            projection_date = self.MaxDate + timedelta(50)
            self.projection_date = projection_date
            self.FT = self.date2num(projection_date)
        self.c_limit = (c_limit + 1.0) / 2.0
        self.colors = [
            "#549dce",
            "#d86950",
            "#28b789",
            "#49dd1c",
            "#864daf",
            "#ef6f34",
            "#db99d1",
            "#4442c4",
            "#4286f4",
            "#46d6cc",
            "#46d6cc",
        ]

        # set the default regressors
        from sklearn.linear_model import LinearRegression

        self.unit_regressor = LinearRegression()
        self.cost_regressor = LinearRegression()
        self.unt_reg = []
        self.cst_reg = []
        self.tu_reg = None
        self.tc_reg = None
        self.fitted = False
        self.analyzed = False
        self.variance_from_trend = {}
        self.constraints = []
        self.bound_flds = set()
        self.init_x = {}
        self.init_y = {}
        self.init_fits = {}
        self.partial_fits = {}
        self.result = None
        self.budget = lambda t: 0.0

    def date2num(self, dt):
        """
        Converts a `datetime` to a number according to `self.num_intrvl`

        :param dt: `datetime`
        """
        slope = (
                float(self.num_intrvl[1] - self.num_intrvl[0])
                / (self.MaxDate - self.start_date).days
        )
        y = slope * (dt - self.start_date).days + self.num_intrvl[0]
        return y

    def set_unit_count_regressor(self, regressor):
        """
        Sets the regressor for unit counts. Any regression inherited from `sk-learn.RegressorMixin` is acceptable

        :param regressor: `RegressorMixin`
        """
        self.unit_regressor = regressor

    def set_cost_regressor(self, regressor):
        """
        Sets the regressor for unit costs. Any regression inherited from `sk-learn.RegressorMixin` is acceptable

        :param regressor: `RegressorMixin`
        """
        self.cost_regressor = regressor

    def fit_regressors(self):
        """
        Initializes the regression objects and fit them on training data
        """
        from numpy import reshape
        from scipy.stats import t
        from copy import copy

        n_flds = len(self.unit_flds)
        t_df = self.df[self.df["T"] >= 0]
        X = reshape(t_df["T"].values, (-1, 1))
        for idx in range(n_flds):
            y_u = t_df[self.unit_flds[idx]].values
            y_c = t_df[self.cost_flds[idx]].values
            reg_u = copy(self.unit_regressor)
            reg_c = copy(self.cost_regressor)
            reg_u.fit(X, y_u)
            reg_c.fit(X, y_c)
            self.init_fits[self.unit_flds[idx]] = copy(reg_u)
            self.init_fits[self.cost_flds[idx]] = copy(reg_c)
            self.unt_reg.append(copy(reg_u))
            self.cst_reg.append(copy(reg_c))
        y_u = t_df["TotalUnits"].values
        y_c = t_df["TotalCost"].values
        reg_u = copy(self.unit_regressor)
        reg_c = copy(self.cost_regressor)
        reg_u.fit(X, y_u)
        reg_c.fit(X, y_c)
        self.tu_reg = copy(reg_u)
        self.tc_reg = copy(reg_c)
        self.fitted = True

    def _conf_ints(self):
        """
        Calculates the confidence intervals for regression curves
        """
        from numpy import array, reshape, power, sum, sqrt, linspace
        from scipy.stats import t

        u_conf = []
        unts = []
        p_unts = []
        c_conf = []
        csts = []
        p_csts = []
        n_flds = len(self.unit_flds)
        t_df = self.df[self.df["T"] >= 0]
        x = t_df["T"].values
        X = reshape(x, (-1, 1))
        mean_x = x.mean()
        n = X.shape[0]
        tstat = t.ppf(self.c_limit, n - 1)
        fx = linspace(0.0, self.FT, 100)
        rfx = reshape(fx, (-1, 1))
        for idx in range(n_flds):
            y_u = t_df[self.unit_flds[idx]].values
            unts.append(y_u)
            p_unts.append(self.unt_reg[idx].predict(rfx))
            s_err_u = sum(power(y_u - self.unt_reg[idx].predict(X), 2))
            self.variance_from_trend[self.unit_flds[idx]] = sqrt(s_err_u / n)
            conf_u = (
                    tstat
                    * sqrt((s_err_u / (n - 2)))
                    * (
                            1.0 / n
                            + (
                                    power(fx - mean_x, 2)
                                    / ((sum(power(x, 2))) - n * (power(mean_x, 2)))
                            )
                    )
            )
            u_conf.append(conf_u)
            y_c = t_df[self.cost_flds[idx]].values
            csts.append(y_c)
            p_csts.append(self.cst_reg[idx].predict(rfx))
            s_err_c = sum(power(y_c - self.cst_reg[idx].predict(X), 2))
            self.variance_from_trend[self.cost_flds[idx]] = sqrt(s_err_c / n)
            conf_c = (
                    tstat
                    * sqrt((s_err_c / (n - 2)))
                    * (
                            1.0 / n
                            + (
                                    power(fx - mean_x, 2)
                                    / ((sum(power(x, 2))) - n * (power(mean_x, 2)))
                            )
                    )
            )
            c_conf.append(conf_c)
        return x, fx, unts, p_unts, u_conf, csts, p_csts, c_conf

    def plot_init_system(self):
        """
        Plots the initial data points and regression curves for projection date
        """
        from numpy import abs
        import matplotlib.pyplot as plt
        # from matplotlib import colors as mcolors

        plt.figure(figsize=(30, 20))
        # self.colors = list(mcolors.CSS4_COLORS.keys())[5:]
        if not self.fitted:
            self.fit_regressors()
        x, fx, u, p_u, cnf_u, c, p_c, cnf_c = self._conf_ints()
        n_flds = len(self.unit_flds)
        all_x = self.df["T"].values
        fig, axes = plt.subplots(
            nrows=2, ncols=1, figsize=(20, 20), sharex=False, sharey=False
        )
        for idx in range(n_flds):
            axes[0].plot(fx, p_u[idx], color=self.colors[idx % len(self.colors)])
            axes[0].scatter(
                all_x,
                self.df[self.unit_flds[idx]].values,
                color=self.colors[idx % len(self.colors)],
                s=6,
            )
            axes[0].fill_between(
                fx,
                p_u[idx] - abs(cnf_u[idx]),
                p_u[idx] + abs(cnf_u[idx]),
                color=self.colors[idx % len(self.colors)],
                alpha=0.1,
            )
            axes[0].grid(True)
            axes[1].plot(fx, p_c[idx], color=self.colors[idx % len(self.colors)])
            axes[1].scatter(
                all_x,
                self.df[self.cost_flds[idx]].values,
                color=self.colors[idx % len(self.colors)],
                s=5,
            )
            axes[1].fill_between(
                fx,
                p_c[idx] - abs(cnf_c[idx]),
                p_c[idx] + abs(cnf_c[idx]),
                color=self.colors[idx % len(self.colors)],
                alpha=0.1,
            )
            axes[1].grid(True)
        axes[0].legend(self.unit_flds)
        axes[1].legend(self.cost_flds)
        return fig

    def plot_analysis(self):
        """
        Plots the outcome of the adjustment.
        """
        from numpy import abs, array, multiply
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from datetime import timedelta

        plt.figure(figsize=(40, 20))
        if not self.analyzed:
            self.adjust_system("b")
        x, fx, u, p_u, cnf_u, c, p_c, cnf_c = self._conf_ints()
        tot_trend_cost = None
        tot_actual_cost = None
        tot_changed_cost = None
        n_flds = len(self.unit_flds)
        all_x = self.df["T"].values
        fig, axes = plt.subplots(
            nrows=2, ncols=2, figsize=(20, 20), sharex=False, sharey=False
        )
        for idx in range(n_flds):
            #
            axes[0, 0].plot(
                fx, p_u[idx], color=self.colors[idx % len(self.colors)], ls=":"
            )
            axes[0, 0].scatter(
                all_x,
                self.df[self.unit_flds[idx]].values,
                color=self.colors[idx % len(self.colors)],
                s=6,
            )
            axes[0, 0].fill_between(
                fx,
                p_u[idx] - abs(cnf_u[idx]),
                p_u[idx] + abs(cnf_u[idx]),
                color=self.colors[idx % len(self.colors)],
                alpha=0.1,
            )
            ys_u = array([self.partial_fits[self.unit_flds[idx]](_) for _ in fx])
            axes[0, 0].plot(fx, ys_u, color=self.colors[idx % len(self.colors)])
            axes[0, 0].grid(True)
            for cns in self.constraints:
                if cns[0] in self.unit_flds:
                    axes[0, 0].scatter(
                        [self.date2num(cns[2])], [cns[1]], cmap="cubehelix", alpha=0.2
                    )

            axes[0, 1].plot(
                fx, p_c[idx], color=self.colors[idx % len(self.colors)], ls=":"
            )
            axes[0, 1].scatter(
                all_x,
                self.df[self.cost_flds[idx]].values,
                color=self.colors[idx % len(self.colors)],
                s=5,
            )
            axes[0, 1].fill_between(
                fx,
                p_c[idx] - abs(cnf_c[idx]),
                p_c[idx] + abs(cnf_c[idx]),
                color=self.colors[idx % len(self.colors)],
                alpha=0.1,
            )
            ys_c = array([self.partial_fits[self.cost_flds[idx]](_) for _ in fx])
            axes[0, 1].plot(fx, ys_c, color=self.colors[idx % len(self.colors)])
            axes[0, 1].grid(True)
            for cns in self.constraints:
                if cns[0] in self.cost_flds:
                    axes[0, 1].scatter(
                        [self.date2num(cns[2])], [cns[1]], cmap="cubehelix", alpha=0.2
                    )
            #
            trend_cost = multiply(p_u[idx], p_c[idx])
            actual_cost = multiply(
                self.df[self.unit_flds[idx]].values, self.df[self.cost_flds[idx]].values
            )
            changed_cost = multiply(ys_u, ys_c)
            if tot_trend_cost is None:
                tot_trend_cost = trend_cost
                tot_actual_cost = actual_cost
                tot_changed_cost = changed_cost
            else:
                tot_trend_cost += trend_cost
                tot_actual_cost += actual_cost
                tot_changed_cost += changed_cost
            axes[1, 0].plot(
                fx, trend_cost, color=self.colors[idx % len(self.colors)], ls=":"
            )
            axes[1, 0].scatter(
                all_x, actual_cost, color=self.colors[idx % len(self.colors)], s=6
            )
            axes[1, 0].plot(fx, changed_cost, color=self.colors[idx % len(self.colors)])
            axes[1, 0].grid(True)
        #
        budget = array([self.budget(_) for _ in fx])
        residual = budget - tot_changed_cost
        gain = tot_trend_cost - tot_changed_cost
        axes[1, 1].plot(
            fx, tot_trend_cost, color=self.colors[n_flds % len(self.colors)], ls=":"
        )
        axes[1, 1].scatter(
            all_x,
            tot_actual_cost,
            color=self.colors[(n_flds + 1) % len(self.colors)],
            s=6,
        )
        axes[1, 1].plot(
            fx, tot_changed_cost, color=self.colors[(n_flds + 2) % len(self.colors)]
        )
        axes[1, 1].plot(fx, budget, color=self.colors[(n_flds + 3) % len(self.colors)])
        axes[1, 1].plot(
            fx, residual, color=self.colors[(n_flds + 4) % len(self.colors)]
        )
        axes[1, 1].plot(fx, gain, color=self.colors[(n_flds + 6) % len(self.colors)])
        axes[1, 1].grid(True)
        #
        axes[0, 0].legend(
            [
                item
                for sublist in [["_nolegend_", loc] for loc in self.unit_flds]
                for item in sublist
            ]
        )
        axes[0, 0].set_title("Capacity")
        axes[0, 1].legend(
            [
                item
                for sublist in [["_nolegend_", loc] for loc in self.cost_flds]
                for item in sublist
            ]
        )
        axes[0, 1].set_title("Unit Costs")
        axes[1, 0].legend(
            [item for sublist in [["_nolegend_", self.unit_flds[_] + "*" + self.cost_flds[_]]
                                  for _ in range(n_flds)] for item in sublist]
        )
        axes[1, 0].set_title("Costs")
        axes[1, 1].legend(
            [
                "Trend of total cost",
                "Proj. expected cost",
                "Budget",
                "Residual",
                "Gain",
                "Total cost",
            ]
        )
        axes[1, 1].set_title("Total Costs")
        return plt, fig, axes

    def constraint(self, fld, value, dt):
        """
        Suggest a constraint for future.

        :param fld: `str` the column whose values is about to be adjusted
        :param value: `float` the suggested value for the given date
        :param dt: `datetime` the suggested date for adjustment
        """
        self.constraints.append((fld, value, dt))
        self.bound_flds.add(fld)

    def make_date_interval(self, dt, n_days):
        """
        Makes a list of 2*`n_days` dates centered at `dt`
        """
        from datetime import timedelta

        return [dt + timedelta(days=_) for _ in range(-n_days, n_days + 1)]

    def make_date_interval_val(self, dt, n_days):
        """
        Converts the outcome of `self.make_date_interval` into a list of floats
        """
        from datetime import timedelta

        return [
            self.date2num(dt + timedelta(days=_)) for _ in range(-n_days, n_days + 1)
        ]

    def refit(self, fld, val, dt, n_points):
        """
        Refits the regressor of the `fld` after producing `n_points` samples points
        around `dt` using a normal distribution centered at `val`

        :param fld: the regression associated to `fld` will be refitted
        :param val: the suggested value for the regression curve at `dt`
        :param dt: the suggested `datetime` to make adjustments to the values of `fld`
        :param n_points: number of samples to be generated for refitting
        """
        from numpy import array, append
        from numpy.random import normal
        from copy import copy

        date_interval = array(self.make_date_interval_val(dt, n_points)).reshape(
            (-1, 1)
        )
        y_sample = normal(val, self.variance_from_trend[fld], 2 * n_points + 1)
        X = append(self.init_x[fld], date_interval).reshape((-1, 1))
        y = append(self.init_y[fld], y_sample)
        if fld in self.unit_flds:
            regressor = copy(self.unit_regressor)
        else:
            regressor = copy(self.cost_regressor)
        regressor.fit(X, y)
        return lambda t, reg=copy(regressor): reg.predict(array([t]).reshape(-1, 1))[0]

    def adjust_system(self, tbo="u"):
        """
        Forms and solves the optimization problem for trend adjustment

        :param tbo: `char` if 'u' only trends will be adjusted regardless of unit costs.
            if 'b' costs of units will be used to adjust trends
        """
        from numpy import array, append, reshape
        from numpy.random import normal
        from scipy.optimize import minimize
        from copy import copy

        if not self.fitted:
            self.fit_regressors()
        num_points = max(
            self.min_samples, int(self.training_size * (1.0 - self.c_limit))
        )
        t_df = self.df[self.df["T"] >= 0]
        np_ft = array([self.FT]).reshape(-1, 1)
        ########################
        # add cost constraints #
        ########################
        for fld in self.cost_flds:
            if fld not in self.bound_flds:
                val = self.init_fits[fld].predict(np_ft)
                self.constraint(fld, val, self.projection_date)
        ########################
        x_ = t_df["T"].values
        X = reshape(x_, (-1, 1))
        if tbo == "u":
            sel_flds = self.unit_flds
            tfn = self.tu_reg.predict([[self.FT]])[0]
        elif tbo == "c":
            sel_flds = self.cost_flds
            tfn = self.tc_reg.predict(np_ft)[0]
        else:
            sel_flds = self.unit_cost
            tfn = self.tu_reg.predict(np_ft)[0]
        for fld in sel_flds:
            self.init_x[fld] = X
            self.init_y[fld] = t_df[fld].values
        for cns in self.constraints:
            fld = cns[0]
            if fld not in sel_flds:
                continue
            date_interval = array(
                self.make_date_interval_val(cns[2], num_points)
            ).reshape((-1, 1))
            y_sample = normal(cns[1], self.variance_from_trend[fld], 2 * num_points + 1)
            self.init_x[fld] = append(self.init_x[fld], date_interval).reshape((-1, 1))
            self.init_y[fld] = append(self.init_y[fld], y_sample)
            if fld in self.unit_flds:
                regressor = copy(self.unit_regressor)
            else:
                regressor = copy(self.cost_regressor)
            regressor.fit(self.init_x[fld], self.init_y[fld])
            self.partial_fits[fld] = lambda t, reg=copy(regressor): reg.predict(
                array([t]).reshape(-1, 1)
            )[0]

        def to_be_optimized(x, tbo="u"):
            from numpy import array, append, reshape
            from scipy.integrate import quad
            from copy import copy

            idx = 0
            fns = {}
            if tbo == "u":
                selected_flds = self.unit_flds
            elif tbo == "c":
                selected_flds = self.cost_flds
            else:
                selected_flds = self.unit_cost
            for fld_ in selected_flds:
                if fld_ not in self.bound_flds:
                    fns[fld_] = self.refit(fld_, x[idx], self.projection_date, num_points)
                    idx += 1
                else:
                    fns[fld_] = lambda t, fld=fld_: self.partial_fits[fld](t)
            obj = quad(
                lambda t, fns=fns: sum(
                    [
                        (
                                fns[fld](t)
                                - self.init_fits[fld].predict(array([t]).reshape(-1, 1))
                        )
                        ** 2
                        for fld in self.unit_flds
                    ]
                ),
                0.0,
                1.0,
            )[0]
            cost_obj = 0.0
            if tbo == "b":
                cost_obj = quad(
                    lambda t: sum(
                        [
                            fns[self.unit_flds[_]](t) * fns[self.cost_flds[_]](t)
                            - self.budget(t)
                            for _ in range(len(self.unit_flds))
                        ]
                    ),
                    0.0,
                    self.FT,
                )[0]
            return obj + cost_obj

        residual = 0.0
        cns = ()
        for fld in sel_flds:
            if fld in self.bound_flds:
                if fld in self.unit_flds:
                    residual += self.partial_fits[fld](self.FT)

        def cost_residual(x, sel_flds):
            cst_res = 0.0
            fld_idx = {}
            idx = 0
            for fld_ in sel_flds:
                if fld_ not in self.bound_flds:
                    fld_idx[fld_] = idx
                    idx += 1
            for fld_ in sel_flds:
                t_cst = 0.0
                if fld_ in self.unit_flds:
                    u_fld = fld_
                    c_fld = self.cost_flds[self.unit_flds.index(fld_)]
                else:
                    c_fld = fld_
                    u_fld = self.unit_flds[self.cost_flds.index(fld_)]
                if u_fld in self.bound_flds:
                    t_cst = self.partial_fits[u_fld](self.FT)
                else:
                    t_cst = x[fld_idx[u_fld]]
                if c_fld in self.bound_flds:
                    t_cst *= self.partial_fits[c_fld](self.FT)
                else:
                    t_cst *= x[fld_idx[c_fld]]
                cst_res += t_cst
            res = self.budget(self.FT) - cst_res / 2.0
            return res

        if tbo in ["u", "b"]:
            cns = (
                {
                    "type": "ineq",
                    "fun": lambda x, rsdl=residual, tfn=tfn: sum(x)
                                                             + rsdl
                                                             - tfn
                                                             + self.error_tol,
                },
                {
                    "type": "ineq",
                    "fun": lambda x, rsdl=residual, tfn=tfn: -(sum(x) + rsdl - tfn)
                                                             + self.error_tol,
                },
                {
                    "type": "ineq",
                    "fun": lambda x, sel_flds=tuple(sel_flds): cost_residual(x, sel_flds),
                },
            )
        x0 = []
        idx_flds = []
        for fld in sel_flds:
            if fld not in self.bound_flds:
                idx_flds.append(fld)
                x0.append(self.init_fits[fld].predict(np_ft)[0])

        res = minimize(
            to_be_optimized, x0=array(x0), method="COBYLA", constraints=cns, args=(tbo)
        )
        self.result = res
        adj_x = res.x
        for fld in idx_flds:
            self.partial_fits[fld] = self.refit(
                fld, adj_x[idx_flds.index(fld)], self.projection_date, num_points
            )
        self.analyzed = True
