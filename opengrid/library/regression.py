# -*- coding: utf-8 -*-
"""
General analysis functions.

Try to write all methods such that they take a dataframe as input
and return a dataframe or list of dataframes.
"""
import datetime as dt
import pandas as pd

from .plotting import plot_style
plt = plot_style()
import numpy as np
import statsmodels.formula.api as fm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from patsy import ModelDesc, Term, LookupFactor
from copy import deepcopy
import re

from .analysis import Analysis


class MultiVarLinReg(Analysis):
    """
    Multi-variable linear regression based on statsmodels and Ordinary Least Squares (ols)

    Pass a dataframe with the variable to be modelled y (dependent variable) and the possible independent variables x.
    Specify as string the name of the dependent variable, and optionally pass a list with names of
    independent variables to try (by default all other columns will be tried as independent variables).

    The analysis is based on a forward-selection approach: starting from a simple model, the model is iteratively
    refined and verified until no statistical relevant improvements can be obtained.  Each model in the iteration loop
    is stored in the attribute self.list_of_fits.  The selected model is self.fit (=pointer to the last element of
    self.list_of_fits).

    The dataframe can contain daily, weekly, monthly, yearly ... values.  Each row is an instance.


    Examples
    --------

    >> mvlr = MultiVarLinReg(df, 'gas', p_max=0.04)
    >> mvlr = MultiVarLinReg(df, 'gas', list_of_x=['heatingDegreeDays14', 'GlobalHorizontalIrradiance', 'WindSpeed'])


    """

    def __init__(self, df, y, p_max=0.05, list_of_x=None, confint=0.95, cross_validation=False,
                 allow_negative_predictions=False):
        """

        Parameters
        ----------
        df : pd.DataFrame
            Datetimeindex and both independent variables (x) and dependent variable (y) as columns
        y : str
            Name of the dependent (endogeneous) variable to model
        p_max : float (default=0.05)
            Acceptable p-value of the t-statistic for estimated parameters
        list_of_x : list of str (default=None)
            If None (default), try to build a model with all columns in the dataframe
            If a list with column names is given, only try these columns as independent variables
        confint : float, default=0.95
            Two-sided confidence interval for predictions.
        cross_validation : bool, default=False
            If True, compute the model based on cross-validation (leave one out)
            Only possible if the df has less than 15 entries.
            Note : this will take much longer computation times!
        allow_negative_predictions : bool, default=False
            If True, allow predictions to be negative.
            For gas consumption or PV production, this is not physical so allow_negative_predictions should be False
        """
        self.df = df.copy()  # type: pd.DataFrame
        assert y in self.df.columns, "The dependent variable {} is not a column in the dataframe".format(y)
        self.y = y

        self.p_max = p_max
        self.list_of_x = list_of_x or self.df.columns.tolist()
        self.confint = confint
        self.cross_validation = cross_validation
        self.allow_negative_predictions = allow_negative_predictions
        try:
            self.list_of_x.remove(self.y)
        except ValueError:
            pass
        self._fit = None
        self._list_of_fits = []


    @property
    def fit(self):
        if self._fit is None:
            raise UnboundLocalError('Run "do_analysis()" first to fit a model to the data.')
        else:
            return self._fit


    @property
    def list_of_fits(self):
        if not self._list_of_fits:
            raise UnboundLocalError('Run "do_analysis()" first to fit a model to the data.')
        else:
            return self._list_of_fits


    def do_analysis(self):
        """
        Find the best model (fit) and create self.list_of_fits and self.fit

        """
        if self.cross_validation:
            return self._do_analysis_cross_validation()
        else:
            return self._do_analysis_no_cross_validation()

    def _do_analysis_no_cross_validation(self):
        """
        Find the best model (fit) and create self.list_of_fits and self.fit
        """

        # first model is just the mean
        response_term = [Term([LookupFactor(self.y)])]
        model_terms = [Term([])] # empty term is the intercept
        all_model_terms_dict = {x:Term([LookupFactor(x)]) for x in self.list_of_x}
        # ...then add another term for each candidate
        #model_terms += [Term([LookupFactor(c)]) for c in candidates]
        model_desc = ModelDesc(response_term, model_terms)
        self._list_of_fits.append(fm.ols(model_desc, data=self.df).fit())
        # try to improve the model until no improvements can be found

        while all_model_terms_dict:
            # try each x and overwrite the best_fit if we find a better one
            # the first best_fit is the one from the previous round
            ref_fit = self._list_of_fits[-1]
            best_fit = self._list_of_fits[-1]
            best_bic = best_fit.bic
            for x, term in all_model_terms_dict.items():
                # make new_fit, compare with best found so far
                model_desc = ModelDesc(response_term, ref_fit.model.formula.rhs_termlist + [term])
                fit = fm.ols(model_desc, data=self.df).fit()
                if fit.bic < best_bic:
                    best_bic = fit.bic
                    best_fit = fit
                    best_x = x
            # Sometimes, the obtained fit may be better, but contains unsignificant parameters.
            # Correct the fit by removing the unsignificant parameters and estimate again
            best_fit = self._prune(best_fit, p_max=self.p_max)

            # if best_fit does not contain more variables than ref fit, exit
            if len(best_fit.model.formula.rhs_termlist) == len(ref_fit.model.formula.rhs_termlist):
                break
            else:
                self._list_of_fits.append(best_fit)
                all_model_terms_dict.pop(best_x)
        self._fit = self._list_of_fits[-1]

    def _do_analysis_cross_validation(self):
        """
        Find the best model (fit) based on cross-valiation (leave one out)

        """
        assert len(self.df) < 15, "Cross-validation is not implemented if your sample contains more than 15 datapoints"

        # initialization: first model is the mean, but compute cv correctly.
        errors = []
        response_term = [Term([LookupFactor(self.y)])]
        model_terms = [Term([])]  # empty term is the intercept
        model_desc = ModelDesc(response_term, model_terms)
        for i in self.df.index:
            # make new_fit, compute cross-validation and store error
            df_ = self.df.drop(i, axis=0)
            fit = fm.ols(model_desc, data=df_).fit()
            cross_prediction = self._predict(fit=fit, df=self.df.loc[[i], :])
            errors.append(cross_prediction['predicted'] - cross_prediction[self.y])

        self._list_of_fits = [fm.ols(model_desc, data=self.df).fit()]
        self.list_of_cverrors = [np.mean(np.abs(np.array(errors)))]

        # try to improve the model until no improvements can be found
        all_model_terms_dict = {x: Term([LookupFactor(x)]) for x in self.list_of_x}
        while all_model_terms_dict:
            # import pdb;pdb.set_trace()
            # try each x in all_exog and overwrite if we find a better one
            # at the end of iteration (and not earlier), save the best of the iteration
            better_model_found = False
            best = dict(fit=self._list_of_fits[-1], cverror=self.list_of_cverrors[-1])
            for x, term in all_model_terms_dict.items():
                model_desc = ModelDesc(response_term, self._list_of_fits[-1].model.formula.rhs_termlist + [term])
                # cross_validation, currently only implemented for monthly data
                # compute the mean error for a given formula based on leave-one-out.
                errors = []
                for i in self.df.index:
                    # make new_fit, compute cross-validation and store error
                    df_ = self.df.drop(i, axis=0)
                    fit = fm.ols(model_desc, data=df_).fit()
                    cross_prediction = self._predict(fit=fit, df=self.df.loc[[i], :])
                    errors.append(cross_prediction['predicted'] - cross_prediction[self.y])
                cverror = np.mean(np.abs(np.array(errors)))
                # compare the model with the current fit
                if cverror < best['cverror']:
                    # better model, keep it
                    # first, reidentify using all the datapoints
                    best['fit'] = fm.ols(model_desc, data=self.df).fit()
                    best['cverror'] = cverror
                    better_model_found = True
                    best_x = x

            if better_model_found:
                self._list_of_fits.append(best['fit'])
                self.list_of_cverrors.append(best['cverror'])

            else:
                # if we did not find a better model, exit
                break

            # next iteration with the found exog removed
            all_model_terms_dict.pop(best_x)

        self._fit = self._list_of_fits[-1]


    def _prune(self, fit, p_max):
        """
        If the fit contains statistically insignificant parameters, remove them.
        Returns a pruned fit where all parameters have p-values of the t-statistic below p_max

        Parameters
        ----------
        fit: fm.ols fit object
            Can contain insignificant parameters
        p_max : float
            Maximum allowed probability of the t-statistic

        Returns
        -------
        fit: fm.ols fit object
            Won't contain any insignificant parameters

        """

        def remove_from_model_desc(x, model_desc):
            """
            Return a model_desc without x
            """

            rhs_termlist = []
            for t in model_desc.rhs_termlist:
                if not t.factors:
                    # intercept, add anyway
                    rhs_termlist.append(t)
                elif not x == t.factors[0]._varname:
                    # this is not the term with x
                    rhs_termlist.append(t)

            md = ModelDesc(model_desc.lhs_termlist, rhs_termlist)
            return md

        corrected_model_desc = ModelDesc(fit.model.formula.lhs_termlist[:], fit.model.formula.rhs_termlist[:])
        pars_to_prune = fit.pvalues.where(fit.pvalues > p_max).dropna().index.tolist()
        try:
            pars_to_prune.remove('Intercept')
        except:
            pass
        while pars_to_prune:
            corrected_model_desc = remove_from_model_desc(pars_to_prune[0], corrected_model_desc)
            fit = fm.ols(corrected_model_desc, data=self.df).fit()
            pars_to_prune = fit.pvalues.where(fit.pvalues > p_max).dropna().index.tolist()
            try:
                pars_to_prune.remove('Intercept')
            except:
                pass
        return fit

    @staticmethod
    def find_best_rsquared(list_of_fits):
        """Return the best fit, based on rsquared"""
        res = sorted(list_of_fits, key=lambda x: x.rsquared)
        return res[-1]

    @staticmethod
    def find_best_akaike(list_of_fits):
        """Return the best fit, based on Akaike information criterion"""
        res = sorted(list_of_fits, key=lambda x: x.aic)
        return res[0]

    @staticmethod
    def find_best_bic(list_of_fits):
        """Return the best fit, based on Akaike information criterion"""
        res = sorted(list_of_fits, key=lambda x: x.bic)
        return res[0]

    def _predict(self, fit, df):
        """
        Return a df with predictions and confidence interval

        Notes
        -----
        The df will contain the following columns:
        - 'predicted': the model output
        - 'interval_u', 'interval_l': upper and lower confidence bounds.

        The result will depend on the following attributes of self:
        confint : float (default=0.95)
            Confidence level for two-sided hypothesis
        allow_negative_predictions : bool (default=True)
            If False, correct negative predictions to zero (typically for energy consumption predictions)

        Parameters
        ----------
        fit : Statsmodels fit
        df : pandas DataFrame or None (default)
            If None, use self.df


        Returns
        -------
        df_res : pandas DataFrame
            Copy of df with additional columns 'predicted', 'interval_u' and 'interval_l'
        """

        # Add model results to data as column 'predictions'
        df_res = df.copy()
        if 'Intercept' in fit.model.exog_names:
            df_res['Intercept'] = 1.0
        df_res['predicted'] = fit.predict(df_res)
        if not self.allow_negative_predictions:
            df_res.loc[df_res['predicted'] < 0, 'predicted'] = 0

        prstd, interval_l, interval_u = wls_prediction_std(fit,
                                                           df_res[fit.model.exog_names],
                                                           alpha=1 - self.confint)
        df_res['interval_l'] = interval_l
        df_res['interval_u'] = interval_u

        if 'Intercept' in df_res:
            df_res.drop(labels=['Intercept'], axis=1, inplace=True)

        return df_res

    def add_prediction(self):
        """
        Add predictions and confidence interval to self.df
        self.df will contain the following columns:
        - 'predicted': the model output
        - 'interval_u', 'interval_l': upper and lower confidence bounds.

        Parameters
        ----------
        None, but the result depends on the following attributes of self:
        confint : float (default=0.95)
            Confidence level for two-sided hypothesis
        allow_negative_predictions : bool (default=True)
            If False, correct negative predictions to zero (typically for energy consumption predictions)

        Returns
        -------
        Nothing, adds columns to self.df
        """
        self.df = self._predict(fit=self.fit, df=self.df)

    def plot(self, model=True, bar_chart=True, **kwargs):
        """
        Plot measurements and predictions.

        By default, use self._fit and self.df, but both can be overruled by the arguments df and fit
        This function will detect if the data has been used for the modelling or not and will
        visualize them differently.

        Parameters
        ----------
        model : boolean, default=True
            If True, show the modified energy signature
        bar_chart : boolean, default=True
            If True, make a bar chart with predicted and measured data

        Other Parameters
        ----------------
        df : pandas Dataframe, default=None
            The data to be plotted.  If None, use self.df
            If the dataframe does not have a column 'predicted', a prediction will be made
        fit : statsmodels fit, default=None
            The model to be used.  if None, use self._fit

        Returns
        -------
        figures : List of plt.figure objects.

        """
        plot_style()
        figures = []
        fit = kwargs.get('fit', self.fit)
        df = kwargs.get('df', self.df)

        if not 'predicted' in df.columns:
            df = self._predict(fit=fit, df=df)
        # split the df in the auto-validation and prognosis part
        df_auto = df.loc[self.df.index[0]:self.df.index[-1]]
        if df_auto.empty:
            df_prog = df
        else:
            df_prog = df.loc[df_auto.index[-1]:].iloc[1:]

        if model:
            # The first variable in the formula is the most significant.  Use it as abcis for the plot
            try:
                exog1 = fit.model.exog_names[1]
            except IndexError:
                exog1 = self.list_of_x[0]

            # plot model as an adjusted trendline
            # get sorted model values
            dfmodel = df[[exog1, 'predicted', 'interval_u', 'interval_l']]
            dfmodel.index = dfmodel[exog1]
            dfmodel = dfmodel.sort_index()
            plt.plot(dfmodel.index, dfmodel['predicted'], '--', color='royalblue')
            plt.plot(dfmodel.index, dfmodel['interval_l'], ':', color='royalblue')
            plt.plot(dfmodel.index, dfmodel['interval_u'], ':', color='royalblue')
            # plot dots for the measurements
            if len(df_auto) > 0:
                plt.plot(df_auto[exog1], df_auto[self.y], 'o', mfc='orangered', mec='orangered', ms=8,
                         label='Data used for model fitting')
            if len(df_prog) > 0:
                plt.plot(df_prog[exog1], df_prog[self.y], 'o', mfc='seagreen', mec='seagreen', ms=8,
                         label='Data not used for model fitting')
            plt.title('rsquared={:.2f} - BIC={:.1f}'.format(fit.rsquared, fit.bic))
            plt.xlabel(exog1)
            figures.append(plt.gcf())

        if bar_chart:
            ind = np.arange(len(df.index))  # the x locations for the groups
            width = 0.35  # the width of the bars

            fig, ax = plt.subplots()
            title = 'Measured'  # will be appended based on the available data
            if len(df_auto) > 0:
                model = ax.bar(ind[:len(df_auto)], df_auto['predicted'], width * 2, color='#FDD787', ecolor='#FDD787',
                               yerr=df_auto['interval_u'] - df_auto['predicted'], label=self.y + ' modelled')
                title = title + ', modelled'
            if len(df_prog) > 0:
                prog = ax.bar(ind[len(df_auto):], df_prog['predicted'], width * 2, color='#6CD5A1', ecolor='#6CD5A1',
                              yerr=df_prog['interval_u'] - df_prog['predicted'], label=self.y + ' expected')
                title = title + ' and predicted'

            meas = ax.bar(ind, df[self.y], width, label=self.y + ' measured', color='#D5756C')
            # add some text for labels, title and axes ticks
            ax.set_title('{} {}'.format(title, self.y))
            ax.set_xticks(ind)
            ax.set_xticklabels([x.strftime('%d-%m-%Y') for x in df.index], rotation='vertical')
            ax.yaxis.grid(True)
            ax.xaxis.grid(False)

            plt.legend(ncol=3, loc='upper center')
            figures.append(plt.gcf())

        plt.show()

        return figures

    def _modeldesc_to_dict(self, md):
        """Return a string representation of a patsy ModelDesc object"""
        d = {'lhs_termlist': [md.lhs_termlist[0].factors[0].name()]}
        rhs_termlist = []

        # add other terms, if any
        for term in md.rhs_termlist[:]:
            if len(term.factors) == 0:
                # intercept, represent by empty string
                rhs_termlist.append('')
            else:
                rhs_termlist.append(term.factors[0].name())

        d['rhs_termlist'] = rhs_termlist
        return d

    def _modeldesc_from_dict(self, d):
        """Return a string representation of a patsy ModelDesc object"""
        lhs_termlist = [Term([LookupFactor(d['lhs_termlist'][0])])]
        rhs_termlist = []
        for name in d['rhs_termlist']:
            if name == '':
                rhs_termlist.append(Term([]))
            else:
                rhs_termlist.append(Term([LookupFactor(name)]))

        md = ModelDesc(lhs_termlist, rhs_termlist)
        return md

    def __getstate__(self):
        """
        Remove attributes that cannot be pickled and store as dict.

        Each fit has a model.formula which is a patsy ModelDesc and this cannot be pickled.
        We use our knowledge of this ModelDesc (as we build it up manually in the do_analysis() method)
        and decompose it into a dictionary.  This dictionary is stored in the list 'formulas',
        one dict per fit.

        Finally we have to remove each fit entirely (not just the formula), it is built-up again
        from self.formulas in the __setstate__ method.
        """
        d = self.__dict__
        d['formulas'] = []
        for fit in self._list_of_fits:
                d['formulas'].append(self._modeldesc_to_dict(fit.model.formula))
                #delattr(fit.model, 'formula')
        d.pop('_list_of_fits')
        d.pop('_fit')
        
        print("Pickling...  Removing the 'formula' from each fit.model.\n\
             You have to unpickle your object or run __setstate__(self.__dict__) to restore them.".format(d))
        return d

    def __setstate__(self, state):
        """Restore the attributes that cannot be pickled"""
        for k,v in state.items():
            if k is not 'formulas':
                setattr(self, k, v)
        self._list_of_fits = []
        for formula in state['formulas']:
            self._list_of_fits.append(fm.ols(self._modeldesc_from_dict(formula), data=self.df).fit())
        self._fit = self._list_of_fits[-1]

