# -*- coding: utf-8 -*-
"""
General analysis functions.

Try to write all methods such that they take a dataframe as input
and return a dataframe or list of dataframes.
"""

# pylint: disable=E0611
from patsy import ModelDesc, Term, LookupFactor
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.formula.api as fm
import numpy as np

from matplotlib import pyplot


from .analysis import Analysis
from .plotting import plot_style


def dict_to_model_desc(dictionary):
    """Return a string representation of a patsy ModelDesc object"""
    lhs_termlist = [Term([LookupFactor(dictionary['lhs_termlist'][0])])]
    rhs_termlist = []
    for name in dictionary['rhs_termlist']:
        if name == '':
            rhs_termlist.append(Term([]))
        else:
            rhs_termlist.append(Term([LookupFactor(name)]))

    return ModelDesc(lhs_termlist, rhs_termlist)


def model_desc_to_dict(model_desc):
    """Return a string representation of a patsy ModelDesc object"""
    result = {
        'lhs_termlist': [model_desc.lhs_termlist[0].factors[0].name()]
    }
    rhs_termlist = []

    # add other terms, if any
    for term in model_desc.rhs_termlist[:]:
        if len(term.factors) == 0:
            # intercept, represent by empty string
            rhs_termlist.append('')
        else:
            rhs_termlist.append(term.factors[0].name())

    result['rhs_termlist'] = rhs_termlist
    return result


class MultiVarLinReg(Analysis):
    """
    Multi-variable linear regression based on statsmodels and Ordinary Least Squares (ols)

    Pass a dataframe with the variable to be modelled y (dependent variable) and
    the possible independent variables x. Specify as string the name of the dependent variable,
    and optionally pass a list with names of independent variables to try (by default all other
    columns will be tried as independent variables).

    The analysis is based on a forward-selection approach: starting from a simple model,
    the model is iteratively refined and verified until no statistical relevant improvements
    can be obtained.  Each model in the iteration loop is stored in the attribute self.list_of_fits.
    The selected model is self.fit (=pointer to the last element of self.list_of_fits).

    The dataframe can contain daily, weekly, monthly, yearly ... values.  Each row is an instance.


    Examples
    --------

    >> mvlr = MultiVarLinReg(data_frame, 'gas', p_max=0.04)
    >> mvlr = MultiVarLinReg(data_frame, 'gas', list_of_x=[ 'heatingDegreeDays14',
                                                    'GlobalHorizontalIrradiance',
                                                    'WindSpeed' ])


    """

    def __init__(self,
                 data_frame,
                 dependent_var,
                 options=None):
        """

        Parameters
        ----------
        data_frame : pd.DataFrame
            Datetimeindex and both independent variables (x) and dependent variable (y) as columns
        dependent_var : str
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
            Only possible if the data_frame has less than 15 entries.
            Note : this will take much longer computation times!
        negative_predictions : bool, default=False
            If True, allow predictions to be negative.
            For gas consumption or PV production, this is not physical so negative_predictions
            should be False.
        """

        super().__init__(data_frame=data_frame)

        assert_error = "%s not a column in the dataframe" % dependent_var
        assert dependent_var in self.data_frame.columns, assert_error

        self.dependent_var = dependent_var

        options = options if options else {}

        self.p_max = options.pop('p_max', 0.05)
        self.confint = options.pop('confint', 0.95)
        self.list_of_x = options.pop(
            'list_of_x', self.data_frame.columns.tolist())
        self.cross_validation = options.pop('cross_validation', False)
        self.negative_predictions = options.pop('negative_predictions', False)

        try:
            self.list_of_x.remove(self.dependent_var)
        except ValueError:
            pass

        self._fit = None
        self._list_of_fits = []
        self.list_of_cverrors = []

    @property
    def fit(self):
        """ TODO docstring """
        if self._fit is None:
            raise UnboundLocalError(
                'Run "do_analysis()" first to fit a model to the data.')
        return self._fit

    @property
    def list_of_fits(self):
        """ TODO docstring """
        if not self._list_of_fits:
            raise UnboundLocalError(
                'Run "do_analysis()" first to fit a model to the data.')
        return self._list_of_fits

    def do_analysis(self):
        """
        Find the best model (fit) and create self.list_of_fits and self.fit
        """
        if self.cross_validation:
            return self._do_analysis_cross_validation()
        return self._do_analysis_no_cross_validation()

    def _do_analysis_no_cross_validation(self):
        """
        Find the best model (fit) and create self.list_of_fits and self.fit
        """

        # first model is just the mean
        response_term = [Term([LookupFactor(self.dependent_var)])]
        model_terms = [Term([])]  # empty term is the intercept
        all_model_terms_dict = {x: Term([LookupFactor(x)])
                                for x in self.list_of_x}
        # ...then add another term for each candidate
        # model_terms += [Term([LookupFactor(c)]) for c in candidates]
        model_desc = ModelDesc(response_term, model_terms)
        self._list_of_fits.append(
            fm.ols(model_desc, data=self.data_frame).fit())
        # try to improve the model until no improvements can be found

        while all_model_terms_dict:
            # try each value and overwrite the best_fit if we find a better one
            # the first best_fit is the one from the previous round
            ref_fit = self._list_of_fits[-1]
            best_fit = self._list_of_fits[-1]
            best_bic = best_fit.bic
            for value, term in all_model_terms_dict.items():
                # make new_fit, compare with best found so far
                model_desc = ModelDesc(
                    response_term, ref_fit.model.formula.rhs_termlist + [term])
                fit = fm.ols(model_desc, data=self.data_frame).fit()
                if fit.bic < best_bic:
                    best_bic = fit.bic
                    best_fit = fit
                    best_val = value
            # Sometimes, the obtained fit may be better, but contains unsignificant parameters.
            # Correct the fit by removing the unsignificant parameters and estimate again
            best_fit = self._prune(best_fit, p_max=self.p_max)

            # if best_fit does not contain more variables than ref fit, exit
            if len(best_fit.model.formula.rhs_termlist) == len(ref_fit.model.formula.rhs_termlist):
                break

            self._list_of_fits.append(best_fit)
            all_model_terms_dict.pop(best_val)
        self._fit = self._list_of_fits[-1]

    def _do_analysis_cross_validation(self):
        """
        Find the best model (fit) based on cross-valiation (leave one out)
        """
        assert len(self.data_frame) < 15, "Minimum 15 datapoints"

        # initialization: first model is the mean, but compute cv correctly.
        errors = []
        response_term = [Term([LookupFactor(self.dependent_var)])]
        model_desc = ModelDesc(response_term, [Term([])])
        for i in self.data_frame.index:
            # make new_fit, compute cross-validation and store error
            data_frame_ = self.data_frame.drop(i, axis=0)
            fit = fm.ols(model_desc, data=data_frame_).fit()
            cross_prediction = self._predict(
                fit=fit, data_frame=self.data_frame.loc[[i], :])
            errors.append(
                cross_prediction['predicted'] - cross_prediction[self.dependent_var])

        self._list_of_fits = [fm.ols(model_desc, data=self.data_frame).fit()]
        self.list_of_cverrors = [np.mean(np.abs(np.array(errors)))]

        # try to improve the model until no improvements can be found
        all_model_terms_dict = {x: Term([LookupFactor(x)])
                                for x in self.list_of_x}
        while all_model_terms_dict:
            # import pdb;pdb.set_trace()
            # try each x in all_exog and overwrite if we find a better one
            # at the end of iteration (and not earlier), save the best of the iteration
            better_model_found = False
            best = {
                "fit": self._list_of_fits[-1],
                "cverror": self.list_of_cverrors[-1]
            }
            for value, term in all_model_terms_dict.items():
                model_desc = ModelDesc(
                    response_term, self._list_of_fits[-1].model.formula.rhs_termlist + [term])
                # cross_validation, currently only implemented for monthly data
                # compute the mean error for a given formula based on leave-one-out.
                errors = []
                for i in self.data_frame.index:
                    # make new_fit, compute cross-validation and store error
                    data_frame_ = self.data_frame.drop(i, axis=0)
                    fit = fm.ols(model_desc, data=data_frame_).fit()
                    cross_prediction = self._predict(
                        fit=fit, data_frame=self.data_frame.loc[[i], :])
                    errors.append(
                        cross_prediction['predicted'] - cross_prediction[self.dependent_var])
                cverror = np.mean(np.abs(np.array(errors)))
                # compare the model with the current fit
                if cverror < best['cverror']:
                    # better model, keep it
                    # first, reidentify using all the datapoints
                    best['fit'] = fm.ols(
                        model_desc, data=self.data_frame).fit()
                    best['cverror'] = cverror
                    better_model_found = True
                    best_val = value

            if better_model_found:
                self._list_of_fits.append(best['fit'])
                self.list_of_cverrors.append(best['cverror'])

            else:
                # if we did not find a better model, exit
                break

            # next iteration with the found exog removed
            all_model_terms_dict.pop(best_val)

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

        def remove_from_model_desc(varname, model_desc):
            """
            Return a model_desc without varname
            """

            rhs_termlist = []
            for term in model_desc.rhs_termlist:
                if not term.factors:
                    # intercept, add anyway
                    rhs_termlist.append(term)
                elif not varname == term.factors[0]._varname:
                    # this is not the term with x
                    rhs_termlist.append(term)

            return ModelDesc(model_desc.lhs_termlist, rhs_termlist)

        corrected_model_desc = ModelDesc(
            fit.model.formula.lhs_termlist[:], fit.model.formula.rhs_termlist[:])
        pars_to_prune = fit.pvalues.where(
            fit.pvalues > p_max).dropna().index.tolist()
        pars_to_prune.remove('Intercept')
        while pars_to_prune:
            corrected_model_desc = remove_from_model_desc(
                pars_to_prune[0], corrected_model_desc)
            fit = fm.ols(corrected_model_desc, data=self.data_frame).fit()
            pars_to_prune = fit.pvalues.where(
                fit.pvalues > p_max).dropna().index.tolist()
            pars_to_prune.remove('Intercept')
        return fit

    @ staticmethod
    def find_best_rsquared(list_of_fits):
        """Return the best fit, based on rsquared"""
        res = sorted(list_of_fits, key=lambda x: x.rsquared)
        return res[-1]

    @ staticmethod
    def find_best_akaike(list_of_fits):
        """Return the best fit, based on Akaike information criterion"""
        res = sorted(list_of_fits, key=lambda x: x.aic)
        return res[0]

    @ staticmethod
    def find_best_bic(list_of_fits):
        """Return the best fit, based on Akaike information criterion"""
        res = sorted(list_of_fits, key=lambda x: x.bic)
        return res[0]

    def _predict(self, fit, data_frame):
        """
        Return a data_frame with predictions and confidence interval

        Notes
        -----
        The data_frame will contain the following columns:
        - 'predicted': the model output
        - 'interval_u', 'interval_l': upper and lower confidence bounds.

        The result will depend on the following attributes of self:
        confint : float (default=0.95)
            Confidence level for two-sided hypothesis
        negative_predictions : bool (default=True)
            If False, correct negative predictions to zero
            (typically for energy consumption predictions)

        Parameters
        ----------
        fit : Statsmodels fit
        data_frame : pandas DataFrame or None (default)
            If None, use self.data_frame


        Returns
        -------
        data_frame_res : pandas DataFrame
            Copy of data_frame with additional columns 'predicted', 'interval_u' and 'interval_l'
        """

        # Add model results to data as column 'predictions'
        data_frame_res = data_frame.copy()
        if 'Intercept' in fit.model.exog_names:
            data_frame_res['Intercept'] = 1.0
        data_frame_res['predicted'] = fit.predict(data_frame_res)
        if not self.negative_predictions:
            data_frame_res.loc[data_frame_res['predicted']
                               < 0, 'predicted'] = 0

        _, interval_l, interval_u = wls_prediction_std(fit,
                                                       data_frame_res[fit.model.exog_names],
                                                       alpha=1 - self.confint)
        data_frame_res['interval_l'] = interval_l
        data_frame_res['interval_u'] = interval_u

        if 'Intercept' in data_frame_res:
            data_frame_res.drop(
                labels=['Intercept'], axis=1, inplace=True)

        return data_frame_res

    def add_prediction(self):
        """
        Add predictions and confidence interval to self.data_frame
        self.data_frame will contain the following columns:
        - 'predicted': the model output
        - 'interval_u', 'interval_l': upper and lower confidence bounds.

        Parameters
        ----------
        None, but the result depends on the following attributes of self:
        confint : float (default=0.95)
            Confidence level for two-sided hypothesis
        negative_predictions : bool (default=True)
            If False, correct negative predictions to zero
            (typically for energy consumption predictions)

        Returns
        -------
        Nothing, adds columns to self.data_frame
        """
        self.data_frame = self._predict(fit=self.fit,
                                        data_frame=self.data_frame)

    def plot(self,
             model=True,
             bar_chart=True,
             **kwargs):
        """
        Plot measurements and predictions.

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
        data_frame : pandas Dataframe, default=None
            The data to be plotted.  If None, use self.data_frame
            If the dataframe does not have a column 'predicted', a prediction will be made
        fit : statsmodels fit, default=None
            The model to be used.  if None, use self._fit

        Returns
        -------
        figures : List of pyplot.figure objects.

        """
        plot_style()

        figures = []
        fit = kwargs.get('fit', self.fit)
        data_frame = kwargs.get('data_frame', self.data_frame)

        if 'predicted' not in data_frame.columns:
            data_frame = self._predict(fit=fit, data_frame=data_frame)
        # split the data_frame in the auto-validation and prognosis part
        data_frame_auto = data_frame.loc[self.data_frame.index[0]:self.data_frame.index[-1]]
        if data_frame_auto.empty:
            data_frame_prog = data_frame
        else:
            data_frame_prog = data_frame.loc[data_frame_auto.index[-1]:].iloc[1:]

        if model:
            # The first variable in the formula is the most significant.
            # Use it as abcis for the plot
            try:
                exog1 = fit.model.exog_names[1]
            except IndexError:
                exog1 = self.list_of_x[0]

            # plot model as an adjusted trendline
            # get sorted model values
            data_framemodel = data_frame[[
                exog1, 'predicted', 'interval_u', 'interval_l']]
            data_framemodel.index = data_framemodel[exog1]
            data_framemodel = data_framemodel.sort_index()
            pyplot.plot(data_framemodel.index,
                        data_framemodel['predicted'], '--', color='royalblue')
            pyplot.plot(data_framemodel.index,
                        data_framemodel['interval_l'], ':', color='royalblue')
            pyplot.plot(data_framemodel.index,
                        data_framemodel['interval_u'], ':', color='royalblue')
            # plot dots for the measurements
            if len(data_frame_auto) > 0:
                pyplot.plot(data_frame_auto[exog1],
                            data_frame_auto[self.dependent_var],
                            'o',
                            mfc='orangered',
                            mec='orangered',
                            ms=8,
                            label='Data used for model fitting')
            if len(data_frame_prog) > 0:
                pyplot.plot(data_frame_prog[exog1],
                            data_frame_prog[self.dependent_var],
                            'o',
                            mfc='seagreen',
                            mec='seagreen',
                            ms=8,
                            label='Data not used for model fitting')
            pyplot.title(
                'rsquared={:.2f} - BIC={:.1f}'.format(fit.rsquared, fit.bic))
            pyplot.xlabel(exog1)
            figures.append(pyplot.gcf())

        if bar_chart:
            # the x locations for the groups
            ind = np.arange(len(data_frame.index))
            width = 0.35  # the width of the bars

            _fig, axes = pyplot.subplots()
            title = 'Measured'  # will be appended based on the available data
            if len(data_frame_auto) > 0:
                model = axes.bar(ind[:len(data_frame_auto)],
                                 data_frame_auto['predicted'],
                                 width * 2,
                                 color='#FDD787',
                                 ecolor='#FDD787',
                                 yerr=data_frame_auto['interval_u'] -
                                 data_frame_auto['predicted'],
                                 label=self.dependent_var + ' modelled')
                title = title + ', modelled'
            if len(data_frame_prog) > 0:
                axes.bar(ind[len(data_frame_auto):],
                         data_frame_prog['predicted'],
                         width * 2,
                         color='#6CD5A1',
                         ecolor='#6CD5A1',
                         yerr=data_frame_prog['interval_u'] -
                         data_frame_prog['predicted'],
                         label=self.dependent_var + ' expected')
                title = title + ' and predicted'

            axes.bar(ind, data_frame[self.dependent_var], width,
                     label=self.dependent_var + ' measured', color='#D5756C')
            # add some text for labels, title and axes ticks
            axes.set_title('{} {}'.format(title, self.dependent_var))
            axes.set_xticks(ind)
            axes.set_xticklabels([x.strftime('%d-%m-%Y')
                                  for x in data_frame.index], rotation='vertical')
            axes.yaxis.grid(True)
            axes.xaxis.grid(False)

            pyplot.legend(ncol=3, loc='upper center')
            figures.append(pyplot.gcf())

        pyplot.show()

        return figures

    def __getstate__(self):
        """
        Remove attributes that cannot be pickled and store as dict.

        Each fit has a model.formula which is a patsy ModelDesc and this cannot be pickled.
        We use our knowledge of this ModelDesc (as we built it manually in the do_analysis() method)
        and decompose it into a dictionary.  This dictionary is stored in the list 'formulas',
        one dict per fit.

        Finally we have to remove each fit entirely (not just the formula), it is built-up again
        from self.formulas in the __setstate__ method.
        """
        dictionary = self.__dict__
        dictionary['formulas'] = []
        for fit in self._list_of_fits:
            dictionary['formulas'].append(
                model_desc_to_dict(fit.model.formula))
        dictionary.pop('_list_of_fits')
        dictionary.pop('_fit')

        print("Pickling...  Removing the 'formula' from each fit.model.\n\
             You have to unpickle your object or run __setstate__(self.__dict__) to restore them.")

        return dictionary

    def __setstate__(self, state):
        """Restore the attributes that cannot be pickled"""
        for key, value in state.items():
            if key != 'formulas':
                setattr(self, key, value)
        self._list_of_fits = []
        for formula in state['formulas']:
            self._list_of_fits.append(fm.ols(formula=dict_to_model_desc(formula),
                                             data=self.data_frame).fit())
        self._fit = self._list_of_fits[-1]
