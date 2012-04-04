""" Compatibility layer for statsmodels import

Use nipy version if available

If not available, use statsmodels version.

statsmodels import for OLS has changed over versions.
"""
import numpy as np

def get_OLS():
    """ Get OLS implementation from nipy or statsmodels """
    try:
        from nipy.algorithms.statistics.models.regression import OLSModel
    except ImportError:
        pass
    else:
        # Statsmodels compatible wrapper around nipy OLS
        from scipy.stats import f as fdist

        class OLSResult(object):
            def __init__(self, results):
                self.results = results
                self.resid = results.resid
                self.df_resid = results.df_resid
                self.scale = 1.0
            def f_test(self, *args, **kwargs):
                r = self.results.Fcontrast(*args, **kwargs)
                r.fvalue = r.F
                r.pvalue = fdist.sf(r.F, r.df_num, r.df_den)
                return r

        class OLS(OLSModel):
            def __init__(self, Y, X):
                self._Y = Y
                X = np.asarray(X)
                if X.ndim == 1:
                    X = X[:,None]
                super(OLS, self).__init__(X)
            def fit(self):
                return OLSResult(super(OLS, self).fit(self._Y))

        return OLS
    # Statsmodels
    try:
        from statsmodels.api import OLS
        return OLS
    except ImportError:
        pass
    try: # statsmodels 0.3.0
        from scikits.statsmodels.api import OLS
        return OLS
    except ImportError:
        pass
    try: # statsmodels 0.2.0
        from scikits.statsmodels import OLS
        return OLS
    except ImportError: # no statsmodels
        pass


OLS = get_OLS()
have_OLS = not OLS is None
if not have_OLS:
    def OLS(*args, **kwargs):
        raise RuntimeError('Need nipy or statsmodels OLS')
