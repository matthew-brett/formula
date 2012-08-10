""" Testing ancova module
"""

from os.path import join as pjoin, dirname

import numpy as np
# recfromcsv available from numpy 1.3.0
from numpy import recfromcsv

from ..parts import Term, Factor, fromrec
from ..ancova import ANCOVA, typeI, typeII, typeIII

from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_array_equal, dec)

from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_not_equal, assert_raises)

# To allow us to skip tests requring OLS
from ..models_compat import have_OLS


X = Term('X')
F = Factor('F', range(3))
H = Factor('H', range(2))
SALARY = recfromcsv(pjoin(dirname(__file__), '..', 'data', 'salary.csv'))

def test_init():
    a0 = ANCOVA((X,F), (X,(F,H)))
    assert_equal(a0.sequence(),
                 [(1,()), (X,(F,)), (X, (F,H))])


def test_delete_terms():
    a0 = ANCOVA((X,F), (X,(F,H)))
    a1 = a0.delete_terms((X,F))
    assert_equal(a1.sequence(),
                 [(1,()), (X, (F,H))])


def test_eq_neq():
    f1 = ANCOVA(1, F, H, (X, F))
    f2 = ANCOVA(1, F, H, (X, F))
    assert_equal(f1, f2)
    f3 = ANCOVA(1, F, (X, F))
    assert_not_equal(f1, f3)


def test_smoke():
    # smoke test, more or less
    terms = fromrec(SALARY)
    f = ANCOVA(1, terms['e'],terms['p'],(1,(terms['e'],terms['p'])))
    ANCOVA.add_intercept = False
    f2 = ANCOVA(terms['e'],(1,(terms['e'],terms['p'])))
    ANCOVA.add_intercept = True
    f3 = ANCOVA((1,(terms['e'],terms['p'])))


def test_multiply_by_factor():
    terms = fromrec(SALARY)
    f = ANCOVA(1, terms['e'])
    f2 = f.multiply_by_factor(terms['p'])
    assert_equal(ANCOVA(1, terms['p'], (1,(terms['e'], terms['p']))), f2)


@dec.skipif(not have_OLS)
def test_types123():
    # type 1, 2, and 3 sum of squares.
    """ Tested against R thus:
    SALARY = read.csv('formula/data/salary.csv')
    fit = lm(S ~ X:E + X:P + X:P:E, data=SALARY)
    typeI = anova(fit)
    typeII = drop1(fit, test="F")
    fit_3 = lm(S ~ X:E + X:P, data=SALARY)
    typeIII = anova(fit_3, fit)
    """
    terms = fromrec(SALARY)
    x = terms['x']; e = terms['e']; p = terms['p']
    ancova = ANCOVA((x,e),(x,p),(x,(p,e)))
    res1 = typeI('s', ancova, SALARY)
    """ > as.matrix(typeI)
            Df    Sum Sq   Mean Sq   F value       Pr(>F)
    X:E        3 498268907 166089636 35.520846 3.076673e-11
    X:P        1 295195692 295195692 63.132180 1.119036e-09
    X:E:P      2  25275391  12637696  2.702767 7.956761e-02
    Residuals 39 182357587   4675836        NA           NA
    """
    exp_vals = map(float,
                   """3 498268907 166089636 35.520846 3.076673e-11
                   1 295195692 295195692 63.132180 1.119036e-09
                   2  25275391  12637696  2.702767 7.956761e-02
                   39 182357587   4675836 NaN NaN""".split())
    exp_vals = np.array(exp_vals).reshape(4, -1)
    assert_array_equal(res1['df'][1:], exp_vals[:, 0])
    res1bits = res1[['SS', 'MS', 'F', 'p_value']].view(float).reshape(-1, 4)
    assert_almost_equal(np.nan_to_num(res1bits), np.nan_to_num(exp_vals[1:, :]))
    res2 = typeII('s', ancova, SALARY)
    """ > as.matrix(typeII)
        Df Sum of Sq       RSS      AIC  F value      Pr(F)
    <none> NA        NA 182357587 712.8706       NA         NA
    X:E:P   2  25275391 207632978 714.8415 2.702767 0.07956761
    """
    res2row = res2[-2:-1]
    assert_equal(res2row['df'], 2)
    res2bits = res2row[['SS', 'F', 'p_value']].view(float)
    assert_almost_equal(res2bits, [25275391, 2.702767, 0.07956761])
    res3 = typeIII('s', ancova, SALARY)
    """ > as.matrix(typeIII)
    Res.Df       RSS Df Sum of Sq        F     Pr(>F)
    1     41 207632978 NA        NA       NA         NA
    2     39 182357587  2  25275391 2.702767 0.07956761
    """
    # Sum of Sq does not correspond?
    res3row = res3[-2:-1]
    assert_equal(res3row['df'], 2)
    res3bits = res3row[['F', 'p_value']].view(float)
    assert_almost_equal(res3bits, [2.702767, 0.07956761])
    # Reversing the order changes the ANOVA tables, in particular
    # the degrees of freedom associated to each contrast. This is
    # because the codings change when the order of the factors change.
    ancova2 = ANCOVA((x,p),(x,e), (x,(p,e)))
    rres2 = typeII('s', ancova2, SALARY)
    rres3 = typeIII('s', ancova2, SALARY)
