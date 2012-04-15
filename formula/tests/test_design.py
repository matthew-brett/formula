from os import remove
from tempfile import mkstemp
from string import uppercase
import csv

import numpy as np

import sympy

try:
    import rpy2.robjects
except ImportError:
    have_rpy2 = False
else:
    have_rpy2 = True

from ..parts import Term, Factor
from ..utils import rec_append_fields
from ..convenience import terms
from ..ancova import ANCOVA

import nose.tools as nt
from nose.plugins.skip import SkipTest
from numpy.testing import assert_array_almost_equal, assert_array_equal

# Some useful stuff for making test designs
X, Y, Z = terms('X, Y, Z')
F = Factor('F', ['a','b','c'])
G = Factor('G', ['aa','bb','cc'])
H = Factor('H', ['a','b','c','d','e','f','g','h','i','j'])


def setup_module():
    if not have_rpy2:
        raise SkipTest('Need rpy2 for design tests')


def _arr2csv(arr, fname):
    """ Very simple structured array csv writer """
    fobj = open(fname, 'wb')
    writer = csv.writer(fobj)
    writer.writerow(arr.dtype.names)
    for row in arr:
        writer.writerow(row)


def random_letters(size, nlevels=6):
    return [uppercase[i] for i in np.random.random_integers(0,nlevels-1,size=size)]


def random_subset(subset, size):
    s = sorted(subset)
    return [s[i] for i in np.random.random_integers(0,len(s)-1,size=size)]


def random_recarray(size):
    initial = np.empty(size, dtype=np.dtype([('Y', np.float)]))
    initial['Y'] = np.random.standard_normal(size)
    numeric_vars = [np.random.standard_normal(size) for _ in range(10)]
    categorical_vars = [random_letters(size, l) for l in [3,4,7,6,4,5,8]]
    inter = rec_append_fields(initial,
                              ['n%s' % l for l in uppercase[:10]],
                              numeric_vars)
    catvarchars = uppercase[:len(categorical_vars)]
    final = rec_append_fields(inter,
                              ['c%s' % l for l in catvarchars],
                              categorical_vars)
    return (final,
            sympy.symbols(['n%s' % l for l in uppercase[:10]]),
            [Factor('c%s' % s, np.unique(l)) for s, l in zip(catvarchars, categorical_vars)]
           )


def random_categorical_formula(size=500):
    nterms = np.random.poisson(5)
    X, n, c = random_recarray(size)
    d = []
    for _ in range(nterms):
        expr = np.product(np.unique(random_subset(n, np.random.binomial(len(n), 0.5))))
        factors = tuple(set(random_subset(c, np.random.poisson(1))))
        d.append((expr, factors))
    return ANCOVA(*tuple(set(d)))


def random_from_factor(factor, size):
    return random_subset(factor.levels, size)


def random_from_terms_factors(terms, factors, size):
    data = np.empty(size,
                    np.dtype([(str(terms[0]), np.float)]))
    data[str(terms[0])] = np.random.standard_normal(size)
    for t in terms[1:]:
        data = rec_append_fields(data, str(t), np.random.standard_normal(size))
    for f in factors:
        data = rec_append_fields(data, f.name, random_from_factor(f, size))
    return data


def random_from_categorical_formula(cf, size):
    exprs, factors = exprs_factors(cf)
    return random_from_terms_factors(exprs, factors, size)


def exprs_factors(anc):
    # Get expressions and factors from ANCOVA instance
    exprs = []
    factors = []
    for key, value in anc.graded_dict.items():
        if str(key) != '1':
            exprs += str(key).split("*")
        for order in value:
            for fs in value[order]:
                factors += list(fs)
    return list(set(exprs)), list(set(factors))


def test__arr2csv():
    # Test little csv utility against MPL
    try:
        from matplotlib.mlab import rec2csv
    except ImportError:
        raise SkipTest('Need matplotlib for test')
    # make test data
    size = 500
    X = random_from_categorical_formula(simple(), size)
    X = rec_append_fields(X, 'response', np.random.standard_normal(size))
    res = []
    for func in (_arr2csv, rec2csv):
        fh, fname = mkstemp()
        try:
            func(X, fname)
            res.append(np.recfromcsv(fname))
        finally:
            remove(fname)
    a2c, r2c = res
    for name in a2c.dtype.names:
        v1 = a2c[name]
        if v1.dtype.kind == 'S':
            assert_array_equal(v1, r2c[name])
        else:
            assert_array_almost_equal(a2c[name], r2c[name])


def simple():
    d = ANCOVA((X*Y,(G,F)),(X,(F,)),(X,[G,F]),(1,[G]),
                (Z,[H]),(Z,[H,G,F]), (X*Y*Z,[H]),(X*Y*Z,[F]))
    return d


def simple2():
    d = ANCOVA((X*Y,(G,F)),(X,(F,)),(X,[G,F]),(1,[G]),
                (Z,[H]),(Z,[H,G,F]), (X*Y*Z,[F]),(X*Y*Z,[H]))
    return d


def testR(d=None, size=500):
    if d is None:
        d = simple()
    X = random_from_categorical_formula(d, size)
    nR = get_r_coef_names(d, X)
    nt.assert_true('(Intercept)' in nR)
    nR.remove("(Intercept)")
    nF = [str(t).replace("_","").replace("*",":") for t in d.formula.terms]
    nR = sorted([sorted(n.split(":")) for n in nR])
    nt.assert_true('1' in nF)
    nF.remove('1')
    nF = sorted([sorted(n.split(":")) for n in nF])
    nt.assert_equal(nR, nF)
    return d, X, nR, nF


def get_r_coef_names(d, X):
    X = rec_append_fields(X, 'response',
                          np.random.standard_normal(len(X)))
    fh, fname = mkstemp()
    try:
        _arr2csv(X, fname)
        Rstr = '''
        data = read.table("%s", sep=',', header=T)
        cur.lm = lm(response ~ %s, data)
        COEF = coef(cur.lm)
        ''' % (fname, d.Rstr)
        rpy2.robjects.r(Rstr)
    finally:
        remove(fname)
    return list(np.array(rpy2.robjects.r("names(COEF)")))


def test_simplest():
    pass


def test2():
    testR(d=random_categorical_formula())


def test3():
    testR(d=simple2())
