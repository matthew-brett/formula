""" Parts of formulae: Terms, Factors, etc
"""
import numpy as np

import sympy

from .formulae import Formula
from .utils import to_str

from string import ascii_letters, digits

LETTERS_DIGITS = ascii_letters + digits

class Term(sympy.Symbol):
    """A sympy.Symbol type to represent a term an a regression model

    Terms can be added to other sympy expressions with the single
    convention that a term plus itself returns itself.

    It is meant to emulate something on the right hand side of a formula
    in R. In particular, its name can be the name of a field in a
    recarray used to create a design matrix.

    >>> t = Term('x')
    >>> xval = np.array([(3,),(4,),(5,)], np.dtype([('x', np.float)]))
    >>> f = t.formula
    >>> d = f.design(xval)
    >>> print d.dtype.descr
    [('x', '<f8')]
    >>> f.design(xval, return_float=True)
    array([ 3.,  4.,  5.])
    """
    # This flag is defined to avoid using isinstance in getterms
    # and getparams.
    _term_flag = True

    # To allow subclasses to use different formulae
    _formula_maker = Formula

    @property
    def formula(self):
        """
        Return a Formula with only terms=[self].
        """
        return self._formula_maker([self])


class FactorTerm(Term):
    """ Boolean Term derived from a Factor.

    Its properties are the same as a Term except that its product with
    itself is itself.
    """
    # This flag is defined to avoid using isinstance in getterms
    # and _poly_from_formula
    _factor_term_flag = True

    def __new__(cls, name, level):
        # Names or levels can be byte strings
        new = Term.__new__(cls, "%s_%s" % (to_str(name), to_str(level)))
        new.level = level
        new.factor_name = name
        return new

    def __mul__(self, other):

        if is_factor_term(other) and self.factor_name == other.factor_name:
            if self.level == other.level:
                return self
            else:
                return 0
        else:
            return sympy.Symbol.__mul__(self, other)


class Factor(object):
    """ A qualitative variable in a regression model

    A Factor is similar to R's factor. The levels of the Factor can be
    either strings or ints.
    """

    # This flag is defined to avoid using isinstance in getterms
    # and getparams.
    _factor_flag = True

    # To allow subclasses to use different formulae
    _formula_maker = Formula

    def __init__(self, name, levels, char='b',
                 coding='indicator', reference=None):
        """
        Parameters
        ----------
        name : str
        levels : [str or int]
            A sequence of strings or ints.
        char : str
        coding : one of ['main_effect', 'drop_reference', 'indicator']
        reference : element of levels, if None defaults to levels[0]

        Returns
        -------
        """
        # Check whether they can all be cast to strings or ints without
        # loss.
        levelsarr = np.asarray(levels)
        if levelsarr.ndim == 0 and levelsarr.dtype.kind in 'SOU':
            levelsarr = np.asarray(list(levels))
        if levelsarr.dtype.kind not in 'SOU': # the levels are not strings
            if not np.alltrue(np.equal(levelsarr, np.round(levelsarr))):
                raise ValueError('levels must be strings or ints')
            levelsarr = levelsarr.astype(np.int)
        elif levelsarr.dtype.kind == 'S': # Byte strings, convert
            levelsarr = levelsarr.astype('U')
        self.levels = list(levelsarr)
        self.name = name
        self._char = char

        if coding not in ['drop_reference',
                            'main_effect',
                            'indicator']:
            raise ValueError('coding must be one of %s' %
                             `['drop_reference',
                               'main_effect',
                               'indicator']`)

        self.coding = coding
        if reference is None:
            self.reference = self.levels[0]
        else:
            if reference not in self.levels:
                raise ValueError('reference should an element of levels')
            self.reference = reference

    def __getitem__(self, level):
        """
        self.get_term(level)
        """
        return self.get_term(level)

    def __repr__(self):
        return "Factor('%s', %s)" % (self.name, `self.levels`)

    def get_term(self, level):
        """
        Retrieve a term of the Factor...
        """
        if level not in self.levels:
            raise ValueError('level not found')
        return self.formula["%s_%s" % (self.name, str(level))]

    @property
    def terms(self):
        return [self.get_term(L) for L in self.levels]

    @property
    def main_effect(self):
        terms = list(self.indicator.terms)
        ref_term = self.get_term(self.reference)
        terms.remove(ref_term)
        return self._formula_maker([term - ref_term for term in terms])

    @property
    def drop_reference(self):
        """
        The drop_reference formula:
        a binary column for each level
        of the factor except self.reference.
        """
        terms = list(self.indicator.terms)
        ref_term = self.get_term(self.reference)
        terms.remove(ref_term)
        return self._formula_maker(terms)

    @property
    def indicator(self):
        """
        The indicator formula: a binary column for each level
        of the factor.
        """
        if not hasattr(self, "_indicator"):
            self._indicator = self._formula_maker(
                [FactorTerm(self.name, l) for l in self.levels], char=self._char)
        return self._indicator

    @property
    def formula(self):
        """
        Return the formula of the Factor = getattr(self, self.coding)
        """
        return getattr(self, self.coding)

    @staticmethod
    def fromcol(col, name):
        """ Create a Factor from a column array.

        Parameters
        ----------
        col : ndarray
            an array with ndim==1

        name : str
            name of the Factor

        Returns
        -------

        factor : Factor

        Examples
        --------
        >>> data = np.array([(3,'a'),(4,'a'),(5,'b'),(3,'b')], np.dtype([('x', np.float), ('y', 'S1')]))
        >>> f1 = Factor.fromcol(data['y'], 'y')
        >>> f2 = Factor.fromcol(data['x'], 'x')
        >>> d = f1.formula.design(data)
        >>> print d.dtype.descr
        [('y_a', '<f8'), ('y_b', '<f8')]
        >>> d = f2.formula.design(data)
        >>> print d.dtype.descr
        [('x_3', '<f8'), ('x_4', '<f8'), ('x_5', '<f8')]
        """
        col = np.asarray(col)
        if col.ndim != 1 or (col.dtype.names and len(col.dtype.names) > 1):
            raise ValueError('expecting an array that can be thought '
                             'of as a column or field of a recarray')
        levels = np.unique(col)
        if not col.dtype.names and not name:
            name = 'factor'
        elif col.dtype.names:
            name = col.dtype.names[0]
        return Factor(name, levels)

    def __mul__(self, other):
        """ Multiply factor by other formula-like object

        Parameters
        ----------
        other : object
            object that can be multiplied with a formula

        Returns
        -------
        f : ``Formula``
        """
        return self.formula * other


def is_term(obj):
    """ Is obj a Term?
    """
    return hasattr(obj, "_term_flag")


def is_factor_term(obj):
    """ Is obj a FactorTerm?
    """
    return hasattr(obj, "_factor_term_flag")


def is_factor(obj):
    """ Is obj a Factor?
    """
    return hasattr(obj, "_factor_flag")


def getparams(expression):
    """ Return the parameters of an expression that are not Term
    instances but are instances of sympy.Symbol.

    Examples
    --------
    >>> from formula import terms, Formula
    >>> x, y, z = terms('x, y, z')
    >>> f = Formula([x,y,z])
    >>> getparams(f)
    []
    >>> f.mean
    _b0*x + _b1*y + _b2*z
    >>> getparams(f.mean)
    [_b0, _b1, _b2]
    >>>
    >>> th = sympy.Symbol('theta')
    >>> f.mean*sympy.exp(th)
    (_b0*x + _b1*y + _b2*z)*exp(theta)
    >>> getparams(f.mean*sympy.exp(th))
    [theta, _b0, _b1, _b2]
    """
    atoms = set([])
    expression = np.array(expression)
    if expression.shape == ():
        expression = expression.reshape((1,))
    if expression.ndim > 1:
        expression = expression.reshape((np.product(expression.shape),))
    for term in expression:
        atoms = atoms.union(sympy.sympify(term).atoms())
    params = []
    for atom in atoms:
        if isinstance(atom, sympy.Symbol) and not is_term(atom):
            params.append(atom)
    params.sort()
    return params


def getterms(expression):
    """ Return the all instances of Term in an expression.

    Examples
    --------
    >>> from formula import terms, Formula
    >>> x, y, z = terms('x, y, z')
    >>> f = Formula([x,y,z])
    >>> getterms(f)
    [x, y, z]
    >>> getterms(f.mean)
    [x, y, z]
    """
    atoms = set([])
    expression = np.array(expression)
    if expression.shape == ():
        expression = expression.reshape((1,))
    if expression.ndim > 1:
        expression = expression.reshape((np.product(expression.shape),))
    for e in expression:
        atoms = atoms.union(e.atoms())
    terms = []
    for atom in atoms:
        if is_term(atom):
            terms.append(atom)
    terms.sort()
    return terms


def fromrec(recarr):
    """ Create Terms and Factors from structured array

    We assume fields of type object and string are Factors, all others are
    Terms.

    Parameters
    ----------
    recarr : ndarray
        array with composite dtype

    Returns
    -------
    facterms : dict
        dict with keys of ``recarr`` dtype field names, and values being a
        Factor (if the field was object or string type) and a Term otherwise.

    Examples
    --------
    >>> arr = np.array([(100,'blue'), (0, 'red')], dtype=
    ...                [('awesomeness','i'), ('shirt','S7')])
    >>> teams = fromrec(arr)
    >>> is_term(teams['awesomeness'])
    True
    >>> is_factor(teams['shirt'])
    True
    """
    result = {}
    for n, d in recarr.dtype.descr:
        if d[1] in 'SOU':
            result[n] = Factor(n, np.unique(recarr[n]))
        else:
            result[n] = Term(n)
    return result


def stratify(factor, variable):
    """ Create a new variable, stratified by the levels of a Factor.

    Parameters
    ----------
    variable : str or a simple sympy expression whose string representation
        are all lower or upper case letters, i.e. it can be interpreted
        as a name

    Returns
    -------
    formula : Formula
        Formula whose mean has one parameter named _variable%d, for each
        level in factor.levels

    Examples
    --------
    >>> f = Factor('a', ['x','y'])
    >>> sf = stratify(f, 'theta')
    >>> sf.mean
    _theta0*a_x + _theta1*a_y
    """
    if not set(str(variable)).issubset(LETTERS_DIGITS):
        raise ValueError('variable should be interpretable as a '
                         'name and not have anything but digits '
                         'and letters')
    variable = sympy.sympify(variable)
    # Formula class hard coded here; make stratify a method of Factor again?
    f = Formula(factor.formula.terms, char=variable)
    f.name = factor.name
    return f
