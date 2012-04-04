""" Sympy 0.6.7 - 0.7.0 compatibility layer

There were some API changes in sympy that need working round here.

In particular, Jonathan's implemented_function got into 0.7.0, but we have to
carry our own copy until then.
"""

from distutils.version import LooseVersion

import sympy

SYMPY_0p6 = LooseVersion(sympy.__version__) < LooseVersion('0.7.0')

from .fixes.sympy.utilities.lambdify import (implemented_function, lambdify)

def make_dummy(name):
    """ Make dummy variable of given name

    Parameters
    ----------
    name : str
        name of dummy variable

    Returns
    -------
    dum : `Dummy` instance

    Notes
    -----
    The interface to Dummy changed between 0.6.7 and 0.7.0
    """
    if SYMPY_0p6:
        return sympy.Symbol(name, dummy=True)
    return sympy.Dummy(name)
