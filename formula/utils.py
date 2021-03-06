""" Utility routines
"""
import sys
from itertools import combinations

import numpy as np
import numpy.ma as ma

try:
    # matrix_rank in numpy >= 1.5.0
    from numpy.linalg import matrix_rank as rank
except ImportError:
    from numpy.linalg import svd
    def rank(M, tol=None):
        """
        Return matrix rank of array using SVD method

        Rank of the array is the number of SVD singular values of the
        array that are greater than `tol`.

        Parameters
        ----------
        M : array_like
            array of <=2 dimensions
        tol : {None, float}
            threshold below which SVD values are considered zero. If `tol` is
            None, and ``S`` is an array with singular values for `M`, and
            ``eps`` is the epsilon value for datatype of ``S``, then `tol` is
            set to ``S.max() * eps``.
        """
        M = np.asarray(M)
        if M.ndim > 2:
            raise TypeError('array should have 2 or fewer dimensions')
        if M.ndim < 2:
            return int(not all(M==0))
        S = svd(M, compute_uv=False)
        if tol is None:
            tol = S.max() * np.finfo(S.dtype).eps
        return sum(S > tol)


def fullrank(X, r=None):
    """ Return a matrix whose column span is the same as X
    using an SVD decomposition.

    If the rank of X is known it can be specified by r-- no check is
    made to ensure that this really is the rank of X.
    """

    if r is None:
        r = rank(X)

    V, D, U = np.linalg.svd(X, full_matrices=0)
    order = np.argsort(D)
    order = order[::-1]
    value = []
    for i in range(r):
        value.append(V[:,order[i]])
    return np.asarray(np.transpose(value)).astype(np.float64)


def contrast_from_cols_or_rows(L, D, pseudo=None):
    """ Construct a contrast matrix from a design matrix D
    
    (possibly with its pseudo inverse already computed)
    and a matrix L that either specifies something in
    the column space of D or the row space of D.

    Parameters
    ----------
    L : ndarray
       Matrix used to try and construct a contrast.
    D : ndarray
       Design matrix used to create the contrast.
    pseudo : None or array-like, optional
       If not None, gives pseudo-inverse of `D`.  Allows you to pass
       this if it is already calculated. 
       
    Returns
    -------
    C : ndarray
       Matrix with C.shape[1] == D.shape[1] representing an estimable
       contrast.

    Notes
    -----
    From an n x p design matrix D and a matrix L, tries to determine a p
    x q contrast matrix C which determines a contrast of full rank,
    i.e. the n x q matrix

    dot(transpose(C), pinv(D))

    is full rank.

    L must satisfy either L.shape[0] == n or L.shape[1] == p.

    If L.shape[0] == n, then L is thought of as representing
    columns in the column space of D.

    If L.shape[1] == p, then L is thought of as what is known
    as a contrast matrix. In this case, this function returns an estimable
    contrast corresponding to the dot(D, L.T)

    This always produces a meaningful contrast, not always
    with the intended properties because q is always non-zero unless
    L is identically 0. That is, it produces a contrast that spans
    the column space of L (after projection onto the column space of D).
    """
    L = np.asarray(L)
    D = np.asarray(D)
    
    n, p = D.shape

    if L.shape[0] != n and L.shape[1] != p:
        raise ValueError, 'shape of L and D mismatched'

    if pseudo is None:
        pseudo = pinv(D)

    if L.shape[0] == n:
        C = np.dot(pseudo, L).T
    else:
        C = np.dot(pseudo, np.dot(D, L.T)).T
        
    Lp = np.dot(D, C.T)

    if len(Lp.shape) == 1:
        Lp.shape = (n, 1)
        
    if rank(Lp) != Lp.shape[1]:
        Lp = fullrank(Lp)
        C = np.dot(pseudo, Lp).T

    return np.squeeze(C)


def simplicial_complex(*simplices):
    """ Return faces and facets from simplicial complex of `simplices`

    Take a list of simplices and compute the simplicial complex generated by
    these simplices, returning a dict of the faces, the facets (maximal faces)
    of this complex, and list of all faces.

    Parameters
    ----------
    \*simplices : sequence
        simplices from which to create complex.  Simplices are sequence of
        points.

    Returns
    -------
    faces : dict
        Faces at levels n = 0 .. N where N is the largest length of passed
        simplices
    facets : list
        faces that are not subsets of any other faces.  Also called the maximal
        faces.
    faces_list : list
        The faces as a list rather than a dict, as in ``sum([f for f in
        faces.values()], [])``

    Examples
    --------
    >>> faces, facets, all = simplicial_complex(('a', 'b', 'c'), ('a', 'd'))
    >>> faces == {0: set(),
    ...           1: set([('a',), ('b',), ('c',), ('d',)]),
    ...           2: set([('a', 'b'), ('b', 'c'), ('a', 'c'), ('a', 'd')]),
    ...           3: set([('a', 'b', 'c')])}
    True
    >>> facets
    [('a', 'd'), ('a', 'b', 'c')]
    >>> set(all) == set([tuple(L) for L in 'abcd'] +
    ...                 [('a', 'b'), ('b', 'c'), ('a', 'c'), ('a', 'd')] +
    ...                 [('a', 'b', 'c')])
    True

    Notes
    -----
    See:

    http://en.wikipedia.org/wiki/Simplicial_complex
    http://en.wikipedia.org/wiki/Simplices
    http://en.wikipedia.org/wiki/Abstract_simplicial_complex

    The maximal faces of a simplicial complex (i.e., faces that are not subsets
    of any other faces) are called facets of the complex.
    """
    # The empty set is a face of every simplex
    faces = {0: set([])}
    if len(simplices) == 0:
        return faces, [], []
    lengths = [len(list(x)) for x in simplices]
    lmax = max(lengths)
    for i in range(lmax):
        faces[i+1] = set([])
    for simplex in simplices:
        simplex = sorted(simplex)
        for face_no in range(1, len(simplex)+1):
            for v in combinations(simplex, face_no):
                faces[face_no].add(v)
    # now determine the maximal faces (faces not subset of other faces)
    # The largest complex is not a subset of the smaller
    maximal = list(faces[lmax])
    # Descending over faces of decreasing size, starting at next lowest size
    for i in sorted(faces.keys())[-2::-1]:
        # Add face to maximal if not already present
        for simplex in faces[i]:
            keep = True
            for msimplex in maximal:
                if set(simplex).issubset(msimplex):
                    keep = False
            if keep:
                maximal.append(simplex)
    face_list = sum([list(f) for f in faces.values()], [])
    return faces, maximal[::-1], face_list


def factor_codings(*factor_monomials):
    """ Find which factors to code with indicator or contrast variables

    Determine which factors to code with indicator variables (using
    len(factor.levels) columns of 0s and 1s) or contrast coding (using
    len(factor.levels)-1).  The elements of the sequence should be tuples of
    strings.  Further, the factors are assumed to be in *graded* order, that is
    [len(f) for f in factor_monomials] is assumed non-decreasing.

    Examples
    --------
    >>> factor_codings(('b',), ('a',), ('b', 'c'), ('a','b','c'))
    {('b', 'c'): [('b', 'indicator'), ('c', 'contrast')], ('a',): [('a', 'contrast')], ('b',): [('b', 'indicator')], ('a', 'b', 'c'): [('a', 'contrast'), ('b', 'indicator'), ('c', 'indicator')]}
    >>> factor_codings(('a',), ('b',), ('b', 'c'), ('a','b','c'))
    {('b', 'c'): [('b', 'indicator'), ('c', 'contrast')], ('a',): [('a', 'indicator')], ('b',): [('b', 'contrast')], ('a', 'b', 'c'): [('a', 'contrast'), ('b', 'indicator'), ('c', 'indicator')]}

    Here is a version with debug strings to see what is happening:

    >>> factor_codings(('a',), ('b', 'c'), ('a','b','c')) #doctest: +SKIP
    Adding a from ('a',) as indicator because we have not seen any factors yet.
    Adding b from ('b', 'c') as indicator because set([('c',), ()]) is not a subset of set([(), ('a',)])
    Adding c from ('b', 'c') as indicator because set([(), ('b',)]) is not a subset of set([(), ('a',)])
    Adding a from ('a', 'b', 'c') as contrast because set([('c',), ('b', 'c'), (), ('b',)]) is a subset of set([('b', 'c'), (), ('c',), ('b',), ('a',)])
    Adding b from ('a', 'b', 'c') as indicator because set([('c',), (), ('a', 'c'), ('a',)]) is not a subset of set([('b', 'c'), (), ('c',), ('b',), ('a',)])
    Adding c from ('a', 'b', 'c') as indicator because set([('a', 'b'), (), ('b',), ('a',)]) is not a subset of set([('b', 'c'), (), ('c',), ('b',), ('a',)])
    {('b', 'c'): [('b', 'indicator'), ('c', 'indicator')], ('a',): [('a', 'indicator')], ('a', 'b', 'c'): [('a', 'contrast'), ('b', 'indicator'), ('c', 'indicator')]}

    Notes
    -----
    Even though the elements of factor_monomials are assumed to be in graded
    order, the final result depends on the ordering of the strings of the
    factors within each of the tuples.
    """
    lmax = 0
    from copy import copy
    already_seen = set([])
    final_result = []
    for factor_monomial in factor_monomials:
        result = []
        factor_monomial = list(factor_monomial)
        if len(factor_monomial) < lmax:
            raise ValueError('factors are assumed to be in graded order')
        lmax = len(factor_monomial)

        for j in range(len(factor_monomial)):
            cur = copy(list(factor_monomial))
            cur.pop(j)
            terms = simplicial_complex(cur)[2]
            if already_seen and set(terms).issubset(already_seen):
                result.append((factor_monomial[j], 'contrast'))
            else:
                result.append((factor_monomial[j], 'indicator'))
        already_seen = already_seen.union(simplicial_complex(factor_monomial)[2])
        final_result.append((tuple(factor_monomial), result))
    return dict(final_result)


# From matplotlib.cbook
def iterable(obj):
    'return true if *obj* is iterable'
    try:
        iter(obj)
    except TypeError:
        return False
    return True


# From matplotlib.cbook
def is_string_like(obj):
    'Return True if *obj* looks like a string'
    if isinstance(obj, (str, unicode)): return True
    # numpy strings are subclass of str, ma strings are not
    if ma.isMaskedArray(obj):
        if obj.ndim == 0 and obj.dtype.kind in 'SU':
            return True
        else:
            return False
    try: obj + ''
    except: return False
    return True


# From matplotlib.mlab
def rec_append_fields(rec, names, arrs, dtypes=None):
    """
    Return a new record array with field names populated with data
    from arrays in *arrs*.  If appending a single field, then *names*,
    *arrs* and *dtypes* do not have to be lists. They can just be the
    values themselves.
    """
    if (not is_string_like(names) and iterable(names) \
            and len(names) and is_string_like(names[0])):
        if len(names) != len(arrs):
            raise ValueError, "number of arrays do not match number of names"
    else: # we have only 1 name and 1 array
        names = [names]
        arrs = [arrs]
    arrs = map(np.asarray, arrs)
    if dtypes is None:
        dtypes = [a.dtype for a in arrs]
    elif not iterable(dtypes):
        dtypes = [dtypes]
    if len(arrs) != len(dtypes):
        if len(dtypes) == 1:
            dtypes = dtypes * len(arrs)
        else:
            raise ValueError("dtypes must be None, a single dtype or a list")

    newdtype = np.dtype(rec.dtype.descr + zip(names, dtypes))
    newrec = np.recarray(rec.shape, dtype=newdtype)
    for field in rec.dtype.fields:
        newrec[field] = rec[field]
    for name, arr in zip(names, arrs):
        newrec[name] = arr
    return newrec

# Python 3 compatibility routines
if sys.version_info[0] >= 3:
    def to_str(s):
        """ Convert `s` to string, decoding as latin1 if `s` is bytes
        """
        if isinstance(s, bytes):
            return s.decode('latin1')
        return str(s)
else:
    to_str = str
