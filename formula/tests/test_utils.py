""" Testing formula utils
"""

import numpy as np

from .. utils import fullrank, simplicial_complex

from numpy.testing import (assert_array_almost_equal,
                           assert_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_equal, assert_raises


def test_fullrank():
    rng = np.random.RandomState(20120408)
    X = rng.standard_normal((40,5))
    X[:,0] = X[:,1] + X[:,2]
    Y1 = fullrank(X)
    assert_equal(Y1.shape, (40,4))
    Y2 = fullrank(X, r=3)
    assert_equal(Y2.shape, (40,3))
    Y3 = fullrank(X, r=4)
    assert_equal(Y3.shape, (40,4))
    assert_almost_equal(Y1, Y3)


def assert_sequal(complexes, faces, facets):
    # Test whether predicted result for simplicial_complex is correct
    out_faces, out_facets, out_all = simplicial_complex(*complexes)
    assert_equal(faces, out_faces)
    assert_equal(set(facets), set(out_facets))
    face_all = sum([list(f) for f in faces.values()], [])
    assert_equal(set(face_all), set(out_all))


def test_simplicial():
    # Test simplicial complex generation
    assert_sequal((), {0: set()}, [])
    assert_sequal((('a',)),
                 {0: set(), 1: set([('a',)])}, [('a',)])
    assert_sequal((('a','b'),),
                  {0: set(), 1: set([('a',), ('b',)]), 2: set([('a', 'b')])},
                  [('a', 'b')])
    assert_sequal((('a','b'), ('a',)),
                  {0: set(), 1: set([('a',), ('b',)]), 2: set([('a', 'b')])},
                  [('a', 'b')])
    assert_sequal((('a','b'), ('a','b')),
                  {0: set(), 1: set([('a',), ('b',)]), 2: set([('a', 'b')])},
                  [('a', 'b')])
    assert_sequal((('a','b'), ('a','c')),
                 {0: set(),
                  1: set([('a',), ('b',), ('c',)]),
                  2: set([('a', 'b'), ('a', 'c')])},
                  [('a', 'b'), ('a', 'c')])
    assert_sequal((('a','b','c'), ('a','c')),
                 {0: set(),
                  1: set([('a',), ('b',), ('c',)]),
                  2: set([('a', 'b'), ('b', 'c'), ('a', 'c')]),
                  3: set([('a', 'b', 'c')])},
                  [('a', 'b', 'c')])
    assert_sequal((('a','b','c'), ('a','d')),
                 {0: set(),
                  1: set([('a',), ('b',), ('c',), ('d',)]),
                  2: set([('a', 'b'), ('b', 'c'), ('a', 'c'), ('a', 'd')]),
                  3: set([('a', 'b', 'c')])},
                  [('a', 'b', 'c'), ('a', 'd')])
    assert_sequal((('a','b','c'), ('c','d'), ('e','f'), ('e',)),
                  {0: set([]),
                   1: set([('a',), ('c',), ('b',), ('e',), ('d',), ('f',)]),
                   2: set([('b', 'c'), ('a', 'b'), ('e', 'f'),
                           ('c', 'd'), ('a', 'c')]),
                   3: set([('a', 'b', 'c')])},
                  [('c', 'd'), ('e', 'f'), ('a', 'b', 'c')])
