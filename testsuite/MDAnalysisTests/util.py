# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the Lesser GNU Public Licence, v2.1 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#
"""
Useful functions for running tests

"""

import builtins

builtins_name = "builtins"
importer = builtins.__import__

from contextlib import contextmanager
from functools import wraps
import importlib
import shutil
from unittest import mock
import os
import warnings
import pytest

from numpy.testing import assert_warns
import numpy as np
from numpy.lib import NumpyVersion


def block_import(package):
    """Block import of a given package

    eg:

    @block_import('numpy')
    def try_and_do_something():
        import numpy as np  # this will fail!

    Will also block imports of subpackages ie block_import('numpy') should
    block 'import numpy.matrix'

    Shadows the builtin import method, sniffs import requests
    and blocks the designated package.
    """

    def blocker_wrapper(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            with mock.patch(
                "{}.__import__".format(builtins_name), wraps=importer
            ) as mbi:

                def blocker(*args, **kwargs):
                    if package in args[0]:
                        raise ImportError("Blocked by block_import")
                    else:
                        # returning DEFAULT allows the real function to continue
                        return mock.DEFAULT

                mbi.side_effect = blocker
                func(*args, **kwargs)

        return func_wrapper

    return blocker_wrapper


def executable_not_found(*args):
    """Return ``True`` if none of the executables in args can be found.

    ``False`` otherwise (i.e. at least one was found).

    To be used as the argument of::

    @dec.skipif(executable_not_found("binary_name"), msg="skip test because binary_name not available")
    """
    for name in args:
        if shutil.which(name) is not None:
            return False
    return True


def import_not_available(module_name):
    """Helper function to check if a module cannot be imported, intended as an
    argument of pytest.mark.skipif

    Parameters
    ----------
    module_name : str
        Name of module to test importing

    Returns
    -------
    True
        if module cannot be imported
    False
        otherwise (i.e. module can be imported)

    Example
    -------
    To be used in the following manner::

    @pytest.mark.skipif(import_not_available("module_name"),
                        msg="skip test as module_name could not be imported")

    """
    # TODO: remove once these packages have a release
    # with NumPy 2 support
    if NumpyVersion(np.__version__) >= "2.0.0":
        if module_name == "parmed":
            return True
    try:
        test = importlib.import_module(module_name)
    except ImportError:
        return True
    else:
        return False


@contextmanager
def in_dir(dirname):
    """Context manager for safely changing directories.

    Arguments
    ---------
    dirname : string
        directory to change into

    Example
    -------
    Change into a temporary directory and always change back to the
    current one::

      with in_dir("/tmp") as tmpdir:
          # do stuff

    SeeAlso
    -------
    The :mod:`tmpdir` module provides functionality such as :func:`tmpdir.in_tmpdir`
    to create temporary directories that are automatically deleted once they are no
    longer used.
    """

    old_path = os.getcwd()
    os.chdir(dirname)
    yield dirname
    os.chdir(old_path)


def assert_nowarns(warning_class, *args, **kwargs):
    r"""Fail if the given callable throws the specified warning.

    A warning of class warning_class should NOT be thrown by the callable when
    invoked with arguments args and keyword arguments kwargs.
    If a different type of warning is thrown, it will not be caught.

    Parameters
    ----------
    warning_class : class
        The class defining the warning that `func` is expected to throw.
    func : callable
        The callable to test.
    \*args : Arguments
        Arguments passed to `func`.
    \*\*kwargs : Kwargs
        Keyword arguments passed to `func`.

    Returns
    -------
    True
         if no `AssertionError` is raised

    Note
    ----
    numpy.testing.assert_warn returns the value returned by `func`; we would
    need a second func evaluation so in order to avoid it, only True is
    returned if no assertion is raised.

    SeeAlso
    -------
    numpy.testing.assert_warn

    """
    func = args[0]
    args = args[1:]
    try:
        value = assert_warns(DeprecationWarning, func, *args, **kwargs)
    except AssertionError:
        # a warning was NOT emitted: all good
        return True
    else:
        # There was a warning even though we do not want to see one.
        raise AssertionError(
            "function {0} raises warning of class {1}".format(
                func.__name__, warning_class.__name__
            )
        )


@contextmanager
def no_warning(warning_class):
    """contextmanager to check that no warning was raised"""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        yield
    if len(record) != 0:
        raise AssertionError(
            "Raised warning of class {}".format(warning_class.__name__)
        )


class _NoDeprecatedCallContext(object):
    # modified version of similar pytest class object that checks for
    # raised DeprecationWarning

    def __enter__(self):
        self._captured_categories = []
        self._old_warn = warnings.warn
        self._old_warn_explicit = warnings.warn_explicit
        warnings.warn_explicit = self._warn_explicit
        warnings.warn = self._warn

    def _warn_explicit(self, message, category, *args, **kwargs):
        self._captured_categories.append(category)

    def _warn(self, message, category=None, *args, **kwargs):
        if isinstance(message, Warning):
            self._captured_categories.append(message.__class__)
        else:
            # as follows Python documentation at
            # https://docs.python.org/3/library/warnings.html#warnings.warn
            # if category is None, the default UserWarning is used
            if category is None:
                category = UserWarning
            self._captured_categories.append(category)

    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.warn_explicit = self._old_warn_explicit
        warnings.warn = self._old_warn

        if exc_type is None:
            deprecation_categories = (
                DeprecationWarning,
                PendingDeprecationWarning,
            )
            if any(
                issubclass(c, deprecation_categories)
                for c in self._captured_categories
            ):
                __tracebackhide__ = True
                msg = (
                    "Produced DeprecationWarning or PendingDeprecationWarning"
                )
                raise AssertionError(msg)


def no_deprecated_call(func=None, *args, **kwargs):
    # modified version of similar pytest function
    # check that DeprecationWarning is NOT raised
    if not func:
        return _NoDeprecatedCallContext()
    else:
        __tracebackhide__ = True
        with _NoDeprecatedCallContext():
            return func(*args, **kwargs)


def get_userid():
    """
    Calls os.geteuid() where possible, or returns 1000 (usually on windows).
    """
    # no such thing as euid on Windows, assuming normal user 1000
    if os.name == "nt" or not hasattr(os, "geteuid"):
        return 1000
    else:
        return os.geteuid()
