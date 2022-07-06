""" Helper classes. """
from ast import literal_eval
from time import perf_counter
from collections import OrderedDict
from functools import wraps

import numpy as np
import bottleneck as bn


from .functions import to_list



class AugmentedNumpy:
    """ NumPy with better routines for nan-handling. """
    def __getattr__(self, key):
        return getattr(bn, key, getattr(np, key))
augmented_np = AugmentedNumpy()



class LoopedList(list):
    """ List that loops from given position (default is 0).

        Examples
        --------
        >>> l = LoopedList(['a', 'b', 'c'])
        >>> [l[i] for i in range(9)]
        ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']

        >>> l = LoopedList(['a', 'b', 'c', 'd'], loop_from=2)
        >>> [l[i] for i in range(9)]
        ['a', 'b', 'c', 'd', 'c', 'd', 'c', 'd', 'c']

        >>> l = LoopedList(['a', 'b', 'c', 'd', 'e'], loop_from=-1)
        >>> [l[i] for i in range(9)]
        ['a', 'b', 'c', 'd', 'e', 'e', 'e', 'e', 'e']
    """
    def __init__(self, *args, loop_from=0, **kwargs):
        self.loop_from = loop_from
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        if idx >= len(self):
            pos = self.loop_from + len(self) * (self.loop_from < 0)
            if pos < 0:
                raise IndexError(f"List of length {len(self)} is looped from {self.loop_from} index")
            idx = pos + (idx - pos) % (len(self) - pos)
        return super().__getitem__(idx)


class AugmentedList(list):
    """ List that delegates attribute retrieval requests to contained objects and can be indexed with other iterables.
        On successful attribute request returns the list of results, which is itself an instance of `AugmentedList`.
        Auto-completes names to that of contained objects. Meant to be used for storing homogeneous objects.

        Examples
        --------
        1. Let `lst` be an `AugmentedList` of objects that have `mean` method.
        Than the following expression:
        >>> lst.mean()
        Is equivalent to:
        >>> [item.mean() for item in lst]

        2. Let `lst` be an `AugmentedList` of objects that have `shape` attribute.
        Than the following expression:
        >>> lst.shape
        Is equivalent to:
        >>> [item.shape for item in lst]

        Notes
        -----
        Using `AugmentedList` for heterogeneous objects storage is not recommended, due to the following:
        1. Tab autocompletion suggests attributes from the first list item only.
        2. The request of the attribute absent in any of the objects leads to an error.
    """
    def __getitem__(self, key):
        """ Manage indexing via iterable. """
        if isinstance(key, (int, np.integer)):
            return super().__getitem__(key)

        if isinstance(key, slice):
            return type(self)(super().__getitem__(key))

        return type(self)([super().__getitem__(idx) for idx in key])

    # Delegating to contained objects
    def __getattr__(self, key):
        """ Get attributes of list items, recusively delegating this process to items if they are lists themselves. """
        if len(self) == 0:
            return lambda *args, **kwargs: self

        attributes = type(self)([getattr(item, key) for item in self])

        if not callable(attributes.reference_object):
            return type(self)(attributes)

        @wraps(attributes.reference_object)
        def wrapper(*args, **kwargs):
            return type(self)([method(*args, **kwargs) for method in attributes])

        return wrapper

    @property
    def reference_object(self):
        """ First item of a list taking into account its nestedness. """
        return self[0].reference_object if isinstance(self[0], type(self)) else self[0]

    def __dir__(self):
        """ Correct autocompletion for delegated methods. """
        return dir(list) if len(self) == 0 else dir(self[0])

    # Correct type of operations
    def __add__(self, other):
        return type(self)(list.__add__(self, other))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        return type(self)(list.__mul__(self, other))

    def __rmul__(self, other):
        return self.__mul__(other)


class DelegatingList(AugmentedList):
    """ `AugmentedList` that extends that makes delegation its items' attributes

        Examples
        --------
        1. Len `lst` be an `AugmentedList` of objects and `f` be a function that accepts such objects.
        Than the following expression:
        >>> lst.apply(f)
        Is equivalent to:
        >>> [f(item) for item in lst]

        2. Let `l` be an `AugmentedList` of dictionaries:
        >>> l = AugmentedList([{'cmap': 'viridis', 'alpha': 1.0},
                                [{'cmap': 'ocean', 'alpha': 1.0}, {'cmap': 'Reds', 'alpha': 0.7}]])
        That the following expresion:
        >>> l.to_dict()
        Will be evaluated to:
        >>> {'cmap': ['viridis, ['ocean', 'Reds]], 'alpha': [1.0, [1.0, 0.7]]}
    """
    def __init__(self, obj=None):
        """ Perform items recusive casting to `AugmentedList` type if they are lists. """
        obj = [] if obj is None else obj if isinstance(obj, list) else [obj]
        super().__init__([type(self)(item) if isinstance(item, list) else item for item in obj])

    def apply(self, func, *args, shallow=False, **kwargs):
        """ Recursively traverse list items applying given function and return list of results with same nestedness.

        Parameters
        ----------
        func : callable
            Function to apply to items.
        shallow : bool
            If True, apply function directly to outer list items disabling recursive descent.
        args, kwargs : misc
            For `func`.
        """
        result = type(self)()

        for item in self:
            if isinstance(item, type(self)) and not shallow:
                res = item.apply(func, *args, **kwargs)
            else:
                res = func(item, *args, **kwargs)

            if isinstance(res, list):
                res = type(self)(res)

            result.append(res)

        return result

    def to_dict(self):
        """ Convert nested list of dicts to dict of nested lists. Address class docs for usage examples. """
        if not isinstance(self.reference_object, dict):
            raise TypeError('Only lists consisting of `dict` items can be converted.')

        result = {}

        # pylint: disable=cell-var-from-loop
        for key in self.reference_object:
            try:
                result[key] = self.apply(lambda dct: dct[key])
            except KeyError as e:
                raise ValueError(f'KeyError occured due to absence of key `{key}` in some of list items.') from e

        return result

    @property
    def flat(self):
        """ Flat list of items. """
        res = type(self)()

        for item in self:
            if isinstance(item, type(self)):
                res.extend(item.flat)
            else:
                res.append(item)

        return res


class AugmentedDict(OrderedDict):
    """ Ordered dictionary with additional features:
        - can be indexed with ordinals.
        - delegates calls to contained objects.
        For example, `a_dict.method()` is equivalent to `{key : value.method() for key, value in a_dict.items()}`.
        Can be used to retrieve attributes, properties and call methods.
        Returns the dictionary with results, which is itself an instance of `AugmentedDict`.
        - auto-completes names to that of contained objects.
        - can be flattened.
    """
    # Ordinal indexation
    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            key = list(self.keys())[key]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, (int, np.integer)):
            key = list(self.keys())[key]

        if isinstance(value, list):
            value = AugmentedList(value)
        super().__setitem__(key, value)

    # Delegating to contained objects
    def __getattr__(self, key):
        if len(self) == 0:
            return lambda *args, **kwargs: self

        attribute = getattr(self[0], key)

        if not callable(attribute):
            # Attribute or property
            return AugmentedDict({key_ : getattr(value, key) for key_, value in self.items()})

        @wraps(attribute)
        def method_wrapper(*args, **kwargs):
            return AugmentedDict({key_ : getattr(value, key)(*args, **kwargs) for key_, value in self.items()})
        return method_wrapper

    def __dir__(self):
        """ Correct autocompletion for delegated methods. """
        if len(self) != 0:
            return dir(self[0])
        return dir(dict)

    # Convenient iterables
    def flatten(self, keys=None):
        """ Get dict values for requested keys in a single list. """
        keys = to_list(keys) if keys is not None else list(self.keys())
        lists = [self[key] if isinstance(self[key], list) else [self[key]] for key in keys]
        flattened = sum(lists, [])
        return AugmentedList(flattened)

    @property
    def flat(self):
        """ List of all dictionary values. """
        return self.flatten()



class MetaDict(dict):
    """ Dictionary that can dump itself on disk in a human-readable and human-editable way.
    Usually describes cube meta info such as name, coordinates (if known) and other useful data.
    """
    def __repr__(self):
        lines = '\n'.join(f'    "{key}" : {repr(value)},'
                          for key, value in self.items())
        return f'{{\n{lines}\n}}'

    @classmethod
    def load(cls, path):
        """ Load self from `path` by evaluating the containing dictionary. """
        with open(path, 'r', encoding='utf-8') as file:
            content = '\n'.join(file.readlines())
        return cls(literal_eval(content.replace('\n', '').replace('    ', '')))

    def dump(self, path):
        """ Save self to `path` with each key on a separate line. """
        with open(path, 'w', encoding='utf-8') as file:
            print(repr(self), file=file)


    @classmethod
    def placeholder(cls):
        """ Default MetaDict. """
        return cls({
            'name': 'UNKNOWN',
            'ru_name': 'Неизвестно',
            'latitude': None,
            'longitude': None,
            'info': 'дополнительная информация о кубе'
        })



class timer:
    """ Context manager for timing the code. """
    def __init__(self, string=''):
        self.string = string
        self.start_time = None

    def __enter__(self):
        self.start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(f'{self.string} evaluated in {(perf_counter() - self.start_time):4.4f} seconds')



class SafeIO:
    """ Opens the file handler with desired `open` function, closes it at destruction.
    Can log open and close actions to the `log_file`.
    getattr, getitem and `in` operator are directed to the `handler`.
    """
    def __init__(self, path, opener=open, log_file=None, **kwargs):
        self.path = path
        self.log_file = log_file
        self.handler = opener(path, **kwargs)

        if self.log_file:
            self._info(self.log_file, f'Opened {self.path}')

    def _info(self, log_file, msg):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write('\n' + msg)

    def __getattr__(self, key):
        return getattr(self.handler, key)

    def __getitem__(self, key):
        return self.handler[key]

    def __contains__(self, key):
        return key in self.handler

    def __del__(self):
        self.handler.close()

        if self.log_file:
            self._info(self.log_file, f'Closed {self.path}')
