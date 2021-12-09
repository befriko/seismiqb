""" Thread-safe lru cache class and cache mixin. """
from collections import OrderedDict, defaultdict
from functools import wraps
from threading import RLock
from hashlib import blake2b
from copy import copy
import numpy as np

class lru_cache:
    """ Thread-safe least recent used cache. Must be applied to a class methods.
    Adds the `use_cache` argument to the decorated method to control whether the caching logic is applied.
    Stored values are individual for each instance of a class.

    Parameters
    ----------
    maxsize : int
        Maximum amount of stored values.
    attributes: None, str or sequence of str
        Attributes to get from object and use as additions to key.
    apply_by_default : bool
        Whether the cache logic is on by default.
    copy_on_return : bool
        Whether to copy the object on retrieving from cache.

    Examples
    --------
    Store loaded slides::

    @lru_cache(maxsize=128)
    def load_slide(cube_name, slide_no):
        pass

    Notes
    -----
    All arguments to a decorated method must be hashable.
    """
    #pylint: disable=invalid-name, attribute-defined-outside-init
    def __init__(self, maxsize=None, attributes=None, apply_by_default=True, copy_on_return=False):
        self.maxsize = maxsize
        self.apply_by_default = apply_by_default
        self.copy_on_return = copy_on_return

        # Parse `attributes`
        if isinstance(attributes, str):
            self.attributes = [attributes]
        elif isinstance(attributes, (tuple, list)):
            self.attributes = attributes
        else:
            self.attributes = False

        self.default = Singleton
        self.lock = RLock()
        self.reset()

    def reset(self, instance=None):
        """ Clear cache and stats. """
        if instance is None:
            self.cache = defaultdict(OrderedDict)
            self.is_full = defaultdict(lambda: False)
            self.stats = defaultdict(lambda: {'hit': 0, 'miss': 0})
        else:
            self.cache[instance] = OrderedDict()
            self.is_full[instance] = False
            self.stats[instance] = {'hit': 0, 'miss': 0}

    def make_key(self, instance, args, kwargs):
        """ Create a key from a combination of instance reference, method args, and instance attributes. """
        key = [instance] + list(args)
        if kwargs:
            for k, v in sorted(kwargs.items()):
                key.append((k, v))

        if self.attributes:
            for attr in self.attributes:
                attr_hash = stable_hash(getattr(instance, attr))
                key.append(attr_hash)
        return flatten_nested(key)


    def __call__(self, func):
        """ Add the cache to the function. """
        @wraps(func)
        def wrapper(instance, *args, **kwargs):
            use_cache = kwargs.pop('use_cache', self.apply_by_default)
            copy_on_return = kwargs.pop('copy_on_return', self.copy_on_return)

            # Skip the caching logic and evaluate function directly
            if not use_cache:
                result = func(instance, *args, **kwargs)
                return result

            key = self.make_key(instance, args, kwargs)

            # If result is already in cache, just retrieve it and update its timings
            with self.lock:
                result = self.cache[instance].get(key, self.default)
                if result is not self.default:
                    del self.cache[instance][key]
                    self.cache[instance][key] = result
                    self.stats[instance]['hit'] += 1
                    return copy(result) if copy_on_return else result

            # The result was not found in cache: evaluate function
            result = func(instance, *args, **kwargs)

            # Add the result to cache
            with self.lock:
                self.stats[instance]['miss'] += 1
                if key in self.cache[instance]:
                    pass
                elif self.is_full[instance]:
                    self.cache[instance].popitem(last=False)
                    self.cache[instance][key] = result
                else:
                    self.cache[instance][key] = result
                    self.is_full[instance] = (len(self.cache[instance]) >= self.maxsize)
            return copy(result) if copy_on_return else result

        wrapper.__name__ = func.__name__
        wrapper.cache = lambda: self.cache
        wrapper.stats = lambda: self.stats
        wrapper.reset = self.reset
        wrapper.reset_instance = lambda instance: self.reset(instance=instance)
        return wrapper

class SingletonClass:
    """ There must be only one! """
Singleton = SingletonClass()

def stable_hash(key):
    """ Hash that stays the same between different runs of Python interpreter. """
    if not isinstance(key, (str, bytes)):
        key = ''.join(sorted(str(key)))
    if not isinstance(key, bytes):
        key = key.encode('ascii')
    return str(blake2b(key).hexdigest())

def flatten_nested(iterable):
    """ Recursively flatten nested structure of tuples, list and dicts. """
    result = []
    if isinstance(iterable, (tuple, list)):
        for item in iterable:
            result.extend(flatten_nested(item))
    elif isinstance(iterable, dict):
        for key, value in sorted(iterable.items()):
            result.extend((*flatten_nested(key), *flatten_nested(value)))
    else:
        return (iterable,)
    return tuple(result)


class CacheMixin:
    """ Methods for cache management.

    You can use this mixin for cache introspection and clearing cached data.
    """
    def get_cached_methods(self):
        """ Get a list of methods that use caching."""
        methods = []

        for name in dir(self):
            is_property = isinstance(getattr(self.__class__, name, None), property)
            if name.startswith("__") or 'cache' in name or is_property:
                continue

            method = getattr(self, name)

            if callable(method):
                if hasattr(method, 'cache'):
                    methods.append(method)

        return methods

    def reset_cache(self, cached_methods=None):
        """ Clear cached data.

        Parameters:
        ----------
        cached_methods: list
            A list of methods that cache data.
            By default, uses all cached methods of the class.
        """
        if cached_methods is None:
            cached_methods = self.get_cached_methods()

        for method in cached_methods:
            method.reset(instance=self)

    def get_cache_length(self, cached_methods=None):
        """ Total amount of cached objects for specified methods.

        Parameters:
        ----------
        cached_methods: list
            A list of methods that cache data.
            By default, uses all cached methods of the class.
        """
        if cached_methods is None:
            cached_methods = self.get_cached_methods()

        cache_length_accumulator = 0

        for method in cached_methods:
            cache_length_accumulator += len(method.cache()[self])

        return cache_length_accumulator

    def get_cache_size(self, cached_methods=None):
        """ Total size of cached objects for specified methods.

        Parameters:
        ----------
        cached_methods: list
            A list of methods that cache data.
            By default, uses all cached methods of the class.
        """
        if cached_methods is None:
            cached_methods = self.get_cached_methods()

        cache_size_accumulator = 0

        for method in cached_methods:
            method_values = list(method.cache()[self].values())

            for values in method_values:
                if isinstance(values, np.ndarray):
                    cache_size_accumulator += sum(item.nbytes / (1024 ** 3) for item in method_values)

        return cache_size_accumulator

    @property
    def cache_length(self):
        """ Total amount of cached objects. """
        return self.get_cache_length()

    @property
    def cache_size(self):
        """ Total size of cached objects. """
        return self.get_cache_size()

    @property
    def cache_repr(self):
        """ Cache representation is presented by a dictionary with names of methods that cache data as keys.
        Values are dictionaries with information about cache length, size, and method arguments.
        """
        cached_methods = self.get_cached_methods()

        cache_repr_ = {}

        for method in cached_methods:
            method_cache_length = self.get_cache_length(cached_methods=[method])

            if method_cache_length:
                method_cache_size = self.get_cache_size(cached_methods=[method])

                arguments = list(method.cache()[self].keys())[0][1:]
                arguments = dict(zip(arguments[::2], arguments[1::2]))

                cache_repr_[method.__name__] = {
                    'cache_length': method_cache_length,
                    'cache_size': method_cache_size,
                    'arguments': arguments
                }

        return cache_repr_
