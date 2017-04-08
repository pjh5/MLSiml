
from collections import Iterable
from functools import wraps
import logging
import numpy as np
import re


# Flags and options must be alphanumeric, starting with a letter
flag_re = re.compile("-(?P<flag>[a-zA-Z]+[a-zA-Z0-9_\-]*$)")
options = re.compile("--(?P<kw>[a-zA-Z]+[a-zA-Z0-9_\-]*)([=:](?P<val>\S+))?")
ints    = re.compile("(?P<int>[1-9]+[0-9]*|0)$")
floats  = re.compile("(?P<float>([1-9]+[0-9]*|0?)\.[0-9]*$)")

def parse_to_args_and_kwargs(arglist, aliases=None):

    # This code will loop through all the arguments and add each one to either
    # args or kwargs
    args = []
    kwargs = {}

    # Allow aliases. Default only has 'v' -> 'verbose'
    if not aliases:
        aliases = {'v':"verbose"}

    # Loop through arguments
    extra_argument_consumed = False
    for i in range(len(arglist)):

        # This argument has already been consumed
        if extra_argument_consumed:
            extra_argument_consumed = False
            continue

        # Parse flag arguments
        if _parse_flag(kwargs, arglist, i, aliases):
            continue

        # Parse keyword arguments
        parsed, extra_argument_consumed = _parse_keyword(kwargs, arglist, i,
                                                                    aliases)
        if extra_argument_consumed:
            i += 1
        if parsed:
            continue

        # Parse positional arguments, just add them to args
        args.append(_try_to_cast(arglist[i]))

    # Parsed everything
    return args, kwargs


def _parse_flag(kwargs, arglist, i, aliases):

    # Parse flags
    match = flag_re.match(arglist[i])
    if match:
        flag = _clean_and_unalias(aliases, match.group('flag'))

        # Automatically count duplicated flags
        if flag in kwargs:
            kwargs[flag] += 1

        # Set as true in kwargs
        else:
            kwargs[flag] = True

        # Sucessfully parsed
        return True

    # No match
    return False


def _parse_keyword(kwargs, arglist, i, aliases):
    next_argument_consumed = False

    match = options.match(arglist[i])
    if match:
        kw = _clean_and_unalias(aliases, match.group('kw'))

        # If not set with kw=val or kw:val syntax, read the next argument
        if match.group('val'):
            val = match.group('val')

        # Otherwise we already have the value
        else:

            # Require another argument
            if i + 1 >= len(arglist):
                raise Exception("Keyword argument " + arglist[i] +
                        " requires another argument")

            # Next argument should be not be - prefixed
            val = arglist[i + 1]
            if flag_re.match(val) or options.match(val):
                raise Exception("Expecting extra argument after '" + arglist[i] +
                        "' but found flag or option '" + val + "' instead")

            # Found the argument, so move i along
            next_argument_consumed = True


        # Don't reduplicate
        if kw in kwargs:
            raise Exception("Keyword " + kw + " defined twice.")

        # Finally set kw = arg
        kwargs[kw] = _try_to_cast(val)

        # Parse succesful
        return True, next_argument_consumed

    return False, False


def _clean_and_unalias(aliases, arg):

    # Replace aliases with their fuller form
    if arg in aliases:
        return aliases[arg]

    # Else clean the argument to be allowable variable names
    return arg.replace('-', '_')


def _try_to_cast(arg):

    # Try to cast to integer or float
    if ints.match(arg):
        return int(arg)

    if floats.match(arg):
        return float(arg)

    return arg


##############################################################################
# Miscellaneous helper functions
##############################################################################

def truish(obj):
    """Returns if an object evaluates to 'True', handling numpy arrays too"""
    return ((is_iterable(obj) and len(obj) > 0)
            or (not is_iterable(obj) and bool(obj))) and str(obj) != ""

def make_callable(obj):
    """Returns a callable version of obj (with any number of parameters)"""
    return obj if callable(obj) else (lambda o: lambda *z: o)(obj)

def nice_str(thing):
    """Replaces lambda str()s with just 'lambda'"""
    if callable(thing) and not isinstance(thing, Identity):
        return "lambda"
    return str(thing)

class Identity():
    """A wrapper around lambda z: z with a nicer string representation"""

    def __call__(self, z):
        return z

    def __str__(self):
        return "z->z"


##############################################################################
# List / array-like helper functions
##############################################################################

def is_iterable(obj):
    return isinstance(obj, Iterable)

def make_iterable(obj):
    """Returns an iterable version of obj, possibly just [obj]"""
    return obj if is_iterable(obj) else [obj]

def flatten(array, recursive=False):
    """Flattens an array of arrays into a single array

    e.g. [1, [a, b], 2, [c, d, e]] -> [1, a, b, 2, c, d, e]
    """
    flattened = []
    for elem in array:
        if type(elem) is list:
            if recursive:
                flattened += flatten(elem)
            else:
                flattened += elem
        else:
            flattened.append(elem)
    return flattened

def to_flat_np_array(arr):
    """Flatten for np arrays cause the builtin one doesn't work"""
    return np.array(flatten(
                map(list, map( make_iterable, make_iterable(arr)))
                )).ravel()


##############################################################################
# Dictionary helper functions
##############################################################################

def dict_prefix(prefix, dictionary):
    """Returns a copy of dictionary with every key prefixed by prefix"""
    if not prefix:
        return dictionary.copy()
    return {"{!s}_{!s}".format(prefix, k):v for k, v in dictionary.items()}

def filter_truish(dictionary):
    """Returns dictionary without and keys mapped to non-truish values"""
    return {k:v for k, v in dictionary.items() if truish(v)}

def replace_keys(**replacement_dict):
    """Decorator (with arguments) to replace some kwarg keywords with others

    Usage:
    ======
    @replace_keys(
        low="loc",
        high=("scale, lambda val, **kwargs: val + kwargs.get("low", 0))
        )
    def Uniform(low=0, high=1, **kwargs):
        print("Uniform({!s}, {!s}).format(low, high)
        return stats.uniform(**kwargs)

    unif = Uniform(low=5, high=10)                  # prints 'Uniform(5, 10)'
    unif.rvs()                                  # samples from Uniform(5, 10)
    ==========================================================================

    In the example above, stats.uniform expects parameters "loc" and "scale",
    where "loc" is the "low" of the uniform and "scale" is its width. These
    parameters are unintuitive. By decorating as above, the parameters "low"
    and "high" can be used instead.

    This expects key=value pairs of
        user-friendly-key=library-expected-key

    For parameters that also need a transformation, such as high -> scale as
    above, the syntax is
        user-friendly-key=tuple(
            library-expected-key,
            function(entered-value, other-kwargs)=>new-value
            )
    """

    def decorator(func):
        """The actual decorator function. Technically the wrapping function
        replace_keys is a decorator that returns a decorator.
        """

        @wraps(func)
        def new_func(*orig_args, **orig_kwargs):
            """The decorated function with keys in kwargs substituted out"""

            for old_key, new_key in replacement_dict.items():

                # Replace the key in orig_kwargs if its in our replacement dict
                if old_key in orig_kwargs:

                    # Some substituions transform the value as well
                    # For this syntax, we expect new_key=(new_key_str,
                    # trans_func) where trans_func is fed (old_key_value,
                    # all_other_kwargs)
                    if isinstance(new_key, tuple):
                        orig_kwargs[new_key[0]] = (
                                new_key[1](orig_kwargs[old_key], **orig_kwargs)
                                )

                    else:
                        orig_kwargs[new_key] = orig_kwargs[old_key]

                    # Remove the old key to avoid eventual exceptions
                    # (unexpected keyword parameter)
                    orig_kwargs.pop(old_key)

            return func(*orig_args, **orig_kwargs)
        return new_func
    return decorator
