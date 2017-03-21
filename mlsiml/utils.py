from collections import Iterable
import re


# Flags and options must be alphanumeric, starting with a letter
flag_re = re.compile("-(?P<flag>[a-zA-Z]+[a-zA-Z0-9_\-]*$)")
options = re.compile("--(?P<kw>[a-zA-Z]+[a-zA-Z0-9_\-]*)([=:](?P<val>\S+))?")
ints    = re.compile("(?P<int>[1-9]+[0-9]*|0)$")
floats  = re.compile("(?P<float>([1-9]+[0-9]*|0)\.[0-9]*$)")

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

        # Don't reduplicate
        if flag in kwargs:
            raise Error("Flag " + flag + " defined twice.")

        # Set as true in kwargs
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
                raise Error("Keyword argument " + sysarg[i] +
                        " requires another argument")

            # Next argument should be not be - prefixed
            val = arglist[i + 1]
            if flag_re.match(val) or options.match(val):
                raise Error("Expecting extra argument after '" + sysarg[i] +
                        "' but found flag or option '" + val + "' instead")

            # Found the argument, so move i along
            next_argument_consumed = True


        # Don't reduplicate
        if kw in kwargs:
            raise Error("Keyword " + kw + " defined twice.")

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


def flatten(array, recursive=False):
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


def make_iterable(obj):
    return obj if isinstance(obj, Iterable) else [obj]

def make_callable(obj):
    return obj if callable(obj) else (lambda o: lambda z: o)(obj)
