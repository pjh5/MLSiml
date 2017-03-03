"""
Functions to sample from distributions and to make generators that sample from
frozen distributions.
"""
from scipy import stats


class Distribution:

    def __init__(self, base):
        self.base = base

    @classmethod
    def Normal(cls):
        return cls(stats.norm)

    @classmethod
    def Exponential(cls):
        return cls(stats.expon)

    def sample(self, **kwargs):
        return self.base.rvs(**kwargs)

    """
    def sampler_for(self, param_function, **kwargs):

        # If given a function to use to generate the distribution parameters
        if param_function:
            def make_sampler_with_lambda_arg(sampler, generator):
                return lambda args: sampler(**generator(args))

            return make_sampler_with_lambda_arg(self.sample, param_function)

        # Else just parametize the distribution with the keyword arguments
        return (lambda f: lambda: f.rvs())(self.base(**kwargs))
    """

    def sampler_for(self, **kwargs):

        # Build dictionary of all callable keyword arguments
        callable_kwargs = {
                kw:(arg if callable(arg) else (lambda a: lambda z: a)(arg))
                for kw, arg in kwargs.items()
                }

        # Make lambda to pass argument to every keyword function in dictionary
        kwargs_generator = (lambda _kwargs: lambda z: 
                {kw:arg(z) for kw,arg in _kwargs.items()})(callable_kwargs)

        # Return a sampler with the lambda function
        return (lambda sampler, param_generator: 
                (lambda z: sampler(**param_generator(z)))
                )(self.sample, kwargs_generator)
        

# Bernoulli Distribution

def bernoulli(prob):
    return (lambda p: lambda: float(stats.uniform.rvs() > p))(prob)

def bernoulli_sampler_for(p):
    return float(stats.uniform.rvs() > p)

