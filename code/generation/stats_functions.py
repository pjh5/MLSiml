import scipy.stats


def normal_generator(mean, var):
    return (lambda f: lambda: f.rvs())(scipy.stats.norm(loc=mean, scale=var))


def normal(mean, var):
    return scipy.stats.norm.rvs(loc=mean, scale=var)

def bernoulli_generator(prob):
    return (lambda p: lambda: float(scipy.stats.uniform.rvs() > p))(prob)

def bernoulli(p):
    return float(scipy.stats.uniform.rvs() > p)
