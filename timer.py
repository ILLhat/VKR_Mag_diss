from functools import wraps
from time import perf_counter


def timeit(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = function(*args, **kwargs)
        end_time = perf_counter()
        time = end_time - start_time
        print(f"{function.__name__} took {time * 1000} ms ({time} s)")
        return result

    return wrapper
