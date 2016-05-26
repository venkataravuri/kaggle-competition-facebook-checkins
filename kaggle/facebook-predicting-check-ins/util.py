import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('Execution of {0}() took {3:2.2f} sec\n'.format(method.__name__, args, kw, te - ts))
        return result

    return timed
