from multiprocessing import cpu_count, Pool
from multiprocessing.pool import ThreadPool


def get_process_pool(cores: int = cpu_count() - 1):
    if not cores:
        cores = cpu_count() - 1
    print(('Using {} CPUs'.format(cores)))
    pool = Pool(cores)
    return pool


def get_thread_pool():
    thread_pool = ThreadPool(processes=cpu_count()-1)
    return thread_pool

