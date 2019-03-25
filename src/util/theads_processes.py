from multiprocessing import cpu_count, Pool
from multiprocessing.pool import ThreadPool


def get_process_pool():
    # Process in multiple blocking processes
    print(('Detected {} CPUs'.format(cpu_count())))
    pool = Pool(processes=cpu_count())
    return pool


def get_thread_pool():
    thread_pool = ThreadPool(processes=cpu_count())
    return thread_pool
