from multiprocessing import cpu_count, Pool


def get_process_pool():
    # Process in multiple blocking processes
    print(('Detected {} CPUs'.format(cpu_count())))
    pool = Pool(processes=cpu_count())
    return pool