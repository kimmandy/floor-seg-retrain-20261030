"""One ordered CPU batch ahead; the model, RNG and sampling stay on their old path."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import time

@contextmanager
def prefetched_batches(batches, annotations, pipe, loader):
    def load(items):
        began=time.monotonic()
        data=loader(items,annotations,pipe)
        return data,time.monotonic()-began
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='training-input') as pool:
        def iterate():
            future=pool.submit(load,batches[0]) if batches else None
            for n,items in enumerate(batches):
                began=time.monotonic()
                data,load_seconds=future.result()
                wait_seconds=time.monotonic()-began
                future=pool.submit(load,batches[n+1]) if n+1<len(batches) else None
                yield n,items,data,load_seconds,wait_seconds
        yield iterate()
