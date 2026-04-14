from functools import wraps
import time, logging
from tqdm import trange
from utils.common import logging_handler

logging.basicConfig(level=logging.INFO, handlers=[logging_handler])


def record_latency(iterations=1):
    """
    The outer factory takes the custom arguments.
    """
    def run_iterations(func):
        """
        The middle layer is the actual decorator.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            The inner wrapper runs the logic.
            """
            total_latency = 0.0
            for _ in trange(iterations, desc="Iterations", unit="iteration"):
                start_time = time.time()
                result = func(*args, **kwargs) # Runs the function with any arguments
                end_time = time.time()
                total_latency += (end_time - start_time)
            logging.info(f"Function {func.__name__} averagely took {total_latency / iterations:.4f} s")
        return wrapper
    return run_iterations