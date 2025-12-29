
# class Test:
#     a = 0
#     b = 0

#     def __init__(self) -> None:
#         self.c = 0
#         self.d = 0

#     def func_a(self):
#         self.a = 1
#         self.b = 1

#     @classmethod
#     def func_b(cls):
#         cls.a = 1
#         cls.b = 1

# test = Test()

# test.func_a()

# print(test.a)
# print(Test.a)

from functools import wraps
import time

def my_decorator(func):
    # @wraps(func)  # <--- Use it here on the inner function
    def wrapper(*args, **kwargs):
        """
        This is the wrapper's docstring.
        """
        print("Before call")
        result = func(*args, **kwargs)
        print("After call")
        return result
    return wrapper

@my_decorator
def say_hello():
    """
    Greet the user.
    """
    print("Hello!")

# Now these look correct:
# print(say_hello.__name__)  # Prints "say_hello" (instead of "wrapper")
# print(say_hello.__doc__)   # Prints "Greet the user." (instead of wrapper's doc)

# say_hello()

from utils.benchmark import record_latency

@record_latency(iterations=10)
def benchmark():
    time.sleep(1.0)

benchmark()





