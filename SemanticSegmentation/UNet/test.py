
class Test:
    a = 0
    b = 0

    def __init__(self) -> None:
        self.c = 0
        self.d = 0

    def func_a(self):
        self.a = 1
        self.b = 1

    @classmethod
    def func_b(cls):
        cls.a = 1
        cls.b = 1

test = Test()

test.func_a()

print(test.a)
print(Test.a)