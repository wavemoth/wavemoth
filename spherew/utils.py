
class FakeFuture(object):
    def __init__(self, func, *args, **kw):
        self.data = (func, args, kw)
    def result(self):
        func, args, kw = self.data
        return func(*args, **kw)

class FakeExecutor(object):
    def submit(self, func, *args, **kw):
        return FakeFuture(func, *args, **kw)

