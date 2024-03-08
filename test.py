class BaseFoo:
    def __init__(self, a, b):
        print(a)
        print(b)


class DeriveFoo(BaseFoo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


DeriveFoo(1, 2)
