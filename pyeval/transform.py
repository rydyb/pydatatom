class ExtractKey:
    def __init__(self, key: str):
        self.key = key

    def __call__(self, data):
        return data[self.key]
