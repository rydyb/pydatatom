class PickKey:
    def __init__(self, key: str):
        self.key = key

    def __call__(self, item: dict):
        if self.key not in item:
            raise KeyError(f"key '{self.key}' not found in data")
        return item[self.key]
