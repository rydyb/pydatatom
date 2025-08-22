class DropKeys:
    def __init__(self, *keys: list[str]):
        self.keys = keys

    def __call__(self, item: dict):
        for key in self.keys:
            item.pop(key, None)
        return item
