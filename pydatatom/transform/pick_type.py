from typing import TypeVar

T = TypeVar("T")


class PickType:
    def __init__(self, t: type[T]):
        self.t = t

    def __call__(self, item: dict):
        return {
            key: value for (key, value) in item.items() if isinstance(value, self.t)
        }
