from typing import List
from numpy.random import randint, random_sample
from math import ceil


class ValueType:
    def __init__(self, name: str, size: int, can_be_optional: bool = False):
        self.name = name
        self.size = size
        self.can_be_optional = can_be_optional

    def encoding_size(self):
        return self.size + (1 if self.can_be_optional else 0)

    def value_in_bounds(self, value):
        raise NotImplementedError("Must be implemented by a subclass")

    def uniform_random_value(self):
        raise NotImplementedError("Must be implemented by a subclass")


class Continuous(ValueType):
    def __init__(self, name, bounds, can_be_optional=False):
        super().__init__(name, 1, can_be_optional)
        self.bounds = bounds

    def value_in_bounds(self, value):
        min_value, max_value = self.bounds
        return min_value <= value <= max_value

    def uniform_random_value(self):
        l, h = self.bounds
        return (h - l) * random_sample() + l

    def __str__(self):
        l, h = self.bounds
        return f"{self.name}: {'optional ' if self.can_be_optional else ''}Continuous[{l}, {h}]"


class Discrete(ValueType):
    def __init__(self, name, bounds, increment=1, can_be_optional=False):
        super().__init__(name, 1, can_be_optional)
        self.bounds = bounds
        self.increment = increment

    def value_in_bounds(self, value):
        min_value, max_value = self.bounds
        return min_value <= value <= max_value

    def uniform_random_value(self):
        min_value, max_value = self.bounds
        num_values = ceil((max_value - min_value) / self.increment)
        return int(min_value + self.increment * randint(0, num_values))

    def __str__(self):
        l, h = self.bounds
        return f"{self.name}: {'optional ' if self.can_be_optional else ''}Discrete[{l}:{self.increment}:{h}]"


class Boolean(ValueType):
    def __init__(self, name, can_be_optional=False):
        super().__init__(name, 1, can_be_optional)

    def value_in_bounds(self, value):
        return value in [True, False]

    def uniform_random_value(self):
        return True if random_sample() < 0.5 else False

    def __str__(self):
        return f"{self.name}: {'optional ' if self.can_be_optional else ''}Boolean"


class Categorical(ValueType):
    def __init__(self, name, values: List[str], can_be_optional=False):
        super().__init__(name, len(values), can_be_optional)
        self.values = values

    def value_in_bounds(self, value):
        return value in self.values

    def uniform_random_value(self):
        return self.values[randint(0, len(self.values))]

    def __str__(self):
        return f"{self.name}: {'optional ' if self.can_be_optional else ''}Categorical[{len(self.values)} values]"
