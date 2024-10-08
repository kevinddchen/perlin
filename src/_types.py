from typing import Protocol, Self


class ArrayLike(Protocol):

    def __add__(self, other: Self | float, /) -> Self: ...
    def __sub__(self, other: Self | float, /) -> Self: ...
    def __mul__(self, other: Self | float, /) -> Self: ...
    def __rmul__(self, other: Self | float, /) -> Self: ...
