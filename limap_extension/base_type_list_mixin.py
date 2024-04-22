from abc import ABCMeta, abstractmethod
from typing import List, TypeVar, Generic, Optional, Iterable, Iterator
from collections import UserList

import torch

from .base_type_mixin import BaseTypeMixin

# Generic for dynamic type hinting of derived classes.
T = TypeVar('T', bound=BaseTypeMixin)


# Generic[T] means that we can use the type template that BaseTypeList was initialized with in the
# class body.
class BaseTypeList(UserList, Generic[T], metaclass=ABCMeta):
    """The base type that all CDCPD-type lists inherit from."""
    data: List[T]

    def __init__(self, initlist: Optional[Iterable[T]] = None):
        self._is_torch: bool = None
        self._verify_initlist_type(initlist)
        super().__init__(initlist)

    @abstractmethod
    def _get_item_type(self) -> type(T):
        """Returns the item type stored in this list

        The reason this is an abstract method is so that we can maintain MyItemListBase as an
        abstract class while also being able to enforce the storage of only a single type in the
        list.
        """
        pass

    def append(self, item: T):
        self._verify_item(item)
        super().append(item)

    def __setitem__(self, i, item):
        self._verify_item(item)
        super().__setitem__(i, item)

    def __iter__(self) -> Iterator[T]:
        """Overloaded to provide type hinting of derived classes"""
        return super().__iter__()

    def __getitem__(self, i) -> T:
        """Overloaded to provide type hinting of derived classes"""
        return super().__getitem__(i)

    def is_torch(self) -> Optional[bool]:
        """Returns None if empty, True if all items are torch, False if all numpy"""
        return self._is_torch

    def to_torch(self, dtype: torch.dtype = torch.double, device: torch.device = "cpu"):
        """Convert all stored objects to PyTorch tensors"""
        for obj in self.data:
            obj.to_torch(dtype=dtype, device=device)

    def to_numpy(self):
        """Convert all stored objects to NumPy arrays"""
        for obj in self.data:
            obj.to_numpy()

    def copy(self) -> 'BaseTypeList[T]':
        """Returns a copy of this list"""
        return self.__class__([obj.copy() for obj in self])

    def _is_item_correct_type(self, item: T) -> bool:
        type_expected = self._get_item_type()
        return isinstance(item, type_expected)

    def _verify_item_type(self, item: T):
        if not self._is_item_correct_type(item):
            raise TypeError(
                f"Expected item to be of type {self._get_item_type()} but got {type(item)}")

    def _is_item_correct_torchyness(self, item: T) -> bool:
        """Checks that the item holds tensors if this list holds tensors. Same for numpy"""
        if (self._is_torch is None) or (self._is_initialized() and (len(self) == 0)):
            self._is_torch = item.is_torch
            return True
        return (item.is_torch == self._is_torch)

    def _verify_item_torchyness(self, item: T):
        if not self._is_item_correct_torchyness(item):
            raise TypeError(
                f"item.is_torch ({item.is_torch}) != self.is_torch() ({self.is_torch()})")

    def _check_item_attributes(self, item: T):
        """Method to override if type in list requires other checks

        An example of this is PointClouds needing to all have/not have RGB values.

        NOTE: This could probably be better accomplished with relying on a method in BaseTypeMixin
        to check that two BaseTypeMixin-derived classes are "similar".
        """
        return True

    def _verify_item_attributes(self, item: T):
        if not self._check_item_attributes(item):
            raise TypeError("Item's attributes didn't match expected attributes")

    def _verify_item(self, item: T):
        self._verify_item_type(item)
        self._verify_item_torchyness(item)
        self._verify_item_attributes(item)

    def _verify_initlist_type(self, initlist: Optional[Iterable[T]]):
        """Verifies that the constructor argument is of the expected type"""
        if isinstance(initlist, Iterable):
            # Empty iterable is okay since UserList.__init__ handles empty iterables.

            # Keeping this around as this is more informative and might switch to this error
            # message.
            # iterable_types = {type(item) for item in iterable}
            # if (len(iterable) > 0) and (iterable_types != {item_type_expected}):
            #     raise TypeError(
            #         f"Expected constructor argument to be iterable of type {item_type_expected} "
            #         f"but got iterable with types: {iterable_types}")

            # So we just check that each item is the expected type.
            for item in initlist:
                self._verify_item(item)
        elif initlist is None:
            # This is okay as UserList.__init__ handles None type.
            pass
        else:
            raise TypeError(
                f"Expected constructor argument to be iterable of type {self._get_item_type()}")

    def _is_initialized(self):
        return hasattr(self, "data")
