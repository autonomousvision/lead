"""Property decorators that cache a value until the simulation step or a key changes."""

import typing


def step_cached_property(func: typing.Callable) -> property:
    """Decorator to cache the result of a method based on the current step.
    This is useful for properties that are expensive to compute and
    should only be recalculated when the step changes.

    Args:
        func: Function to be decorated.

    Returns:
        A property that caches its value based on the step attribute of the instance.
    """
    cache_attr = f"_{func.__name__}_cache"
    step_attr = f"_{func.__name__}_step"

    def getter(self):
        if getattr(self, step_attr, None) != self.step:
            setattr(self, cache_attr, func(self))
            setattr(self, step_attr, self.step)
        return getattr(self, cache_attr)

    return property(getter)


def cached_property_by(key_getter: typing.Callable):
    """Decorator to cache the result of a method based on a custom key.
    This is useful for properties that are expensive to compute and should only be
    recalculated when a specific attribute or value changes.

    Args:
        key_getter: Function that takes the instance and returns the cache key.

    Returns:
        A decorator that creates a property cached by the key_getter result.

    Example:
        @cached_property_by(lambda self: self.privileged_route_planner.route_index)
        def some_expensive_calculation(self):
            return expensive_computation()
    """

    def decorator(func: typing.Callable) -> property:
        cache_attr = f"_{func.__name__}_cache"
        key_attr = f"_{func.__name__}_cache_key"

        def getter(self):
            current_key = key_getter(self)
            if getattr(self, key_attr, None) != current_key:
                setattr(self, cache_attr, func(self))
                setattr(self, key_attr, current_key)
            return getattr(self, cache_attr)

        return property(getter)

    return decorator
