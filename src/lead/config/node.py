"""Config tree building blocks: nodes, derived properties and override resolution."""

import copy
import functools
import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Generic, TypeVar, no_type_check, overload

if TYPE_CHECKING:
    from lead.config.lead_config import LeadConfig

LOG = logging.getLogger(__name__)

NodeT = TypeVar("NodeT", bound="ConfigNode")
ValueT = TypeVar("ValueT")

_MISSING = object()


@no_type_check
def child_node(node_class: type[NodeT]) -> NodeT:
    """Declare a child node on a :class:`ConfigNode` subclass.

    Returns the class itself; :meth:`ConfigNode.__init__` instantiates it per
    config instance. Typed as an instance so attribute access on the tree
    type-checks.

    The instance annotation is a deliberate lie for the static checker, so the
    function is exempt from runtime type checking (``LEAD_RUNTIME_TYPE_CHECKING``);
    beartype would otherwise reject the class it actually returns.

    Args:
        node_class: The child node class.

    Returns:
        The class itself (typed as an instance).
    """
    return node_class  # type: ignore[return-value]


class overridable_property(property, Generic[ValueT]):  # noqa: N801 — decorator, lowercase like ``property``
    """Derived config default that can still be overridden via profile/env/CLI/file.

    The override value is coerced to the type of the computed default; a failed
    coercion raises instead of silently falling back to the default.
    """

    def __init__(self, fget: Callable[[Any], ValueT]) -> None:
        super().__init__(fget)
        self._fget = fget
        self._name = fget.__name__

    @overload
    def __get__(
        self,
        obj: None,
        objtype: type | None = None,
    ) -> "overridable_property[ValueT]": ...

    @overload
    def __get__(self, obj: "ConfigNode", objtype: type | None = None) -> ValueT: ...

    def __get__(
        self,
        obj: "ConfigNode | None",
        objtype: type | None = None,
    ) -> "ValueT | overridable_property[ValueT]":
        if obj is None:
            return self
        if self._name in obj._overrides:
            override = obj._overrides[self._name]
            try:
                default = self._fget(obj)
            except Exception:
                # The default may not be computable (e.g. it reads an unset
                # environment); the override then applies uncoerced.
                return override
            return _coerce(default, override)
        return self._fget(obj)


def _coerce(default: Any, value: Any) -> Any:
    """Coerce an override value to the type of the declared default.

    Args:
        default: The declared default the override replaces.
        value: The override value.

    Returns:
        The coerced value.
    """
    if default is None or value is None or isinstance(value, type(default)):
        return value
    if isinstance(default, bool):
        if isinstance(value, str):
            raise TypeError(f"Expected a boolean override, got '{value}'.")
        return bool(value)
    if isinstance(default, int):
        if isinstance(value, float) and value != int(value):
            raise TypeError(f"Expected an integer override, got {value}.")
        # Also reconstructs int-enum knobs from their serialized values.
        return type(default)(value)
    if isinstance(default, float | str):
        return type(default)(value)
    if isinstance(default, tuple):
        return tuple(value)
    return value


class ConfigNode:
    """Node of the config tree.

    Attribute conventions:
        - Annotated class attribute: a tunable knob, overridable via
          profile/env/CLI/file.
        - Class attribute declared with :func:`child_node`: a child section.
        - ``@property``: a derived value, never overridable.
        - ``@overridable_property``: a derived default that may still be
          overridden. Use only when both apply.

    Every node holds ``_root``, the :class:`~lead.config.LeadConfig` it belongs
    to, so derived properties can reference other sections of the tree.
    """

    _root: "LeadConfig"
    _overrides: dict[str, Any]

    def __init__(self, root: "LeadConfig | None" = None) -> None:
        object.__setattr__(self, "_overrides", {})
        object.__setattr__(self, "_root", root if root is not None else self)
        for key, value in self._class_attributes().items():
            if isinstance(value, type) and issubclass(value, ConfigNode):
                object.__setattr__(self, key, value(root=self._root))
            elif isinstance(value, list | dict):
                # Per-instance copies so mutating one config (or a dump of it)
                # can never change the class defaults shared by other instances.
                object.__setattr__(self, key, copy.deepcopy(value))

    @classmethod
    def _class_attributes(cls) -> dict[str, Any]:
        """Public class attributes over the MRO, most-derived definition first."""
        attributes: dict[str, Any] = {}
        for klass in cls.__mro__:
            for key, value in vars(klass).items():
                if not key.startswith("_") and key not in attributes:
                    attributes[key] = value
        return attributes

    def __setattr__(self, name: str, value: Any) -> None:
        # Guard against typos and renamed knobs: only attributes declared on
        # the class (or private state) may be set.
        if not name.startswith("_") and name not in self._class_attributes():
            raise AttributeError(
                f"Can't set unknown config attribute "
                f"'{type(self).__name__}.{name}'. "
                f"Please check if this variable might have been renamed.",
            )
        super().__setattr__(name, value)

    def children(self) -> dict[str, "ConfigNode"]:
        """Child nodes by attribute name."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if isinstance(value, ConfigNode)
        }

    # --- Override application ---
    def apply_overrides(
        self,
        overrides: Mapping[str, Any],
        user_override: bool = True,
        raise_error_on_missing_key: bool = True,
    ) -> None:
        """Apply a nested override mapping onto this subtree.

        Every key must exactly address a knob at its level of the tree; there
        is no abbreviated addressing.

        Args:
            overrides: Nested mapping of overrides.
            user_override: True for env/CLI overrides. Overriding a derived
                ``@property`` then raises instead of being silently skipped.
            raise_error_on_missing_key: Whether unknown keys raise. Pass False
                only for stored configs, whose knobs may have been renamed
                since they were written.
        """
        children = self.children()
        for key, value in overrides.items():
            if key in children:
                if not isinstance(value, Mapping):
                    raise TypeError(
                        f"Config section '{key}' expects a mapping, "
                        f"got {type(value).__name__}.",
                    )
                children[key].apply_overrides(
                    value,
                    user_override,
                    raise_error_on_missing_key,
                )
            elif key in self._class_attributes():
                self._set_leaf(key, value, user_override)
            elif raise_error_on_missing_key:
                raise AttributeError(
                    f"Unknown configuration key '{key}' "
                    f"in section '{type(self).__name__}'.",
                )
            else:
                LOG.warning(
                    "Ignoring unknown configuration key '%s' in section '%s'.",
                    key,
                    type(self).__name__,
                )

    def _set_leaf(self, key: str, value: Any, user_override: bool) -> None:
        """Set a single knob, honoring the property conventions.

        Args:
            key: Attribute name on this node.
            value: Override value.
            user_override: True for env/CLI overrides. Overriding a derived
                ``@property`` then raises instead of being silently ignored.
        """
        class_value = self._class_attributes()[key]
        if isinstance(class_value, overridable_property):
            self._overrides[key] = value
        elif isinstance(class_value, property | functools.cached_property):
            # Derived values contained in stored configs are skipped on reload.
            if user_override:
                raise AttributeError(
                    f"'{type(self).__name__}.{key}' is derived and cannot be overridden.",
                )
        elif isinstance(class_value, list) and isinstance(value, Mapping):
            # E.g. ``cameras.0.width=512``: per-index overrides of list knobs
            # would silently replace the list with a dict; use a profile instead.
            raise TypeError(
                f"'{type(self).__name__}.{key}' expects a list; "
                f"override it as a whole (e.g. via a config profile).",
            )
        else:
            setattr(self, key, _coerce(class_value, value))

    # --- Serialization ---
    def to_dict(self) -> dict[str, Any]:
        """Resolve the subtree into a nested dict of knobs and derived values.

        Properties that raise on access are skipped; values are not filtered
        for serializability (see :func:`~lead.config.yaml_filtered`).
        """
        out: dict[str, Any] = {}
        children = self.children()
        for key, value in self._class_attributes().items():
            if key in children:
                out[key] = children[key].to_dict()
            elif isinstance(value, property | functools.cached_property):
                try:
                    out[key] = getattr(self, key)
                except Exception:
                    continue
            elif not callable(value):
                out[key] = getattr(self, key)
        return out
