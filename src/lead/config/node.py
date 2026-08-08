"""Config tree building blocks: nodes, derived properties and override resolution."""

import copy
import functools
import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

if TYPE_CHECKING:
    from lead.config.lead_config import LeadConfig

LOG = logging.getLogger(__name__)

ConfigNodeT = TypeVar("ConfigNodeT", bound="ConfigNode")
KnobValueT = TypeVar("KnobValueT")


if TYPE_CHECKING:

    def config_child_node(node_class: type[ConfigNodeT]) -> ConfigNodeT:
        """Declare a child node on a :class:`ConfigNode` subclass.

        Returns the class itself — :meth:`ConfigNode.__init__` instantiates it
        per config instance — typed as an instance so attribute access on the
        tree type-checks. The runtime twin carries no annotations, so beartype
        never sees the deliberate lie while pyright still resolves child
        sections to their node type.

        Args:
            node_class: The child node class.

        Returns:
            The class itself (typed as an instance).
        """
        ...

else:

    def config_child_node(node_class):
        return node_class


class overridable_property(property, Generic[KnobValueT]):  # noqa: N801 — decorator, lowercase like ``property``
    """Derived config default that can still be overridden via profile/env/CLI/file.

    The override value is coerced to the type of the computed default; a failed
    coercion raises instead of silently falling back to the default.
    """

    def __init__(self, fget: Callable[[Any], KnobValueT]) -> None:
        super().__init__(fget)
        self._fget = fget
        self._name = fget.__name__

    @overload
    def __get__(
        self,
        obj: None,
        objtype: type | None = None,
    ) -> "overridable_property[KnobValueT]": ...

    @overload
    def __get__(self, obj: "ConfigNode", objtype: type | None = None) -> KnobValueT: ...

    def __get__(
        self,
        obj: "ConfigNode | None",
        objtype: type | None = None,
    ) -> "KnobValueT | overridable_property[KnobValueT]":
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
    """Coerce an override value to the type of the declared default."""
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

    Annotated class attributes are overridable knobs, :func:`config_child_node`
    attributes are child sections, ``@property`` values are derived and never
    overridable, and ``@overridable_property`` marks a derived default that
    may still be overridden; every node holds ``_root``, the
    :class:`~lead.config.LeadConfig` it belongs to, so derived properties can
    reference other sections of the tree.
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
        for node_class in cls.__mro__:
            for key, value in vars(node_class).items():
                if not key.startswith("_") and key not in attributes:
                    attributes[key] = value
        return attributes

    def __setattr__(self, name: str, value: Any) -> None:
        # Only attributes declared on the class (or private state) may be set, to catch typos.
        if not name.startswith("_") and name not in self._class_attributes():
            raise AttributeError(
                f"Can't set unknown config attribute "
                f"'{type(self).__name__}.{name}'. "
                f"Please check if this variable might have been renamed.",
            )
        super().__setattr__(name, value)

    def child_nodes(self) -> dict[str, "ConfigNode"]:
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
        is_user_override: bool = True,
        raise_on_unknown_key: bool = True,
    ) -> None:
        """Apply a nested override mapping onto this subtree.

        Every key must exactly address a knob at its level of the tree; there
        is no abbreviated addressing.

        Args:
            overrides: Nested mapping of overrides.
            is_user_override: True for env/CLI overrides. Overriding a derived
                ``@property`` then raises instead of being silently skipped.
            raise_on_unknown_key: Whether unknown keys raise. Pass False
                only for stored configs, whose knobs may have been renamed
                since they were written.
        """
        child_nodes = self.child_nodes()
        for key, value in overrides.items():
            if key in child_nodes:
                if not isinstance(value, Mapping):
                    raise TypeError(
                        f"Config section '{key}' expects a mapping, "
                        f"got {type(value).__name__}.",
                    )
                child_nodes[key].apply_overrides(
                    value,
                    is_user_override,
                    raise_on_unknown_key,
                )
            elif key in self._class_attributes():
                self._set_knob(key, value, is_user_override)
            elif raise_on_unknown_key:
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

    def _set_knob(self, key: str, value: Any, is_user_override: bool) -> None:
        """Set a single knob, honoring the property conventions."""
        declared_default = self._class_attributes()[key]
        if isinstance(declared_default, overridable_property):
            self._overrides[key] = value
        elif isinstance(declared_default, property | functools.cached_property):
            # Derived values contained in stored configs are skipped on reload.
            if is_user_override:
                raise AttributeError(
                    f"'{type(self).__name__}.{key}' is derived and cannot be overridden.",
                )
        elif isinstance(declared_default, list) and isinstance(value, Mapping):
            # E.g. ``cameras.0.width=512``: per-index overrides of list knobs
            # would silently replace the list with a dict; use a profile instead.
            raise TypeError(
                f"'{type(self).__name__}.{key}' expects a list; "
                f"override it as a whole (e.g. via a config profile).",
            )
        else:
            setattr(self, key, _coerce(declared_default, value))

    # --- Serialization ---
    def to_dict(self) -> dict[str, Any]:
        """Resolve the subtree into a nested dict of knobs and derived values.

        Properties that raise on access are skipped; values are not filtered
        for serializability (see :func:`~lead.config.yaml_filtered`).
        """
        resolved: dict[str, Any] = {}
        child_nodes = self.child_nodes()
        for key, value in self._class_attributes().items():
            if key in child_nodes:
                resolved[key] = child_nodes[key].to_dict()
            elif isinstance(value, property | functools.cached_property):
                try:
                    resolved[key] = getattr(self, key)
                except Exception:
                    continue
            elif not callable(value):
                resolved[key] = getattr(self, key)
        return resolved
