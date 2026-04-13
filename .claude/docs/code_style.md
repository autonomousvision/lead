# Code style

## Typing

- Annotate every parameter and return type, including `-> None`
- Modern union syntax: `str | None`, `list | None`
- Never use bare `list`, `dict`, `tuple` — always parameterise: `list[int]`, `dict[str, float]`, `tuple[float, carla.VehicleControl, list[str] | None]`. Try to maximize precision.
- Domain types: `carla.Actor`, `carla.Transform`,  `numbers.Real`
- `jaxtyping` and `numpy.typing` aka. `npt` for shaped arrays: `jt.Float[npt.NDArray, "N 3"]`, `jt.UInt8[npt.NDArray, "H W"]`
- `@beartype` on public/important private methods, matching existing usage in the file

## Docstrings

Google style, no types in docstrings (types go in the signature). Keep descriptions concise. The `Returns` section should describe the return value(s) without types.

Add docstrings or type annotations to code you changed and leave other code parts untouched.
