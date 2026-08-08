"""Encode one cached sample tensor into a self-describing record.

Array values are compressed by a per-name codec; everything else (scalars,
strings, bools) rides a passthrough lane. A record names its own codec, so it
decodes without the config that wrote it and a cache outlives that config.
"""

from __future__ import annotations

import abc
import io
import pickle
import zlib
from collections.abc import Mapping
from typing import Any, Final, Literal, TypedDict

import imagecodecs
import numpy as np
from jaxtyping import Shaped, UInt8
from PIL import Image

# Field keys inside an encoded record, kept short because they repeat once per
# stored tensor. The values persist in every built store — never change them.
# Declared ``Final`` without an explicit type so they carry their literal type
# and can index the record TypedDicts below.
_CODEC_FIELD: Final = "c"
_DTYPE_FIELD: Final = "d"
_SHAPE_FIELD: Final = "s"
_BYTES_FIELD: Final = "b"
_PARAMS_FIELD: Final = "p"
_PAYLOAD_FIELD: Final = "v"

_PASSTHROUGH: Final = "passthrough"

# What a sample tensor may be. Arrays take a codec; the rest — including the
# numpy scalars a builder returns, which are not ``ndarray`` — ride passthrough.
CacheableValue = np.ndarray | np.generic | str | int | float | bool | None
# An array of any dtype and shape: a codec is generic over both.
AnyArray = Shaped[np.ndarray, "..."]
# A codec entry of the policy config: a codec name, or the name plus its kwargs.
CodecSpec = str | Mapping[str, Any]


class EncodedTensor(TypedDict):
    """One array's stored record: codec name, dtype string, shape and payload bytes."""

    c: str
    d: str
    s: tuple[int, ...]
    b: bytes


class EncodedImageTensor(EncodedTensor):
    """An image-codec record, carrying the quantization that decoding must undo."""

    p: dict[str, float]


class PassthroughRecord(TypedDict):
    """A non-array value stored verbatim."""

    c: str
    v: CacheableValue


# The layouts PIL round-trips through: greyscale, or channel-last RGB.
PilLayout = Shaped[np.ndarray, "h w"] | Shaped[np.ndarray, "h w 3"]


class TensorCodec(abc.ABC):
    """Encodes a single numpy array to bytes and back."""

    codec_name: str

    @abc.abstractmethod
    def encode(self, array: AnyArray) -> EncodedTensor:
        """Encode one array.

        Args:
            array: The array to encode.

        Returns:
            A self-describing record holding the array's bytes and metadata.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, encoded_record: Mapping[str, Any]) -> AnyArray:
        """Decode one record produced by :meth:`encode`.

        Args:
            encoded_record: The record to decode.

        Returns:
            The reconstructed array.
        """
        raise NotImplementedError


def encode_tensor(
    value: CacheableValue,
    codec_spec: CodecSpec,
) -> bytes:
    """Encode one sample tensor (or plain value) into its stored record.

    Args:
        value: The tensor or plain value to store.
        codec_spec: The codec of the tensor, e.g. ``"raw"`` or
            ``{"name": "png", "quantization_scale": 1}``.

    Returns:
        The pickled record.
    """
    encoded_record: EncodedTensor | PassthroughRecord
    if isinstance(value, np.ndarray):
        encoded_record = build_tensor_codec(codec_spec).encode(value)
    else:
        encoded_record = {_CODEC_FIELD: _PASSTHROUGH, _PAYLOAD_FIELD: value}
    return pickle.dumps(encoded_record, protocol=pickle.HIGHEST_PROTOCOL)


def decode_tensor(tensor_blob: bytes) -> CacheableValue:
    """Rebuild one stored tensor. The record names its own codec.

    Args:
        tensor_blob: The pickled record.

    Returns:
        The tensor or plain value.
    """
    encoded_record = pickle.loads(tensor_blob)
    if encoded_record[_CODEC_FIELD] == _PASSTHROUGH:
        return encoded_record[_PAYLOAD_FIELD]
    return build_tensor_codec(encoded_record[_CODEC_FIELD]).decode(encoded_record)


def _dtype_of(encoded_record: Mapping[str, Any]) -> np.dtype:
    return np.dtype(encoded_record[_DTYPE_FIELD])


class RawCodec(TensorCodec):
    """Stores the array's buffer verbatim. Lossless for every dtype, the safe default."""

    codec_name = "raw"

    def encode(self, array: AnyArray) -> EncodedTensor:
        """Inherited, see superclass."""
        return {
            _CODEC_FIELD: self.codec_name,
            _DTYPE_FIELD: array.dtype.str,
            _SHAPE_FIELD: tuple(array.shape),
            _BYTES_FIELD: np.ascontiguousarray(array).tobytes(),
        }

    def decode(self, encoded_record: Mapping[str, Any]) -> AnyArray:
        """Inherited, see superclass."""
        # np.frombuffer returns a read-only view over the record; downstream
        # needs it writable.
        array = np.frombuffer(
            encoded_record[_BYTES_FIELD],
            dtype=_dtype_of(encoded_record),
        ).reshape(encoded_record[_SHAPE_FIELD])
        return array.copy()


class ZlibCodec(TensorCodec):
    """zlib-compressed raw buffer. Lossless for every dtype; shrinks sparse arrays.

    The default level 1 already collapses the mostly-zero label grids; higher
    levels buy little there and cost encode time.
    """

    codec_name = "zlib"

    def __init__(self, level: int = 1) -> None:
        self._compression_level = int(level)

    def encode(self, array: AnyArray) -> EncodedTensor:
        """Inherited, see superclass."""
        return {
            _CODEC_FIELD: self.codec_name,
            _DTYPE_FIELD: array.dtype.str,
            _SHAPE_FIELD: tuple(array.shape),
            _BYTES_FIELD: zlib.compress(
                np.ascontiguousarray(array).tobytes(),
                self._compression_level,
            ),
        }

    def decode(self, encoded_record: Mapping[str, Any]) -> AnyArray:
        """Inherited, see superclass."""
        return (
            np.frombuffer(
                zlib.decompress(encoded_record[_BYTES_FIELD]),
                dtype=_dtype_of(encoded_record),
            )
            .reshape(encoded_record[_SHAPE_FIELD])
            .copy()
        )


def _to_pil_layout(array: AnyArray) -> PilLayout:
    """Reshape a (H, W), (1, H, W) or (3, H, W) array into what PIL expects."""
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 3 and array.shape[0] == 3:
        return np.transpose(array, (1, 2, 0))
    raise ValueError(
        f"image codecs need (H, W), (1, H, W) or (3, H, W); got {array.shape}",
    )


def _decode_image_bytes(image_bytes: bytes, kind: Literal["png", "jpeg"]) -> PilLayout:
    decode = imagecodecs.png_decode if kind == "png" else imagecodecs.jpeg8_decode
    return decode(image_bytes)


def _from_pil_layout(image: PilLayout, shape: tuple[int, ...]) -> AnyArray:
    """Invert :func:`_to_pil_layout` back to the array's original shape."""
    if len(shape) == 3 and shape[0] == 3:
        image = np.transpose(image, (2, 0, 1))
    return image.reshape(shape)


class _ImageCodec(TensorCodec):
    """Shared 8-bit value quantization: ``u8 = round(value * quantization_scale)``.

    Maps the array's values onto uint8 pixels, never its spatial size:
    ``quantization_scale=255`` for an array normalized to [0, 1];
    ``quantization_scale=1`` for one already holding uint8-ranged values
    (raw images, small integer class ids).
    """

    def __init__(self, quantization_scale: float | int = 255.0) -> None:
        self._quantization_scale = float(quantization_scale)

    def _quantize(self, array: AnyArray) -> UInt8[np.ndarray, ...]:
        return np.clip(
            np.rint(array.astype(np.float64) * self._quantization_scale),
            0,
            255,
        ).astype(np.uint8)

    def _dequantize(
        self,
        image: PilLayout,
        encoded_record: Mapping[str, Any],
    ) -> AnyArray:
        quantization_scale = encoded_record[_PARAMS_FIELD]["quantization_scale"]
        array = _from_pil_layout(image, tuple(encoded_record[_SHAPE_FIELD]))
        if quantization_scale != 1.0:
            array = array.astype(np.float64) / quantization_scale
        return array.astype(_dtype_of(encoded_record))


class PngCodec(_ImageCodec):
    """Lossless PNG. Raises on arrays that 8-bit quantization would not round-trip exactly.

    That check is the point: silently quantizing an array whose values matter (class ids,
    occupancy rasters) would corrupt training data.
    """

    codec_name = "png"

    def encode(self, array: AnyArray) -> EncodedImageTensor:
        """Inherited, see superclass."""
        quantized = self._quantize(array)
        round_tripped = (
            quantized.astype(np.float64) / self._quantization_scale
            if self._quantization_scale != 1.0
            else quantized
        )
        if not np.allclose(round_tripped.astype(array.dtype), array, rtol=0, atol=0):
            raise ValueError(
                f"PngCodec(quantization_scale={self._quantization_scale}) is not "
                f"lossless for this array "
                f"(dtype={array.dtype}, range=[{array.min()}, {array.max()}]). "
                f"Use quantization_scale=1 for integer-valued arrays, 255 for "
                f"[0, 1] arrays, or fall back to the raw codec.",
            )
        png_buffer = io.BytesIO()
        Image.fromarray(_to_pil_layout(quantized)).save(
            png_buffer,
            format="PNG",
            optimize=False,
        )
        return {
            _CODEC_FIELD: self.codec_name,
            _DTYPE_FIELD: array.dtype.str,
            _SHAPE_FIELD: tuple(array.shape),
            _BYTES_FIELD: png_buffer.getvalue(),
            _PARAMS_FIELD: {"quantization_scale": self._quantization_scale},
        }

    def decode(self, encoded_record: Mapping[str, Any]) -> AnyArray:
        """Inherited, see superclass."""
        image = _decode_image_bytes(encoded_record[_BYTES_FIELD], "png")
        return self._dequantize(image, encoded_record)


class JpegCodec(_ImageCodec):
    """Lossy JPEG, for natural images only. Never for arrays whose exact values carry meaning."""

    codec_name = "jpeg"

    def __init__(
        self,
        quality: int = 90,
        quantization_scale: float | int = 255.0,
    ) -> None:
        super().__init__(quantization_scale=quantization_scale)
        self._quality = int(quality)

    def encode(self, array: AnyArray) -> EncodedImageTensor:
        """Inherited, see superclass."""
        quantized = self._quantize(array)
        jpeg_buffer = io.BytesIO()
        Image.fromarray(_to_pil_layout(quantized)).save(
            jpeg_buffer,
            format="JPEG",
            quality=self._quality,
        )
        return {
            _CODEC_FIELD: self.codec_name,
            _DTYPE_FIELD: array.dtype.str,
            _SHAPE_FIELD: tuple(array.shape),
            _BYTES_FIELD: jpeg_buffer.getvalue(),
            _PARAMS_FIELD: {
                "quantization_scale": self._quantization_scale,
                "quality": self._quality,
            },
        }

    def decode(self, encoded_record: Mapping[str, Any]) -> AnyArray:
        """Inherited, see superclass."""
        image = _decode_image_bytes(encoded_record[_BYTES_FIELD], "jpeg")
        return self._dequantize(image, encoded_record)


_CODEC_CLASSES_BY_NAME: Final[dict[str, type]] = {
    RawCodec.codec_name: RawCodec,
    ZlibCodec.codec_name: ZlibCodec,
    PngCodec.codec_name: PngCodec,
    JpegCodec.codec_name: JpegCodec,
}


def build_tensor_codec(codec_spec: CodecSpec) -> TensorCodec:
    """Build one codec from a config entry, e.g. ``{"name": "jpeg", "quality": 90}`` or ``"raw"``.

    Args:
        codec_spec: Codec name, or a mapping of the name plus that codec's
            keyword arguments.

    Returns:
        The constructed codec.
    """
    if isinstance(codec_spec, str):
        codec_spec = {"name": codec_spec}
    kwargs = {key: value for key, value in codec_spec.items() if key != "name"}
    name = codec_spec["name"]
    if name not in _CODEC_CLASSES_BY_NAME:
        raise ValueError(
            f"unknown codec {name!r}; available: {sorted(_CODEC_CLASSES_BY_NAME)}",
        )
    return _CODEC_CLASSES_BY_NAME[name](**kwargs)
