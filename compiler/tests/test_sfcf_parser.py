"""Seam tests for SFCF binary format parsing with hand-crafted byte sequences.

Tests low-level binary readers and ``parse_sfcf_blob()`` from
``compiler.sfcf_parser`` using struct.pack to construct minimal valid
SFCF blobs. No compiled dylib, no sf-dialect, no torch.
"""

import struct

from compiler.sfcf_parser import (
    _read_str,
    _read_u8,
    _read_u32,
    _read_u64,
    parse_sfcf_blob,
)


class TestBinaryReaders:
    """Verify low-level binary readers decode little-endian data correctly."""

    def test_read_u8(self) -> None:
        """Read a single unsigned byte."""
        data = b"\x2a\xff\x00"
        val, pos = _read_u8(data, 0)
        assert val == 0x2a
        val, pos = _read_u8(data, pos)
        assert val == 0xff

    def test_read_u32(self) -> None:
        """Read a little-endian unsigned 32-bit integer."""
        data = struct.pack("<I", 0xDEADBEEF)
        val, pos = _read_u32(data, 0)
        assert val == 0xDEADBEEF
        assert pos == 4

    def test_read_u64(self) -> None:
        """Read a little-endian unsigned 64-bit integer."""
        data = struct.pack("<Q", 123456789012345)
        val, pos = _read_u64(data, 0)
        assert val == 123456789012345
        assert pos == 8

    def test_read_str(self) -> None:
        """Read a length-prefixed UTF-8 string."""
        s = "hello_sfcf"
        encoded = s.encode("utf-8")
        data = struct.pack("<I", len(encoded)) + encoded
        val, pos = _read_str(data, 0)
        assert val == s
        assert pos == 4 + len(encoded)


class TestParseSfcfBlob:
    """Verify ``parse_sfcf_blob()`` parses a valid minimal SFCF v2 blob."""

    def _make_minimal_blob(
        self,
        name_mappings: list[tuple[str, str]] | None = None,
        version: int = 2,
    ) -> bytes:
        """Build a minimal SFCF binary blob."""
        if name_mappings is None:
            name_mappings = [("w0", "model.w0")]

        parts: list[bytes] = []
        # Magic (4 bytes)
        parts.append(b"SFCF")
        # Version (u32 LE)
        parts.append(struct.pack("<I", version))

        # Name mappings
        parts.append(struct.pack("<I", len(name_mappings)))
        for compiled_key, hf_key in name_mappings:
            cbytes = compiled_key.encode("utf-8")
            hbytes = hf_key.encode("utf-8")
            parts.append(struct.pack("<I", len(cbytes)) + cbytes)
            parts.append(struct.pack("<I", len(hbytes)) + hbytes)

        # Constants count (0 — no constants for minimal blob)
        parts.append(struct.pack("<I", 0))

        return b"".join(parts)

    def test_parses_magic_and_version(self) -> None:
        """Verify SFCF magic 'SFCF' and version field are read correctly."""
        blob = self._make_minimal_blob(version=2)
        name_map, constants, pos, ver = parse_sfcf_blob(blob)
        assert ver == 2
        assert pos > 0  # advanced past header + name mappings

    def test_parses_single_name_mapping(self) -> None:
        """Verify single name mapping is parsed correctly."""
        blob = self._make_minimal_blob([("w_proj", "model.layers.0.w_proj")])
        name_map, constants, pos, ver = parse_sfcf_blob(blob)
        assert name_map == {"w_proj": "model.layers.0.w_proj"}
        assert constants == {}

    def test_parses_multiple_name_mappings(self) -> None:
        """Verify multiple name mappings are parsed in order."""
        mappings = [
            ("w0", "a.b.c"),
            ("w1", "d.e.f"),
            ("w2", "g.h.i"),
        ]
        blob = self._make_minimal_blob(mappings)
        name_map, constants, pos, ver = parse_sfcf_blob(blob)
        assert len(name_map) == 3
        assert name_map["w0"] == "a.b.c"
        assert name_map["w1"] == "d.e.f"
        assert name_map["w2"] == "g.h.i"
