"""Plug drones, rovers and sensors into Polaris through one interface. See devices/README.md."""

from .registry import (
    KINDS,
    DeviceSpec,
    DeviceUnavailable,
    Fleet,
    TaggedMap,
    build,
    fusion_accepts_ids,
    register,
)
from .roles import Locator, MapSource, Mobile, roles_of

__all__ = [
    'KINDS',
    'DeviceSpec',
    'DeviceUnavailable',
    'Fleet',
    'Locator',
    'MapSource',
    'Mobile',
    'TaggedMap',
    'build',
    'fusion_accepts_ids',
    'register',
    'roles_of',
]
