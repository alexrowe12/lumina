"""Fixture abstraction layer for DMX lighting control."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


class FixtureType(Enum):
    """Categories of lighting fixtures with different capabilities."""
    MOVING_HEAD = auto()
    PAR = auto()
    LASER = auto()
    SMOKE = auto()
    STROBE = auto()
    WASH = auto()


class ChannelType(Enum):
    """Semantic channel types for fixture profiles."""
    # Intensity
    DIMMER = auto()

    # Color (RGB/RGBW/RGBA)
    RED = auto()
    GREEN = auto()
    BLUE = auto()
    WHITE = auto()
    AMBER = auto()
    UV = auto()

    # Movement (moving heads)
    PAN = auto()
    TILT = auto()
    PAN_FINE = auto()
    TILT_FINE = auto()
    SPEED = auto()

    # Effects
    GOBO = auto()
    GOBO_ROTATION = auto()
    COLOR_WHEEL = auto()
    SHUTTER = auto()
    FOCUS = auto()
    PRISM = auto()

    # Laser specific
    LASER_PATTERN = auto()
    LASER_ROTATION = auto()
    LASER_MODE = auto()

    # Smoke/fog
    SMOKE_OUTPUT = auto()
    SMOKE_FAN = auto()


@dataclass
class ChannelMapping:
    """Maps a semantic channel type to a DMX offset and value range."""
    channel_type: ChannelType
    offset: int  # 0-indexed offset from fixture's base address
    min_value: int = 0
    max_value: int = 255
    default_value: int = 0
    invert: bool = False  # Some fixtures have inverted value ranges


@dataclass
class FixtureProfile:
    """
    Defines a fixture model's DMX channel layout and capabilities.

    This is the "template" - it describes what a fixture model CAN do,
    not a specific instance in a rig.
    """
    name: str
    manufacturer: str
    fixture_type: FixtureType
    channel_count: int
    channels: List[ChannelMapping]

    # Capability flags (computed in __post_init__)
    has_color: bool = field(init=False, default=False)
    has_movement: bool = field(init=False, default=False)
    has_dimmer: bool = field(init=False, default=False)
    has_strobe: bool = field(init=False, default=False)

    def __post_init__(self):
        """Derive capability flags from channel list."""
        channel_types = {c.channel_type for c in self.channels}

        self.has_color = bool(channel_types & {
            ChannelType.RED, ChannelType.GREEN, ChannelType.BLUE,
            ChannelType.COLOR_WHEEL
        })
        self.has_movement = bool(channel_types & {
            ChannelType.PAN, ChannelType.TILT
        })
        self.has_dimmer = ChannelType.DIMMER in channel_types
        self.has_strobe = ChannelType.SHUTTER in channel_types

    def get_channel(self, channel_type: ChannelType) -> Optional[ChannelMapping]:
        """Get channel mapping by type, or None if not present."""
        for ch in self.channels:
            if ch.channel_type == channel_type:
                return ch
        return None

    def get_channels_by_types(self, *channel_types: ChannelType) -> List[ChannelMapping]:
        """Get all channels matching any of the given types."""
        type_set = set(channel_types)
        return [ch for ch in self.channels if ch.channel_type in type_set]


@dataclass
class Fixture:
    """
    A specific fixture instance in a rig.

    Combines a profile (what the fixture CAN do) with a DMX address
    (where it IS in the universe).
    """
    id: str  # User-defined identifier (e.g., "mh_left", "par_1")
    profile: FixtureProfile
    universe: int  # Art-Net universe (0-indexed)
    address: int  # DMX start address (1-512)
    position: Optional[Tuple[float, float, float]] = None  # x, y, z for visualization

    @property
    def channel_range(self) -> Tuple[int, int]:
        """Return (start, end) DMX channel range (1-indexed, inclusive)."""
        return (self.address, self.address + self.profile.channel_count - 1)

    def get_absolute_channel(self, channel_type: ChannelType) -> Optional[int]:
        """Get the absolute DMX channel number for a channel type."""
        ch = self.profile.get_channel(channel_type)
        if ch:
            return self.address + ch.offset
        return None


@dataclass
class Rig:
    """Complete fixture rig configuration."""
    name: str
    fixtures: List[Fixture]
    universes: int = 1

    def get_fixtures_by_type(self, fixture_type: FixtureType) -> List[Fixture]:
        """Get all fixtures of a given type."""
        return [f for f in self.fixtures if f.profile.fixture_type == fixture_type]

    def get_fixture_by_id(self, fixture_id: str) -> Optional[Fixture]:
        """Get a fixture by its ID."""
        for f in self.fixtures:
            if f.id == fixture_id:
                return f
        return None

    def get_all_fixture_ids(self) -> List[str]:
        """Get list of all fixture IDs."""
        return [f.id for f in self.fixtures]

    def validate(self) -> List[str]:
        """
        Validate rig configuration, return list of errors.
        Empty list means valid.
        """
        errors = []

        # Check for duplicate IDs
        ids = [f.id for f in self.fixtures]
        if len(ids) != len(set(ids)):
            errors.append("Duplicate fixture IDs found")

        # Check for channel overlaps within each universe
        for universe in range(self.universes):
            universe_fixtures = [f for f in self.fixtures if f.universe == universe]
            for i, f1 in enumerate(universe_fixtures):
                for f2 in universe_fixtures[i+1:]:
                    r1 = f1.channel_range
                    r2 = f2.channel_range
                    if r1[0] <= r2[1] and r2[0] <= r1[1]:
                        errors.append(
                            f"Channel overlap: {f1.id} ({r1[0]}-{r1[1]}) and "
                            f"{f2.id} ({r2[0]}-{r2[1]}) in universe {universe}"
                        )

        # Check address bounds
        for f in self.fixtures:
            if f.address < 1 or f.address > 512:
                errors.append(f"Fixture {f.id}: address {f.address} out of range (1-512)")
            if f.channel_range[1] > 512:
                errors.append(f"Fixture {f.id}: channels exceed 512")

        return errors


# =============================================================================
# Factory functions for common fixture profiles
# =============================================================================

def create_generic_moving_head() -> FixtureProfile:
    """Create a generic 16-channel moving head profile."""
    return FixtureProfile(
        name="Generic Moving Head 16CH",
        manufacturer="Generic",
        fixture_type=FixtureType.MOVING_HEAD,
        channel_count=16,
        channels=[
            ChannelMapping(ChannelType.PAN, offset=0),
            ChannelMapping(ChannelType.PAN_FINE, offset=1),
            ChannelMapping(ChannelType.TILT, offset=2),
            ChannelMapping(ChannelType.TILT_FINE, offset=3),
            ChannelMapping(ChannelType.SPEED, offset=4, default_value=0),
            ChannelMapping(ChannelType.DIMMER, offset=5),
            ChannelMapping(ChannelType.SHUTTER, offset=6),
            ChannelMapping(ChannelType.RED, offset=7),
            ChannelMapping(ChannelType.GREEN, offset=8),
            ChannelMapping(ChannelType.BLUE, offset=9),
            ChannelMapping(ChannelType.WHITE, offset=10),
            ChannelMapping(ChannelType.COLOR_WHEEL, offset=11),
            ChannelMapping(ChannelType.GOBO, offset=12),
            ChannelMapping(ChannelType.GOBO_ROTATION, offset=13),
            ChannelMapping(ChannelType.PRISM, offset=14),
            ChannelMapping(ChannelType.FOCUS, offset=15),
        ],
    )


def create_generic_par() -> FixtureProfile:
    """Create a generic 8-channel RGBW par profile."""
    return FixtureProfile(
        name="Generic Par RGBW 8CH",
        manufacturer="Generic",
        fixture_type=FixtureType.PAR,
        channel_count=8,
        channels=[
            ChannelMapping(ChannelType.DIMMER, offset=0),
            ChannelMapping(ChannelType.RED, offset=1),
            ChannelMapping(ChannelType.GREEN, offset=2),
            ChannelMapping(ChannelType.BLUE, offset=3),
            ChannelMapping(ChannelType.WHITE, offset=4),
            ChannelMapping(ChannelType.SHUTTER, offset=5),
            ChannelMapping(ChannelType.COLOR_WHEEL, offset=6),
            ChannelMapping(ChannelType.SPEED, offset=7),
        ],
    )


def create_generic_laser() -> FixtureProfile:
    """Create a generic 6-channel laser profile."""
    return FixtureProfile(
        name="Generic Laser 6CH",
        manufacturer="Generic",
        fixture_type=FixtureType.LASER,
        channel_count=6,
        channels=[
            ChannelMapping(ChannelType.LASER_MODE, offset=0),
            ChannelMapping(ChannelType.LASER_PATTERN, offset=1),
            ChannelMapping(ChannelType.LASER_ROTATION, offset=2),
            ChannelMapping(ChannelType.RED, offset=3),
            ChannelMapping(ChannelType.GREEN, offset=4),
            ChannelMapping(ChannelType.DIMMER, offset=5),
        ],
    )


def create_generic_smoke() -> FixtureProfile:
    """Create a generic 2-channel smoke machine profile."""
    return FixtureProfile(
        name="Generic Smoke Machine 2CH",
        manufacturer="Generic",
        fixture_type=FixtureType.SMOKE,
        channel_count=2,
        channels=[
            ChannelMapping(ChannelType.SMOKE_OUTPUT, offset=0),
            ChannelMapping(ChannelType.SMOKE_FAN, offset=1),
        ],
    )


def create_example_rig() -> Rig:
    """Create an example rig matching the user's setup."""
    mh_profile = create_generic_moving_head()
    par_profile = create_generic_par()
    laser_profile = create_generic_laser()
    smoke_profile = create_generic_smoke()

    return Rig(
        name="Example Rig",
        universes=1,
        fixtures=[
            Fixture("mh_left", mh_profile, universe=0, address=1, position=(-2.0, 3.0, 0.0)),
            Fixture("mh_right", mh_profile, universe=0, address=17, position=(2.0, 3.0, 0.0)),
            Fixture("par_left", par_profile, universe=0, address=33, position=(-1.5, 2.0, 0.0)),
            Fixture("par_right", par_profile, universe=0, address=41, position=(1.5, 2.0, 0.0)),
            Fixture("laser_center", laser_profile, universe=0, address=49, position=(0.0, 4.0, 0.0)),
            Fixture("smoke_main", smoke_profile, universe=0, address=55, position=(0.0, 0.0, 0.0)),
        ],
    )


if __name__ == "__main__":
    # Test the module
    rig = create_example_rig()

    print(f"Rig: {rig.name}")
    print(f"Universes: {rig.universes}")
    print(f"Fixtures: {len(rig.fixtures)}")
    print()

    for fixture in rig.fixtures:
        profile = fixture.profile
        print(f"  {fixture.id}:")
        print(f"    Type: {profile.fixture_type.name}")
        print(f"    Channels: {fixture.channel_range[0]}-{fixture.channel_range[1]} ({profile.channel_count} ch)")
        print(f"    Capabilities: color={profile.has_color}, movement={profile.has_movement}, "
              f"dimmer={profile.has_dimmer}, strobe={profile.has_strobe}")

    # Validate
    errors = rig.validate()
    if errors:
        print("\nValidation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\nRig validation: OK")
