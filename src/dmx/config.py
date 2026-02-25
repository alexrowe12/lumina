"""YAML configuration loader for fixtures and rigs."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from src.dmx.fixtures import (
    ChannelMapping,
    ChannelType,
    Fixture,
    FixtureProfile,
    FixtureType,
    Rig,
)


# Default directories
DEFAULT_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
DEFAULT_RIGS_DIR = Path(__file__).parent.parent.parent / "rigs"


def _parse_channel_type(type_str: str) -> ChannelType:
    """Convert string to ChannelType enum."""
    type_map = {
        "dimmer": ChannelType.DIMMER,
        "red": ChannelType.RED,
        "green": ChannelType.GREEN,
        "blue": ChannelType.BLUE,
        "white": ChannelType.WHITE,
        "amber": ChannelType.AMBER,
        "uv": ChannelType.UV,
        "pan": ChannelType.PAN,
        "tilt": ChannelType.TILT,
        "pan_fine": ChannelType.PAN_FINE,
        "tilt_fine": ChannelType.TILT_FINE,
        "speed": ChannelType.SPEED,
        "gobo": ChannelType.GOBO,
        "gobo_rotation": ChannelType.GOBO_ROTATION,
        "color_wheel": ChannelType.COLOR_WHEEL,
        "shutter": ChannelType.SHUTTER,
        "focus": ChannelType.FOCUS,
        "prism": ChannelType.PRISM,
        "laser_pattern": ChannelType.LASER_PATTERN,
        "laser_rotation": ChannelType.LASER_ROTATION,
        "laser_mode": ChannelType.LASER_MODE,
        "smoke_output": ChannelType.SMOKE_OUTPUT,
        "smoke_fan": ChannelType.SMOKE_FAN,
    }
    type_lower = type_str.lower()
    if type_lower not in type_map:
        raise ValueError(f"Unknown channel type: {type_str}")
    return type_map[type_lower]


def _parse_fixture_type(type_str: str) -> FixtureType:
    """Convert string to FixtureType enum."""
    type_map = {
        "moving_head": FixtureType.MOVING_HEAD,
        "par": FixtureType.PAR,
        "laser": FixtureType.LASER,
        "smoke": FixtureType.SMOKE,
        "strobe": FixtureType.STROBE,
        "wash": FixtureType.WASH,
    }
    type_lower = type_str.lower()
    if type_lower not in type_map:
        raise ValueError(f"Unknown fixture type: {type_str}")
    return type_map[type_lower]


def load_fixture_profile(
    path: Union[str, Path],
    fixtures_dir: Optional[Path] = None,
) -> FixtureProfile:
    """
    Load a fixture profile from a YAML file.

    Args:
        path: Path to YAML file, or just the profile name (without .yaml)
        fixtures_dir: Directory to search for profiles (default: fixtures/)

    Returns:
        FixtureProfile instance
    """
    fixtures_dir = fixtures_dir or DEFAULT_FIXTURES_DIR
    path = Path(path)

    # If just a name, look in fixtures directory
    if not path.suffix:
        path = fixtures_dir / f"{path}.yaml"
    elif not path.is_absolute():
        path = fixtures_dir / path

    if not path.exists():
        raise FileNotFoundError(f"Fixture profile not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Parse channels
    channels = []
    for ch_data in data.get("channels", []):
        channel = ChannelMapping(
            channel_type=_parse_channel_type(ch_data["type"]),
            offset=ch_data["offset"],
            min_value=ch_data.get("min", 0),
            max_value=ch_data.get("max", 255),
            default_value=ch_data.get("default", 0),
            invert=ch_data.get("invert", False),
        )
        channels.append(channel)

    return FixtureProfile(
        name=data["name"],
        manufacturer=data.get("manufacturer", "Unknown"),
        fixture_type=_parse_fixture_type(data["fixture_type"]),
        channel_count=data["channel_count"],
        channels=channels,
    )


def load_rig(
    path: Union[str, Path],
    fixtures_dir: Optional[Path] = None,
    rigs_dir: Optional[Path] = None,
) -> Rig:
    """
    Load a rig configuration from a YAML file.

    Args:
        path: Path to rig YAML file, or just the rig name (without .yaml)
        fixtures_dir: Directory to search for fixture profiles
        rigs_dir: Directory to search for rig configs (default: rigs/)

    Returns:
        Rig instance with all fixtures loaded
    """
    fixtures_dir = fixtures_dir or DEFAULT_FIXTURES_DIR
    rigs_dir = rigs_dir or DEFAULT_RIGS_DIR
    path = Path(path)

    # If just a name, look in rigs directory
    if not path.suffix:
        path = rigs_dir / f"{path}.yaml"
    elif not path.is_absolute():
        path = rigs_dir / path

    if not path.exists():
        raise FileNotFoundError(f"Rig config not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Cache loaded profiles to avoid reloading
    profile_cache: Dict[str, FixtureProfile] = {}

    def get_profile(profile_name: str) -> FixtureProfile:
        if profile_name not in profile_cache:
            profile_cache[profile_name] = load_fixture_profile(
                profile_name, fixtures_dir
            )
        return profile_cache[profile_name]

    # Parse fixtures
    fixtures = []
    for fix_data in data.get("fixtures", []):
        profile = get_profile(fix_data["profile"])

        position = None
        if "position" in fix_data:
            pos = fix_data["position"]
            position = (pos[0], pos[1], pos[2]) if len(pos) >= 3 else None

        fixture = Fixture(
            id=fix_data["id"],
            profile=profile,
            universe=fix_data.get("universe", 0),
            address=fix_data["address"],
            position=position,
        )
        fixtures.append(fixture)

    rig = Rig(
        name=data.get("name", "Unnamed Rig"),
        fixtures=fixtures,
        universes=data.get("universes", 1),
    )

    # Validate
    errors = rig.validate()
    if errors:
        raise ValueError(f"Rig validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return rig


def save_fixture_profile(profile: FixtureProfile, path: Union[str, Path]) -> None:
    """Save a fixture profile to a YAML file."""
    data = {
        "name": profile.name,
        "manufacturer": profile.manufacturer,
        "fixture_type": profile.fixture_type.name.lower(),
        "channel_count": profile.channel_count,
        "channels": [
            {
                "type": ch.channel_type.name.lower(),
                "offset": ch.offset,
                "min": ch.min_value,
                "max": ch.max_value,
                "default": ch.default_value,
                **({"invert": True} if ch.invert else {}),
            }
            for ch in profile.channels
        ],
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def save_rig(rig: Rig, path: Union[str, Path]) -> None:
    """Save a rig configuration to a YAML file."""
    data = {
        "name": rig.name,
        "universes": rig.universes,
        "fixtures": [
            {
                "id": f.id,
                "profile": f.profile.name.lower().replace(" ", "_"),
                "universe": f.universe,
                "address": f.address,
                **({"position": list(f.position)} if f.position else {}),
            }
            for f in rig.fixtures
        ],
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    # Test loading
    print("Testing config loader...")
    print(f"Fixtures dir: {DEFAULT_FIXTURES_DIR}")
    print(f"Rigs dir: {DEFAULT_RIGS_DIR}")

    # List available profiles
    if DEFAULT_FIXTURES_DIR.exists():
        profiles = list(DEFAULT_FIXTURES_DIR.glob("*.yaml"))
        print(f"\nAvailable fixture profiles: {len(profiles)}")
        for p in profiles:
            print(f"  - {p.stem}")

    # List available rigs
    if DEFAULT_RIGS_DIR.exists():
        rigs = list(DEFAULT_RIGS_DIR.glob("*.yaml"))
        print(f"\nAvailable rig configs: {len(rigs)}")
        for r in rigs:
            print(f"  - {r.stem}")

    # Try loading example rig
    try:
        rig = load_rig("user_rig")
        print(f"\nLoaded rig: {rig.name}")
        print(f"Fixtures: {len(rig.fixtures)}")
        for f in rig.fixtures:
            print(f"  - {f.id}: {f.profile.name} @ ch {f.address}")
    except FileNotFoundError as e:
        print(f"\nNo example rig found (expected): {e}")
