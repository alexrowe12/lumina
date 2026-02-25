"""Scene and cue data structures for DMX show generation."""

import colorsys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from src.dmx.fixtures import FixtureProfile, ChannelType


class TransitionType(Enum):
    """How to transition into a scene."""
    SNAP = auto()        # Instant change
    FADE = auto()        # Linear crossfade
    EASE_IN = auto()     # Slow start, fast end
    EASE_OUT = auto()    # Fast start, slow end
    EASE_IN_OUT = auto() # Slow start and end


@dataclass
class Color:
    """RGB color with utility methods for conversion."""
    r: int  # 0-255
    g: int  # 0-255
    b: int  # 0-255
    w: int = 0  # Optional white channel (0-255)

    @classmethod
    def from_hsv(cls, h: float, s: float, v: float) -> "Color":
        """
        Create color from HSV values.

        Args:
            h: Hue (0-360)
            s: Saturation (0-1)
            v: Value/brightness (0-1)
        """
        r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
        return cls(int(r * 255), int(g * 255), int(b * 255))

    @classmethod
    def from_hex(cls, hex_str: str) -> "Color":
        """Create color from hex string (e.g., '#FF0000' or 'FF0000')."""
        hex_str = hex_str.lstrip("#")
        return cls(
            int(hex_str[0:2], 16),
            int(hex_str[2:4], 16),
            int(hex_str[4:6], 16),
        )

    def to_hex(self) -> str:
        """Convert to hex string (e.g., '#FF0000')."""
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def to_hsv(self) -> tuple:
        """Convert to HSV tuple (h: 0-360, s: 0-1, v: 0-1)."""
        h, s, v = colorsys.rgb_to_hsv(self.r / 255.0, self.g / 255.0, self.b / 255.0)
        return (h * 360, s, v)

    def to_tuple(self) -> tuple:
        """Return as (r, g, b, w) tuple."""
        return (self.r, self.g, self.b, self.w)

    def scaled(self, factor: float) -> "Color":
        """Return a new color scaled by factor (for dimming)."""
        return Color(
            int(self.r * factor),
            int(self.g * factor),
            int(self.b * factor),
            int(self.w * factor),
        )

    def __repr__(self) -> str:
        if self.w > 0:
            return f"Color({self.r}, {self.g}, {self.b}, w={self.w})"
        return f"Color({self.r}, {self.g}, {self.b})"


# Common colors
class Colors:
    """Predefined color constants."""
    RED = Color(255, 0, 0)
    GREEN = Color(0, 255, 0)
    BLUE = Color(0, 0, 255)
    WHITE = Color(255, 255, 255)
    WARM_WHITE = Color(255, 200, 150, w=255)
    CYAN = Color(0, 255, 255)
    MAGENTA = Color(255, 0, 255)
    YELLOW = Color(255, 255, 0)
    ORANGE = Color(255, 128, 0)
    PURPLE = Color(128, 0, 255)
    PINK = Color(255, 100, 200)
    BLACK = Color(0, 0, 0)


@dataclass
class Position:
    """Pan/tilt position for moving heads (normalized 0-1)."""
    pan: float   # 0.0 = full left, 1.0 = full right
    tilt: float  # 0.0 = full down, 1.0 = full up

    def __post_init__(self):
        """Clamp values to 0-1 range."""
        self.pan = max(0.0, min(1.0, self.pan))
        self.tilt = max(0.0, min(1.0, self.tilt))

    @classmethod
    def center(cls) -> "Position":
        """Return center position."""
        return cls(0.5, 0.5)

    def to_dmx(self) -> tuple:
        """Convert to 8-bit DMX values (0-255)."""
        return (int(self.pan * 255), int(self.tilt * 255))


@dataclass
class FixtureState:
    """
    Complete state of a single fixture at a point in time.

    This is a semantic representation - actual DMX values are computed
    based on the fixture's profile.
    """
    dimmer: float = 1.0                # 0.0-1.0
    color: Optional[Color] = None
    position: Optional[Position] = None
    strobe_rate: float = 0.0           # 0.0 = off, 1.0 = max speed
    gobo: int = 0                      # Gobo wheel position (0 = open)
    extras: Dict[str, int] = field(default_factory=dict)  # Other raw channel values

    def to_dmx_values(self, profile: FixtureProfile) -> Dict[int, int]:
        """
        Convert this state to DMX channel values based on fixture profile.

        Args:
            profile: The fixture profile defining channel layout

        Returns:
            Dict mapping channel offset (0-indexed) to DMX value (0-255)
        """
        values = {}

        # Dimmer
        if profile.has_dimmer:
            ch = profile.get_channel(ChannelType.DIMMER)
            if ch:
                val = int(self.dimmer * 255)
                if ch.invert:
                    val = 255 - val
                values[ch.offset] = max(ch.min_value, min(ch.max_value, val))

        # Color (RGB/RGBW)
        if self.color and profile.has_color:
            color_channels = [
                (ChannelType.RED, self.color.r),
                (ChannelType.GREEN, self.color.g),
                (ChannelType.BLUE, self.color.b),
                (ChannelType.WHITE, self.color.w),
            ]
            for ct, val in color_channels:
                ch = profile.get_channel(ct)
                if ch:
                    values[ch.offset] = max(ch.min_value, min(ch.max_value, val))

        # Position (moving heads)
        if self.position and profile.has_movement:
            pan_ch = profile.get_channel(ChannelType.PAN)
            tilt_ch = profile.get_channel(ChannelType.TILT)
            if pan_ch:
                val = int(self.position.pan * 255)
                if pan_ch.invert:
                    val = 255 - val
                values[pan_ch.offset] = val
            if tilt_ch:
                val = int(self.position.tilt * 255)
                if tilt_ch.invert:
                    val = 255 - val
                values[tilt_ch.offset] = val

        # Strobe
        if self.strobe_rate > 0 and profile.has_strobe:
            ch = profile.get_channel(ChannelType.SHUTTER)
            if ch:
                # Typically 0 = open, higher values = faster strobe
                # Map strobe_rate (0-1) to a strobe range (e.g., 16-255)
                strobe_min = 16
                strobe_max = 255
                val = int(strobe_min + self.strobe_rate * (strobe_max - strobe_min))
                values[ch.offset] = val
        elif profile.has_strobe:
            # No strobe - set to open
            ch = profile.get_channel(ChannelType.SHUTTER)
            if ch:
                values[ch.offset] = 8  # Typically "open" value

        # Gobo
        if self.gobo > 0:
            ch = profile.get_channel(ChannelType.GOBO)
            if ch:
                values[ch.offset] = self.gobo

        # Extra raw values
        for offset_str, val in self.extras.items():
            offset = int(offset_str) if isinstance(offset_str, str) else offset_str
            values[offset] = val

        return values


@dataclass
class Scene:
    """
    A lighting scene defining the state of all fixtures.
    """
    name: str
    fixture_states: Dict[str, FixtureState]  # fixture_id -> state
    transition_type: TransitionType = TransitionType.FADE
    transition_time_sec: float = 0.5

    def get_state(self, fixture_id: str) -> Optional[FixtureState]:
        """Get the state for a fixture, or None if not defined."""
        return self.fixture_states.get(fixture_id)

    def set_state(self, fixture_id: str, state: FixtureState) -> None:
        """Set the state for a fixture."""
        self.fixture_states[fixture_id] = state

    def set_all_dimmer(self, dimmer: float) -> None:
        """Set dimmer for all fixtures in this scene."""
        for state in self.fixture_states.values():
            state.dimmer = dimmer

    def set_all_color(self, color: Color) -> None:
        """Set color for all fixtures in this scene."""
        for state in self.fixture_states.values():
            state.color = color


@dataclass
class Cue:
    """A timed cue in the show timeline."""
    timestamp_sec: float
    scene: Scene
    section_type: Optional[str] = None  # 'intro', 'build', 'drop', etc.

    def __repr__(self) -> str:
        return f"Cue({self.timestamp_sec:.2f}s, {self.scene.name}, {self.section_type})"


@dataclass
class Show:
    """Complete show with timeline of cues."""
    name: str
    song_path: str
    duration_sec: float
    cues: List[Cue]
    seed: int  # For reproducibility

    def get_cue_at_time(self, time_sec: float) -> Optional[Cue]:
        """Get the active cue at a given timestamp."""
        active_cue = None
        for cue in self.cues:
            if cue.timestamp_sec <= time_sec:
                active_cue = cue
            else:
                break
        return active_cue

    def get_scene_at_time(self, time_sec: float) -> Optional[Scene]:
        """Get the active scene at a given timestamp."""
        cue = self.get_cue_at_time(time_sec)
        return cue.scene if cue else None

    def get_section_boundaries(self) -> List[float]:
        """Get list of all cue timestamps."""
        return [cue.timestamp_sec for cue in self.cues]

    def summary(self) -> str:
        """Return a summary string of the show."""
        lines = [
            f"Show: {self.name}",
            f"Song: {self.song_path}",
            f"Duration: {self.duration_sec:.1f}s",
            f"Seed: {self.seed}",
            f"Cues: {len(self.cues)}",
            "",
            "Timeline:",
        ]
        for i, cue in enumerate(self.cues):
            minutes = int(cue.timestamp_sec // 60)
            secs = cue.timestamp_sec % 60
            section = cue.section_type or "unknown"
            lines.append(f"  {i+1:2d}. {minutes}:{secs:05.2f} - {cue.scene.name} [{section}]")
        return "\n".join(lines)


if __name__ == "__main__":
    # Test the module
    from src.dmx.fixtures import create_example_rig

    rig = create_example_rig()

    # Create a simple scene
    scene = Scene(
        name="test_scene",
        fixture_states={},
        transition_type=TransitionType.FADE,
        transition_time_sec=1.0,
    )

    # Set states for each fixture
    for fixture in rig.fixtures:
        if fixture.profile.has_color:
            state = FixtureState(
                dimmer=0.8,
                color=Colors.BLUE,
                position=Position.center() if fixture.profile.has_movement else None,
            )
        else:
            state = FixtureState(dimmer=0.5)
        scene.set_state(fixture.id, state)

    # Print DMX values for each fixture
    print("Scene:", scene.name)
    print("Transition:", scene.transition_type.name, f"({scene.transition_time_sec}s)")
    print()

    for fixture in rig.fixtures:
        state = scene.get_state(fixture.id)
        if state:
            dmx_values = state.to_dmx_values(fixture.profile)
            print(f"{fixture.id} (addr {fixture.address}):")
            for offset, value in sorted(dmx_values.items()):
                abs_ch = fixture.address + offset
                print(f"  CH {abs_ch:3d} (offset {offset:2d}): {value:3d}")
            print()

    # Create a simple show
    show = Show(
        name="test_show",
        song_path="test.wav",
        duration_sec=180.0,
        cues=[
            Cue(0.0, scene, section_type="intro"),
            Cue(30.0, scene, section_type="build"),
            Cue(60.0, scene, section_type="drop"),
        ],
        seed=12345,
    )

    print(show.summary())
