"""Scene generation from section boundaries with seeded randomness."""

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.dmx.fixtures import Fixture, FixtureType, Rig
from src.dmx.scene import (
    Color,
    Colors,
    Cue,
    FixtureState,
    Position,
    Scene,
    Show,
    TransitionType,
)


# =============================================================================
# Color Palettes for Different Section Types
# =============================================================================

PALETTES: Dict[str, List[Color]] = {
    "intro": [
        Color.from_hex("#1a237e"),  # Deep blue
        Color.from_hex("#311b92"),  # Deep purple
        Color.from_hex("#004d40"),  # Dark teal
        Color.from_hex("#1b5e20"),  # Dark green
    ],
    "verse": [
        Color.from_hex("#1565c0"),  # Blue
        Color.from_hex("#7b1fa2"),  # Purple
        Color.from_hex("#00838f"),  # Cyan
        Color.from_hex("#2e7d32"),  # Green
    ],
    "build": [
        Color.from_hex("#ff6f00"),  # Amber
        Color.from_hex("#ff1744"),  # Red accent
        Color.from_hex("#ffc107"),  # Yellow
        Color.from_hex("#ff9100"),  # Orange
    ],
    "drop": [
        Color.from_hex("#ff0000"),  # Red
        Color.from_hex("#00ff00"),  # Green
        Color.from_hex("#0000ff"),  # Blue
        Color.from_hex("#ff00ff"),  # Magenta
        Color.from_hex("#00ffff"),  # Cyan
        Color.from_hex("#ffffff"),  # White
    ],
    "breakdown": [
        Color.from_hex("#4fc3f7"),  # Light blue
        Color.from_hex("#81c784"),  # Light green
        Color.from_hex("#ce93d8"),  # Light purple
        Color.from_hex("#ffab91"),  # Light coral
    ],
    "chorus": [
        Color.from_hex("#e91e63"),  # Pink
        Color.from_hex("#9c27b0"),  # Purple
        Color.from_hex("#2196f3"),  # Blue
        Color.from_hex("#00bcd4"),  # Cyan
    ],
    "outro": [
        Color.from_hex("#263238"),  # Dark gray-blue
        Color.from_hex("#37474f"),  # Gray-blue
        Color.from_hex("#455a64"),  # Medium gray
        Color.from_hex("#1a237e"),  # Deep blue
    ],
}

# Default palette for unknown section types
DEFAULT_PALETTE = PALETTES["breakdown"]


# =============================================================================
# Scene Templates
# =============================================================================

@dataclass
class SceneTemplate:
    """
    Template defining parameters for generating scenes of a section type.

    These parameters are scaled by the global intensity level.
    """
    section_type: str

    # Base parameters (0.0 - 1.0, scaled by intensity)
    base_dimmer: float
    color_palette: List[Color]

    # Strobe settings
    strobe_enabled: bool
    strobe_probability: float  # Chance any fixture gets strobe
    strobe_intensity: float    # Base strobe rate when enabled

    # Movement settings (for moving heads)
    movement_range: float      # 0.0 = static, 1.0 = full range
    movement_speed: float      # How fast to move (affects speed channel)

    # Transition settings
    transition_type: TransitionType
    transition_time: float

    # Smoke settings
    smoke_enabled: bool
    smoke_intensity: float

    # Laser settings
    laser_enabled: bool
    laser_intensity: float

    @classmethod
    def for_section(cls, section_type: str, intensity: float = 0.7) -> "SceneTemplate":
        """
        Create a template for a section type with given intensity.

        Args:
            section_type: One of 'intro', 'verse', 'build', 'drop', 'breakdown', 'chorus', 'outro'
            intensity: Global intensity modifier (0.0 - 1.0)
        """
        palette = PALETTES.get(section_type, DEFAULT_PALETTE)

        templates = {
            "intro": cls(
                section_type="intro",
                base_dimmer=0.3 * intensity,
                color_palette=palette,
                strobe_enabled=False,
                strobe_probability=0.0,
                strobe_intensity=0.0,
                movement_range=0.2,
                movement_speed=0.1,
                transition_type=TransitionType.FADE,
                transition_time=2.0,
                smoke_enabled=True,
                smoke_intensity=0.3 * intensity,
                laser_enabled=False,
                laser_intensity=0.0,
            ),
            "verse": cls(
                section_type="verse",
                base_dimmer=0.5 * intensity,
                color_palette=palette,
                strobe_enabled=False,
                strobe_probability=0.0,
                strobe_intensity=0.0,
                movement_range=0.3,
                movement_speed=0.2,
                transition_type=TransitionType.FADE,
                transition_time=1.0,
                smoke_enabled=False,
                smoke_intensity=0.0,
                laser_enabled=False,
                laser_intensity=0.0,
            ),
            "build": cls(
                section_type="build",
                base_dimmer=0.6 * intensity,
                color_palette=palette,
                strobe_enabled=True,
                strobe_probability=0.3 * intensity,
                strobe_intensity=0.4,
                movement_range=0.5,
                movement_speed=0.4,
                transition_type=TransitionType.EASE_IN,
                transition_time=1.0,
                smoke_enabled=True,
                smoke_intensity=0.5 * intensity,
                laser_enabled=True,
                laser_intensity=0.4 * intensity,
            ),
            "drop": cls(
                section_type="drop",
                base_dimmer=1.0 * intensity,
                color_palette=palette,
                strobe_enabled=True,
                strobe_probability=0.7 * intensity,
                strobe_intensity=0.8,
                movement_range=1.0,
                movement_speed=0.8,
                transition_type=TransitionType.SNAP,
                transition_time=0.0,
                smoke_enabled=True,
                smoke_intensity=0.8 * intensity,
                laser_enabled=True,
                laser_intensity=1.0 * intensity,
            ),
            "breakdown": cls(
                section_type="breakdown",
                base_dimmer=0.4 * intensity,
                color_palette=palette,
                strobe_enabled=False,
                strobe_probability=0.0,
                strobe_intensity=0.0,
                movement_range=0.3,
                movement_speed=0.2,
                transition_type=TransitionType.FADE,
                transition_time=1.5,
                smoke_enabled=True,
                smoke_intensity=0.4 * intensity,
                laser_enabled=False,
                laser_intensity=0.0,
            ),
            "chorus": cls(
                section_type="chorus",
                base_dimmer=0.8 * intensity,
                color_palette=palette,
                strobe_enabled=True,
                strobe_probability=0.4 * intensity,
                strobe_intensity=0.5,
                movement_range=0.7,
                movement_speed=0.6,
                transition_type=TransitionType.FADE,
                transition_time=0.5,
                smoke_enabled=False,
                smoke_intensity=0.0,
                laser_enabled=True,
                laser_intensity=0.6 * intensity,
            ),
            "outro": cls(
                section_type="outro",
                base_dimmer=0.2 * intensity,
                color_palette=palette,
                strobe_enabled=False,
                strobe_probability=0.0,
                strobe_intensity=0.0,
                movement_range=0.1,
                movement_speed=0.05,
                transition_type=TransitionType.EASE_OUT,
                transition_time=3.0,
                smoke_enabled=False,
                smoke_intensity=0.0,
                laser_enabled=False,
                laser_intensity=0.0,
            ),
        }

        return templates.get(section_type, templates["breakdown"])


# =============================================================================
# Scene Generator
# =============================================================================

class SceneGenerator:
    """
    Generates DMX shows from section boundaries with seeded randomness.

    The same seed + boundaries will always produce the same show,
    but different seeds produce different variations.
    """

    def __init__(
        self,
        rig: Rig,
        seed: int,
        intensity: float = 0.7,
    ):
        """
        Args:
            rig: Fixture rig configuration
            seed: Random seed for reproducibility
            intensity: Global intensity level (0.0 - 1.0)
        """
        self.rig = rig
        self.seed = seed
        self.intensity = max(0.0, min(1.0, intensity))
        self.rng = random.Random(seed)

    @staticmethod
    def seed_from_path(path: str) -> int:
        """Generate a reproducible seed from a file path."""
        return int(hashlib.md5(path.encode()).hexdigest()[:8], 16)

    def _get_section_seed(self, section_index: int) -> int:
        """Get a deterministic seed for a specific section."""
        return hash((self.seed, section_index)) & 0xFFFFFFFF

    def generate_show(
        self,
        boundaries: List[float],
        duration: float,
        song_path: str,
        section_types: Optional[List[str]] = None,
    ) -> Show:
        """
        Generate a complete show from boundary timestamps.

        Args:
            boundaries: List of section boundary timestamps in seconds
            duration: Total song duration in seconds
            song_path: Path to the audio file (for metadata)
            section_types: Optional list of section type labels.
                          If not provided, types are inferred from position.

        Returns:
            Complete Show object with cues for each section
        """
        if not boundaries:
            boundaries = [0.0]

        # Ensure we start at 0
        if boundaries[0] > 0.5:
            boundaries = [0.0] + list(boundaries)

        # Infer section types if not provided
        if section_types is None:
            section_types = self._infer_section_types(boundaries, duration)

        # Ensure section_types matches boundaries length
        while len(section_types) < len(boundaries):
            section_types.append("breakdown")

        # Generate cues
        cues = []
        for i, timestamp in enumerate(boundaries):
            # Each section gets its own sub-seed for independent randomness
            section_seed = self._get_section_seed(i)
            section_rng = random.Random(section_seed)

            section_type = section_types[i]
            template = SceneTemplate.for_section(section_type, self.intensity)

            scene = self._generate_scene(
                template=template,
                rng=section_rng,
                scene_name=f"section_{i}_{section_type}",
            )

            cue = Cue(
                timestamp_sec=timestamp,
                scene=scene,
                section_type=section_type,
            )
            cues.append(cue)

        return Show(
            name=f"show_{self.seed}",
            song_path=song_path,
            duration_sec=duration,
            cues=cues,
            seed=self.seed,
        )

    def _infer_section_types(
        self,
        boundaries: List[float],
        duration: float,
    ) -> List[str]:
        """
        Infer section types based on position in song.

        This is a simple heuristic for EDM-style tracks:
        - First 10%: intro
        - Last 10%: outro
        - Middle sections alternate between builds/drops/breakdowns

        Future versions will use ML classification.
        """
        section_types = []
        n = len(boundaries)

        for i, ts in enumerate(boundaries):
            position = ts / duration if duration > 0 else 0

            # First section
            if i == 0:
                section_types.append("intro")
                continue

            # Last section (if we're past 85% of the song)
            if position > 0.85:
                section_types.append("outro")
                continue

            # Early sections (before 25%)
            if position < 0.25:
                # Alternate intro/build
                section_types.append("build" if i % 2 == 1 else "verse")
                continue

            # Middle sections - the meat of the track
            # Use a pattern: build -> drop -> breakdown -> build -> drop...
            middle_index = i - len([b for b in boundaries if b / duration < 0.25])
            pattern = ["build", "drop", "breakdown", "chorus"]
            section_types.append(pattern[middle_index % len(pattern)])

        return section_types

    def _generate_scene(
        self,
        template: SceneTemplate,
        rng: random.Random,
        scene_name: str,
    ) -> Scene:
        """Generate a scene from a template using the given RNG."""
        fixture_states = {}

        # Group fixtures by type for coordinated looks
        moving_heads = self.rig.get_fixtures_by_type(FixtureType.MOVING_HEAD)
        pars = self.rig.get_fixtures_by_type(FixtureType.PAR)
        lasers = self.rig.get_fixtures_by_type(FixtureType.LASER)
        smoke_machines = self.rig.get_fixtures_by_type(FixtureType.SMOKE)

        # Pick colors for this scene (coordinated across fixture groups)
        main_color = rng.choice(template.color_palette)
        accent_color = rng.choice(template.color_palette)

        # Moving heads - coordinated movement and color
        for i, fixture in enumerate(moving_heads):
            # Alternate colors between left/right
            color = main_color if i % 2 == 0 else accent_color

            # Generate position
            position = self._generate_position(template, rng, mirror=(i % 2 == 1))

            # Strobe
            strobe_rate = 0.0
            if template.strobe_enabled and rng.random() < template.strobe_probability:
                strobe_rate = template.strobe_intensity * (0.5 + rng.random() * 0.5)

            fixture_states[fixture.id] = FixtureState(
                dimmer=template.base_dimmer,
                color=color,
                position=position,
                strobe_rate=strobe_rate,
            )

        # Pars - color wash
        for i, fixture in enumerate(pars):
            color = main_color if i % 2 == 0 else accent_color

            strobe_rate = 0.0
            if template.strobe_enabled and rng.random() < template.strobe_probability * 0.5:
                strobe_rate = template.strobe_intensity * 0.7

            fixture_states[fixture.id] = FixtureState(
                dimmer=template.base_dimmer,
                color=color,
                strobe_rate=strobe_rate,
            )

        # Lasers
        for fixture in lasers:
            if template.laser_enabled:
                # Use red/green based on main color warmth
                h, s, v = main_color.to_hsv()
                # Warm colors (red-yellow) = red laser, cool colors = green laser
                if 0 <= h <= 60 or h >= 300:
                    laser_color = Color(255, 0, 0)  # Red laser
                else:
                    laser_color = Color(0, 255, 0)  # Green laser

                fixture_states[fixture.id] = FixtureState(
                    dimmer=template.laser_intensity,
                    color=laser_color,
                    extras={
                        "0": 200,  # DMX mode (usually 150+ for DMX control)
                        "1": rng.randint(0, 255),  # Random pattern
                        "2": rng.randint(64, 192),  # Rotation speed
                    },
                )
            else:
                fixture_states[fixture.id] = FixtureState(dimmer=0.0)

        # Smoke machines
        for fixture in smoke_machines:
            if template.smoke_enabled and template.smoke_intensity > 0:
                fixture_states[fixture.id] = FixtureState(
                    dimmer=1.0,  # Not really a dimmer, but we use it
                    extras={
                        "0": int(template.smoke_intensity * 255),  # Smoke output
                        "1": int(template.smoke_intensity * 200),  # Fan
                    },
                )
            else:
                fixture_states[fixture.id] = FixtureState(
                    dimmer=0.0,
                    extras={"0": 0, "1": 0},
                )

        return Scene(
            name=scene_name,
            fixture_states=fixture_states,
            transition_type=template.transition_type,
            transition_time_sec=template.transition_time,
        )

    def _generate_position(
        self,
        template: SceneTemplate,
        rng: random.Random,
        mirror: bool = False,
    ) -> Position:
        """
        Generate a pan/tilt position based on template settings.

        Args:
            template: Scene template with movement settings
            rng: Random number generator
            mirror: If True, mirror the pan position (for left/right symmetry)
        """
        # Center position
        center_pan = 0.5
        center_tilt = 0.5

        # Random offset within range
        pan_offset = (rng.random() - 0.5) * template.movement_range
        tilt_offset = (rng.random() - 0.5) * template.movement_range * 0.6  # Less tilt range

        pan = center_pan + pan_offset
        tilt = center_tilt + tilt_offset

        # Mirror pan for symmetry
        if mirror:
            pan = 1.0 - pan

        return Position(pan=pan, tilt=tilt)


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_show(
    boundaries: List[float],
    duration: float,
    rig: Rig,
    song_path: str = "unknown",
    seed: Optional[int] = None,
    intensity: float = 0.7,
    section_types: Optional[List[str]] = None,
) -> Show:
    """
    Convenience function to generate a show.

    Args:
        boundaries: Section boundary timestamps in seconds
        duration: Song duration in seconds
        rig: Fixture rig configuration
        song_path: Path to audio file (used for seed if not provided)
        seed: Random seed (derived from song_path if not provided)
        intensity: Global intensity (0.0 - 1.0)
        section_types: Optional section type labels

    Returns:
        Generated Show
    """
    if seed is None:
        seed = SceneGenerator.seed_from_path(song_path)

    generator = SceneGenerator(rig, seed=seed, intensity=intensity)
    return generator.generate_show(
        boundaries=boundaries,
        duration=duration,
        song_path=song_path,
        section_types=section_types,
    )


if __name__ == "__main__":
    from src.dmx.config import load_rig

    # Load rig
    rig = load_rig("user_rig")
    print(f"Loaded rig: {rig.name} ({len(rig.fixtures)} fixtures)")

    # Example boundaries (simulating detected sections)
    boundaries = [0.0, 15.5, 45.2, 75.8, 120.3, 165.0, 195.5, 220.0]
    duration = 240.0

    # Generate show
    show = generate_show(
        boundaries=boundaries,
        duration=duration,
        rig=rig,
        song_path="example_song.wav",
        intensity=0.8,
    )

    print()
    print(show.summary())

    # Show DMX values for first scene
    print("\n" + "=" * 60)
    print("First scene DMX values:")
    print("=" * 60)

    first_scene = show.cues[0].scene
    for fixture in rig.fixtures:
        state = first_scene.get_state(fixture.id)
        if state:
            print(f"\n{fixture.id}:")
            dmx = state.to_dmx_values(fixture.profile)
            for offset, value in sorted(dmx.items()):
                ch = fixture.address + offset
                print(f"  CH {ch:3d}: {value:3d}")

    # Verify reproducibility
    print("\n" + "=" * 60)
    print("Reproducibility test:")
    print("=" * 60)

    show2 = generate_show(
        boundaries=boundaries,
        duration=duration,
        rig=rig,
        song_path="example_song.wav",
        intensity=0.8,
    )

    # Compare section types
    types1 = [c.section_type for c in show.cues]
    types2 = [c.section_type for c in show2.cues]
    print(f"Same section types: {types1 == types2}")

    # Compare first scene colors
    s1_color = show.cues[0].scene.get_state("mh_left").color
    s2_color = show2.cues[0].scene.get_state("mh_left").color
    print(f"Same colors: {s1_color == s2_color}")
    print(f"Seeds match: {show.seed == show2.seed}")
