"""DMX scene generation package for Lumina."""

# Lazy imports to avoid circular import issues when running submodules directly
__all__ = [
    # Fixtures
    "FixtureType",
    "ChannelType",
    "ChannelMapping",
    "FixtureProfile",
    "Fixture",
    "Rig",
    # Scene
    "Color",
    "Colors",
    "Position",
    "FixtureState",
    "Scene",
    "Cue",
    "Show",
    "TransitionType",
]


def __getattr__(name):
    """Lazy import attributes on first access."""
    if name in ("FixtureType", "ChannelType", "ChannelMapping", "FixtureProfile", "Fixture", "Rig"):
        from src.dmx import fixtures
        return getattr(fixtures, name)
    elif name in ("Color", "Colors", "Position", "FixtureState", "Scene", "Cue", "Show", "TransitionType"):
        from src.dmx import scene
        return getattr(scene, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
