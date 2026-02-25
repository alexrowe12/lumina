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
    # Config
    "load_fixture_profile",
    "load_rig",
    "save_fixture_profile",
    "save_rig",
    # Generator
    "SceneTemplate",
    "SceneGenerator",
    "generate_show",
    "PALETTES",
    # Art-Net
    "ArtNetOutput",
    "ShowPlayer",
]


def __getattr__(name):
    """Lazy import attributes on first access."""
    if name in ("FixtureType", "ChannelType", "ChannelMapping", "FixtureProfile", "Fixture", "Rig"):
        from src.dmx import fixtures
        return getattr(fixtures, name)
    elif name in ("Color", "Colors", "Position", "FixtureState", "Scene", "Cue", "Show", "TransitionType"):
        from src.dmx import scene
        return getattr(scene, name)
    elif name in ("load_fixture_profile", "load_rig", "save_fixture_profile", "save_rig"):
        from src.dmx import config
        return getattr(config, name)
    elif name in ("SceneTemplate", "SceneGenerator", "generate_show", "PALETTES"):
        from src.dmx import generator
        return getattr(generator, name)
    elif name in ("ArtNetOutput", "ShowPlayer"):
        from src.dmx import artnet
        return getattr(artnet, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
