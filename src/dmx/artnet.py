"""Art-Net DMX output for QLC+ and other visualizers."""

import socket
import struct
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.dmx.fixtures import Rig
from src.dmx.scene import Scene, Show


# =============================================================================
# Art-Net Protocol Constants
# =============================================================================

ARTNET_PORT = 6454
ARTNET_HEADER = b"Art-Net\x00"
ARTNET_OPCODE_DMX = 0x5000  # ArtDmx opcode
ARTNET_PROTOCOL_VERSION = 14

# DMX constants
DMX_CHANNELS_PER_UNIVERSE = 512
DMX_FRAME_RATE = 40  # Standard DMX refresh rate (Hz)
DMX_FRAME_TIME = 1.0 / DMX_FRAME_RATE  # ~25ms


# =============================================================================
# Art-Net Output
# =============================================================================

class ArtNetOutput:
    """
    Art-Net DMX output over UDP.

    Sends Art-Net packets to a target IP (default: localhost for QLC+).
    Can be used for visualization or real DMX output via Art-Net nodes.

    Art-Net packet structure (ArtDmx):
    - Header: 'Art-Net\\0' (8 bytes)
    - OpCode: 0x5000 (2 bytes, little-endian)
    - Protocol Version: 14 (2 bytes, big-endian)
    - Sequence: 0-255 (1 byte, increments per packet)
    - Physical: 0 (1 byte, physical port)
    - Universe: 0-32767 (2 bytes, little-endian)
    - Length: DMX data length (2 bytes, big-endian)
    - Data: DMX channel values (1-512 bytes)
    """

    def __init__(
        self,
        target_ip: str = "127.0.0.1",
        port: int = ARTNET_PORT,
        broadcast: bool = False,
    ):
        """
        Initialize Art-Net output.

        Args:
            target_ip: IP address to send packets to (default: localhost)
            port: UDP port (default: 6454)
            broadcast: Whether to broadcast packets (for multiple receivers)
        """
        self.target_ip = target_ip
        self.port = port
        self.broadcast = broadcast
        self.sequence = 0

        # Universe data cache (for partial updates)
        self._universe_data: Dict[int, bytearray] = {}

        # Create UDP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if broadcast:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        self._closed = False

    def send_dmx(self, universe: int, channels: Dict[int, int]) -> None:
        """
        Send DMX data for a universe.

        Args:
            universe: Art-Net universe (0-indexed)
            channels: Dict mapping channel number (1-512) to value (0-255)
        """
        if self._closed:
            raise RuntimeError("ArtNetOutput is closed")

        # Get or create universe data buffer
        if universe not in self._universe_data:
            self._universe_data[universe] = bytearray(DMX_CHANNELS_PER_UNIVERSE)

        dmx_data = self._universe_data[universe]

        # Update channel values
        for channel, value in channels.items():
            if 1 <= channel <= DMX_CHANNELS_PER_UNIVERSE:
                dmx_data[channel - 1] = max(0, min(255, value))

        # Build and send packet
        packet = self._build_artdmx_packet(universe, dmx_data)
        self.socket.sendto(packet, (self.target_ip, self.port))

        # Increment sequence
        self.sequence = (self.sequence + 1) % 256

    def send_raw(self, universe: int, data: bytes) -> None:
        """
        Send raw DMX data for a universe.

        Args:
            universe: Art-Net universe (0-indexed)
            data: Raw DMX data (up to 512 bytes)
        """
        if self._closed:
            raise RuntimeError("ArtNetOutput is closed")

        # Pad or truncate to 512 bytes
        dmx_data = bytearray(DMX_CHANNELS_PER_UNIVERSE)
        dmx_data[: len(data)] = data[:DMX_CHANNELS_PER_UNIVERSE]

        # Cache and send
        self._universe_data[universe] = dmx_data
        packet = self._build_artdmx_packet(universe, dmx_data)
        self.socket.sendto(packet, (self.target_ip, self.port))

        self.sequence = (self.sequence + 1) % 256

    def send_frame(self, rig: Rig, scene: Scene) -> None:
        """
        Render a scene to Art-Net for all fixtures in a rig.

        Args:
            rig: Fixture rig configuration
            scene: Scene to render
        """
        # Collect channel values per universe
        universe_channels: Dict[int, Dict[int, int]] = {}

        for fixture_id, state in scene.fixture_states.items():
            fixture = rig.get_fixture_by_id(fixture_id)
            if not fixture:
                continue

            # Get DMX values from state
            dmx_offsets = state.to_dmx_values(fixture.profile)

            # Convert to absolute channel numbers
            if fixture.universe not in universe_channels:
                universe_channels[fixture.universe] = {}

            for offset, value in dmx_offsets.items():
                abs_channel = fixture.address + offset
                universe_channels[fixture.universe][abs_channel] = value

        # Send each universe
        for universe, channels in universe_channels.items():
            self.send_dmx(universe, channels)

    def blackout(self, universes: int = 1) -> None:
        """
        Send blackout (all zeros) to specified number of universes.

        Args:
            universes: Number of universes to blackout
        """
        for universe in range(universes):
            self._universe_data[universe] = bytearray(DMX_CHANNELS_PER_UNIVERSE)
            packet = self._build_artdmx_packet(
                universe, self._universe_data[universe]
            )
            self.socket.sendto(packet, (self.target_ip, self.port))
            self.sequence = (self.sequence + 1) % 256

    def _build_artdmx_packet(self, universe: int, dmx_data: bytearray) -> bytes:
        """Build an ArtDmx packet."""
        packet = bytearray()

        # Header: 'Art-Net\0'
        packet.extend(ARTNET_HEADER)

        # OpCode (little-endian)
        packet.extend(struct.pack("<H", ARTNET_OPCODE_DMX))

        # Protocol version (big-endian)
        packet.extend(struct.pack(">H", ARTNET_PROTOCOL_VERSION))

        # Sequence number
        packet.append(self.sequence)

        # Physical port (0)
        packet.append(0)

        # Universe (little-endian)
        packet.extend(struct.pack("<H", universe))

        # Length (big-endian) - must be even, 2-512
        length = len(dmx_data)
        if length % 2 == 1:
            length += 1
        packet.extend(struct.pack(">H", length))

        # DMX data
        packet.extend(dmx_data)

        return bytes(packet)

    def close(self) -> None:
        """Close the socket."""
        if not self._closed:
            self.socket.close()
            self._closed = True

    def __enter__(self) -> "ArtNetOutput":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# =============================================================================
# Show Player
# =============================================================================

class ShowPlayer:
    """
    Plays a show in real-time, sending Art-Net at each cue.

    This is a simple player for testing - real-time audio sync
    would require a more sophisticated approach.
    """

    def __init__(
        self,
        show: Show,
        rig: Rig,
        artnet: Optional[ArtNetOutput] = None,
        target_ip: str = "127.0.0.1",
    ):
        """
        Initialize show player.

        Args:
            show: Show to play
            rig: Fixture rig
            artnet: Optional ArtNetOutput instance (created if not provided)
            target_ip: Art-Net target IP (if creating new output)
        """
        self.show = show
        self.rig = rig
        self.artnet = artnet or ArtNetOutput(target_ip=target_ip)
        self._owns_artnet = artnet is None

    def play(
        self,
        start_time: float = 0.0,
        speed: float = 1.0,
        callback=None,
    ) -> None:
        """
        Play the show from start_time.

        This blocks until the show is complete.

        Args:
            start_time: Start position in seconds
            speed: Playback speed multiplier (1.0 = real-time)
            callback: Optional callback(time, cue) called at each cue change
        """
        cues = self.show.cues
        if not cues:
            return

        # Find starting cue
        current_cue_idx = 0
        for i, cue in enumerate(cues):
            if cue.timestamp_sec <= start_time:
                current_cue_idx = i

        # Send initial scene
        current_cue = cues[current_cue_idx]
        self.artnet.send_frame(self.rig, current_cue.scene)
        if callback:
            callback(start_time, current_cue)

        # Playback loop
        play_start = time.time()
        song_time = start_time

        try:
            while song_time < self.show.duration_sec:
                # Update song time
                elapsed = (time.time() - play_start) * speed
                song_time = start_time + elapsed

                # Check for cue change
                next_cue_idx = current_cue_idx + 1
                if next_cue_idx < len(cues):
                    next_cue = cues[next_cue_idx]
                    if song_time >= next_cue.timestamp_sec:
                        # Trigger new cue
                        current_cue_idx = next_cue_idx
                        current_cue = next_cue
                        self.artnet.send_frame(self.rig, current_cue.scene)
                        if callback:
                            callback(song_time, current_cue)

                # Maintain frame rate
                time.sleep(DMX_FRAME_TIME)

        except KeyboardInterrupt:
            print("\nPlayback interrupted")
        finally:
            # Blackout on exit
            self.artnet.blackout(self.rig.universes)

    def preview_cue(self, cue_index: int) -> None:
        """Send a single cue for preview."""
        if 0 <= cue_index < len(self.show.cues):
            cue = self.show.cues[cue_index]
            self.artnet.send_frame(self.rig, cue.scene)

    def preview_scene(self, scene: Scene) -> None:
        """Send a scene for preview."""
        self.artnet.send_frame(self.rig, scene)

    def close(self) -> None:
        """Clean up resources."""
        if self._owns_artnet:
            self.artnet.close()


# =============================================================================
# Utility Functions
# =============================================================================

def print_artnet_packet(packet: bytes) -> None:
    """Print Art-Net packet contents for debugging."""
    print(f"Packet length: {len(packet)} bytes")
    print(f"Header: {packet[:8]}")
    print(f"OpCode: 0x{struct.unpack('<H', packet[8:10])[0]:04x}")
    print(f"Version: {struct.unpack('>H', packet[10:12])[0]}")
    print(f"Sequence: {packet[12]}")
    print(f"Physical: {packet[13]}")
    print(f"Universe: {struct.unpack('<H', packet[14:16])[0]}")
    print(f"Length: {struct.unpack('>H', packet[16:18])[0]}")
    print(f"Data (first 32): {list(packet[18:50])}")


def test_artnet_connection(target_ip: str = "127.0.0.1", universes: int = 1) -> bool:
    """
    Test Art-Net connection by sending a brief flash.

    Returns True if no socket errors occurred.
    """
    try:
        with ArtNetOutput(target_ip=target_ip) as artnet:
            # Send white flash
            for universe in range(universes):
                artnet.send_dmx(universe, {i: 255 for i in range(1, 17)})
            time.sleep(0.2)

            # Blackout
            artnet.blackout(universes)

        return True
    except Exception as e:
        print(f"Art-Net test failed: {e}")
        return False


if __name__ == "__main__":
    from src.dmx.config import load_rig
    from src.dmx.generator import generate_show

    print("Art-Net Output Test")
    print("=" * 60)

    # Load rig
    rig = load_rig("user_rig")
    print(f"Loaded rig: {rig.name}")

    # Generate a test show
    boundaries = [0.0, 5.0, 10.0, 15.0, 20.0]
    show = generate_show(
        boundaries=boundaries,
        duration=25.0,
        rig=rig,
        song_path="test.wav",
        intensity=0.8,
    )
    print(f"Generated show with {len(show.cues)} cues")

    # Test packet building
    print("\n" + "=" * 60)
    print("Art-Net Packet Structure:")
    print("=" * 60)

    artnet = ArtNetOutput()
    packet = artnet._build_artdmx_packet(0, bytearray([255, 128, 64] + [0] * 509))
    print_artnet_packet(packet)

    # Preview each cue
    print("\n" + "=" * 60)
    print("Sending cue previews to localhost:6454")
    print("(Open QLC+ with Art-Net input to see the output)")
    print("=" * 60)

    try:
        for i, cue in enumerate(show.cues):
            print(f"\nCue {i + 1}: {cue.section_type} @ {cue.timestamp_sec:.1f}s")

            # Show what we're sending
            scene = cue.scene
            for fixture in rig.fixtures[:2]:  # Just show first 2 fixtures
                state = scene.get_state(fixture.id)
                if state and state.color:
                    print(f"  {fixture.id}: {state.color.to_hex()} @ {state.dimmer:.0%}")

            # Send to Art-Net
            artnet.send_frame(rig, scene)
            time.sleep(1.0)

        # Blackout
        print("\nBlackout...")
        artnet.blackout()

    finally:
        artnet.close()

    print("\nDone!")
