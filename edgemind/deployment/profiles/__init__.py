"""Hardware deployment profile registry."""

from edgemind.deployment.profiles.base_profile import HardwareProfile
from edgemind.deployment.profiles.jetson_nano import JETSON_NANO_PROFILE
from edgemind.deployment.profiles.jetson_orin import JETSON_AGX_ORIN_PROFILE
from edgemind.deployment.profiles.mac_m1 import MAC_M1_PROFILE
from edgemind.deployment.profiles.mac_m2 import MAC_M2_PROFILE
from edgemind.deployment.profiles.mac_m3_pro import MAC_M3_PRO_PROFILE
from edgemind.deployment.profiles.raspberry_pi5 import RASPBERRY_PI5_PROFILE
from edgemind.deployment.profiles.rtx_4090 import RTX_4090_PROFILE
from edgemind.deployment.profiles.rtx_5090 import RTX_5090_PROFILE

PROFILES: dict[str, HardwareProfile] = {
    "rtx_5090": RTX_5090_PROFILE,
    "rtx_4090": RTX_4090_PROFILE,
    "jetson_orin": JETSON_AGX_ORIN_PROFILE,
    "jetson_nano": JETSON_NANO_PROFILE,
    "raspberry_pi5": RASPBERRY_PI5_PROFILE,
    "mac_m1": MAC_M1_PROFILE,
    "mac_m2": MAC_M2_PROFILE,
    "mac_m3_pro": MAC_M3_PRO_PROFILE,
}

PROFILE_DISPLAY_NAMES: dict[str, str] = {
    "rtx_5090": "NVIDIA RTX 5090 (24GB)",
    "rtx_4090": "NVIDIA RTX 4090 (24GB)",
    "jetson_orin": "NVIDIA Jetson AGX Orin (32GB)",
    "jetson_nano": "NVIDIA Jetson Nano (4GB)",
    "raspberry_pi5": "Raspberry Pi 5 (8GB)",
    "mac_m1": "Apple M1 (8-16GB)",
    "mac_m2": "Apple M2 (8-24GB)",
    "mac_m3_pro": "Apple M3 Pro (18-36GB)",
}


def get_profile(name: str) -> HardwareProfile:
    """Retrieve a hardware profile by key name.

    Args:
        name: Profile key (e.g. "rtx_5090", "mac_m1").

    Returns:
        The matching HardwareProfile instance.

    Raises:
        KeyError: If the profile name is not recognized.
    """
    if name not in PROFILES:
        raise KeyError(
            f"Unknown hardware profile: {name!r}. "
            f"Available: {list(PROFILES.keys())}"
        )
    return PROFILES[name]
