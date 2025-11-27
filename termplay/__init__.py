"""
TermPlay - SIXEL Image/Video Viewer for Terminal
Displays images, GIFs, and videos using SIXEL graphics protocol.
"""

__version__ = "1.0.0"
__author__ = "dev1abhi"

from .core import (
    image_to_sixel,
    image_to_sixel_fast,
    display_image,
    display_video,
    display_gif,
    save_as_sixel,
    check_sixel_support,
    get_file_type,
)

__all__ = [
    "image_to_sixel",
    "image_to_sixel_fast", 
    "display_image",
    "display_video",
    "display_gif",
    "save_as_sixel",
    "check_sixel_support",
    "get_file_type",
]
