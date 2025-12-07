# TermPlay

Display images, GIFs, and videos directly in your terminal using SIXEL graphics!

## Demo:
https://github.com/user-attachments/assets/99be5001-88e2-4c0b-940d-5df380b9db34


## Features

- **Image Display** - View PNG, JPG, BMP, WebP, TIFF, and more
- **Animated GIFs** - Play animated GIFs with proper timing
- **Video Playback** - Play MP4, AVI, MKV, MOV, and other video formats
- **Audio Support** - Video audio playback via ffplay (requires FFmpeg)
- **Save to SIXEL** - Export images as `.sixel` files
- **Optimized Performance** - Fast SIXEL encoding for smooth video playback
- **Loop Support** - Loop videos and GIFs continuously

## Requirements

- Python 3.8+
- A SIXEL-compatible terminal:
  - **Windows Terminal** (Windows 10/11 - native support)
  - **WezTerm** (Cross-platform)
  - **mlterm** (Linux/macOS)
  - **mintty** (Windows - Git Bash, Cygwin)
  - **xterm** (with sixel support compiled)
  - **Contour** (Cross-platform)

## Installation

### From PyPI

```bash
pip install termplay
```

This installs everything needed for images and videos, including a bundled FFmpeg.

### With OpenCV fallback (optional)

```bash
pip install termplay[opencv]
```

### From source

```bash
git clone https://github.com/dev1abhi/termPlay.git
cd termPlay
pip install -e .
```

## Usage

### Command Line

```bash
# Display an image
termplay image.png

# Display with custom size
termplay photo.jpg --width 400 --height 300

# Play an animated GIF (loops by default)
termplay animation.gif

# Play a video
termplay video.mp4

# Play video with custom FPS
termplay video.mp4 --fps 15

# Loop a video
termplay video.mp4 --loop

# Save image as .sixel file
termplay image.png --save
termplay image.png --save -o output.sixel

# Reduce colors for faster display
termplay image.png --colors 64
```

### Command Line Options

| Option       | Short | Description                                   |
| ------------ | ----- | --------------------------------------------- |
| `--width`    | `-w`  | Maximum width in pixels (default: 800)        |
| `--height`   | `-H`  | Maximum height in pixels (default: 600)       |
| `--colors`   | `-c`  | Number of colors 2-256 (default: 256)         |
| `--fps`      | `-f`  | Frames per second for video (default: native) |
| `--loop`     | `-l`  | Loop video/GIF playback                       |
| `--save`     | `-s`  | Save as .sixel file instead of displaying     |
| `--output`   | `-o`  | Output filename for --save                    |
| `--no-check` |       | Skip terminal SIXEL support check             |
| `--version`  | `-v`  | Show version                                  |

### Python API

```python
from termplay import display_image, display_video, display_gif, image_to_sixel
from PIL import Image

# Display an image
display_image("photo.jpg", max_width=800, max_height=600, colors=256)

# Display a video
display_video("video.mp4", max_width=320, max_height=240, fps=30, loop=True)

# Display an animated GIF
display_gif("animation.gif", max_width=400, max_height=300, colors=64, loop=True)

# Convert PIL Image to SIXEL string
img = Image.open("photo.jpg")
sixel_data = image_to_sixel(img, max_width=800, max_height=600, colors=256)
print(sixel_data)
```

## Video Playback

Video support is included by default with the `av` package. The `imageio-ffmpeg` package is also bundled which provides FFmpeg binaries automatically.

For audio playback, the bundled FFmpeg is used when available. If you want system-wide FFmpeg (for `ffplay`):

- **Windows**: `winget install FFmpeg` or download from https://ffmpeg.org/download.html
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg` or equivalent

### Optional: OpenCV fallback

```bash
# Alternative video decoder (if av doesn't work)
pip install termplay[opencv]
```

## Supported Formats

### Images

PNG, JPG, JPEG, BMP, WebP, TIFF, TIF, ICO

### Videos (requires av or opencv-python)

MP4, AVI, MKV, MOV, WMV, FLV, WebM, M4V

### Animations

GIF (animated and static)

## Tips

1. **Reduce colors for faster video**: Use `--colors 32` for smoother video playback
2. **Smaller dimensions = faster**: Video defaults to 320x240 for performance
3. **Loop GIFs**: GIFs loop by default, use `--loop` for videos
4. **Save for later**: Use `--save` to create `.sixel` files you can display with `type filename.sixel` (Windows) or `cat filename.sixel` (Linux/Mac)

## Troubleshooting

### "Your terminal may not support SIXEL graphics"

Make sure you're using a SIXEL-compatible terminal. Windows Terminal has native support since version 1.22.

### Video playback is slow

- Reduce dimensions: `--width 240 --height 180`
- Reduce colors: `--colors 16`
- Reduce FPS: `--fps 10`

### No audio in videos

- Install FFmpeg and ensure `ffplay` is in your PATH
- Use `pip install av` for best video/audio handling

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- SIXEL graphics protocol
- Pillow for image processing
- PyAV for video decoding
- All the SIXEL-compatible terminal developers
