#!/usr/bin/env python3
"""
TermPlay - SIXEL Image/Video Viewer for Terminal
Core functionality for displaying images, GIFs, and videos using SIXEL graphics protocol.
"""

import sys
import os
import time
import threading
import tempfile
import subprocess
from pathlib import Path

# Set OpenCV environment variables BEFORE importing cv2
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '100000'

# Suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from PIL import Image


class AudioPlayer:
    """Audio player with pause/resume support using pygame."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.temp_audio = None
        self.is_initialized = False
        self.is_paused = False
        self.is_muted = False
        self._volume = 1.0
        self._position = 0  # Track position in seconds
        self._pause_time = 0
        
    def extract_audio(self):
        """Extract audio from video to a temp file."""
        try:
            # Create temp file for audio
            self.temp_audio = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            self.temp_audio.close()
            
            # Use ffmpeg to extract audio
            ffmpeg_path = self._get_ffmpeg_path()
            if not ffmpeg_path:
                return False
            
            result = subprocess.run(
                [ffmpeg_path, '-i', self.video_path, '-vn', '-acodec', 'libmp3lame', 
                 '-q:a', '2', '-y', self.temp_audio.name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            if result.returncode != 0:
                return False
            
            return True
        except Exception:
            return False
    
    def _get_ffmpeg_path(self):
        """Get ffmpeg path."""
        import shutil
        if shutil.which('ffmpeg'):
            return 'ffmpeg'
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return None
    
    def initialize(self):
        """Initialize pygame mixer and load audio."""
        if not self.extract_audio():
            return False
        
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            pygame.mixer.music.load(self.temp_audio.name)
            self.is_initialized = True
            return True
        except Exception:
            return False
    
    def play(self):
        """Start playing audio."""
        if not self.is_initialized:
            return
        try:
            import pygame
            pygame.mixer.music.play()
        except:
            pass
    
    def pause(self):
        """Pause audio playback."""
        if not self.is_initialized:
            return
        try:
            import pygame
            pygame.mixer.music.pause()
            self.is_paused = True
        except:
            pass
    
    def unpause(self):
        """Resume audio playback."""
        if not self.is_initialized:
            return
        try:
            import pygame
            pygame.mixer.music.unpause()
            self.is_paused = False
        except:
            pass
    
    def toggle_pause(self):
        """Toggle pause state."""
        if self.is_paused:
            self.unpause()
        else:
            self.pause()
        return self.is_paused
    
    def mute(self):
        """Mute audio."""
        if not self.is_initialized:
            return
        try:
            import pygame
            pygame.mixer.music.set_volume(0)
            self.is_muted = True
        except:
            pass
    
    def unmute(self):
        """Unmute audio."""
        if not self.is_initialized:
            return
        try:
            import pygame
            pygame.mixer.music.set_volume(self._volume)
            self.is_muted = False
        except:
            pass
    
    def toggle_mute(self):
        """Toggle mute state."""
        if self.is_muted:
            self.unmute()
        else:
            self.mute()
        return self.is_muted
    
    def stop(self):
        """Stop audio and cleanup."""
        if self.is_initialized:
            try:
                import pygame
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except:
                pass
        
        # Clean up temp file
        if self.temp_audio:
            try:
                os.unlink(self.temp_audio.name)
            except:
                pass
    
    def is_available(self):
        """Check if pygame is available."""
        try:
            import pygame
            return True
        except ImportError:
            return False


# Keyboard input handling for Windows/Unix
if sys.platform == 'win32':
    import msvcrt
    
    def get_key_non_blocking():
        """Get a keypress without blocking (Windows)."""
        if msvcrt.kbhit():
            key = msvcrt.getch()
            # Handle special keys (arrows, etc.)
            if key in (b'\x00', b'\xe0'):
                key = msvcrt.getch()
                if key == b'K':  # Left arrow
                    return 'left'
                elif key == b'M':  # Right arrow
                    return 'right'
                elif key == b'H':  # Up arrow
                    return 'up'
                elif key == b'P':  # Down arrow
                    return 'down'
                return None
            try:
                char = key.decode('utf-8').lower()
                if char == ' ':
                    return 'space'
                elif char == '\x1b':  # Escape
                    return 'esc'
                return char
            except:
                return None
        return None
else:
    import select
    import tty
    import termios
    
    def get_key_non_blocking():
        """Get a keypress without blocking (Unix)."""
        if select.select([sys.stdin], [], [], 0)[0]:
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setraw(sys.stdin.fileno())
                key = sys.stdin.read(1)
                if key == '\x1b':  # Escape sequence
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key += sys.stdin.read(2)
                        if key == '\x1b[D':
                            return 'left'
                        elif key == '\x1b[C':
                            return 'right'
                        elif key == '\x1b[A':
                            return 'up'
                        elif key == '\x1b[B':
                            return 'down'
                    return 'esc'
                elif key == ' ':
                    return 'space'
                return key.lower()
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        return None


def image_to_sixel_fast(image: Image.Image, max_width: int = 800, max_height: int = 600, colors: int = 64) -> str:
    """Ultra-fast SIXEL conversion optimized for video playback."""
    
    # Convert to RGB if necessary (do this before resize for speed)
    if image.mode != 'RGB':
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (0, 0, 0))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert('RGB')
    
    # Quantize to limited color palette
    image = image.quantize(colors=min(colors, 256), method=Image.Quantize.FASTOCTREE)
    
    width, height = image.size
    palette = image.getpalette()
    
    # Get pixel data as bytes for faster access
    pixel_data = image.tobytes()
    
    # Build SIXEL output using list for faster concatenation
    output = ["\033Pq", f'"1;1;{width};{height}']
    
    # Define color palette
    num_palette_colors = len(palette) // 3 if palette else 0
    actual_colors = min(colors, 256, num_palette_colors)
    
    palette_strs = []
    for i in range(actual_colors):
        r = palette[i * 3] * 100 // 255
        g = palette[i * 3 + 1] * 100 // 255
        b = palette[i * 3 + 2] * 100 // 255
        palette_strs.append(f"#{i};2;{r};{g};{b}")
    output.extend(palette_strs)
    
    # Process 6 rows at a time
    for y_base in range(0, height, 6):
        color_data = {}
        
        # Process all columns for this band
        for x in range(width):
            # Get the 6 pixels in this column using direct byte access
            col_colors = {}
            for dy in range(6):
                y = y_base + dy
                if y < height:
                    pixel = pixel_data[y * width + x]
                    if pixel not in col_colors:
                        col_colors[pixel] = 0
                    col_colors[pixel] |= (1 << dy)
            
            # Add to color data
            for color, bits in col_colors.items():
                if color not in color_data:
                    color_data[color] = {}
                color_data[color][x] = chr(63 + bits)
        
        # Output each color's data with simple RLE
        for color in sorted(color_data.keys()):
            positions = color_data[color]
            output.append(f"#{color}")
            
            # Build line efficiently
            chars = []
            last_x = -1
            for x in range(width):
                if x in positions:
                    # Fill gap with '?'
                    if last_x < x - 1:
                        gap = x - last_x - 1
                        if gap >= 3:
                            chars.append(f"!{gap}?")
                        else:
                            chars.append('?' * gap)
                    chars.append(positions[x])
                    last_x = x
            
            # Handle trailing gap
            if last_x < width - 1 and last_x >= 0:
                gap = width - last_x - 1
                if gap >= 3:
                    chars.append(f"!{gap}?")
                else:
                    chars.append('?' * gap)
            
            output.append(''.join(chars))
            output.append("$")
        
        output.append("-")
    
    output.append("\033\\")
    return ''.join(output)


def image_to_sixel(image: Image.Image, max_width: int = 800, max_height: int = 600, colors: int = 256) -> str:
    """Convert a PIL Image to SIXEL format string."""
    
    # For video (fewer colors), use fast path
    if colors <= 64:
        return image_to_sixel_fast(image, max_width, max_height, colors)
    
    # Resize image to fit terminal while maintaining aspect ratio
    image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    
    # Convert to RGB if necessary
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (0, 0, 0))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Quantize to limited color palette
    image = image.quantize(colors=min(colors, 256))
    
    width, height = image.size
    palette = image.getpalette()
    pixels = list(image.getdata())
    
    # Build SIXEL output
    sixel_output = []
    sixel_output.append("\033Pq")
    sixel_output.append(f'"1;1;{width};{height}')
    
    # Define color palette
    num_palette_colors = len(palette) // 3 if palette else 0
    actual_colors = min(colors, 256, num_palette_colors)
    
    for i in range(actual_colors):
        if palette:
            r = palette[i * 3] * 100 // 255
            g = palette[i * 3 + 1] * 100 // 255
            b = palette[i * 3 + 2] * 100 // 255
            sixel_output.append(f"#{i};2;{r};{g};{b}")
    
    # Convert pixels to SIXEL data
    for y_base in range(0, height, 6):
        line_data = {}
        
        for x in range(width):
            for color in range(actual_colors):
                sixel_char = 0
                for bit in range(6):
                    y = y_base + bit
                    if y < height:
                        pixel_index = y * width + x
                        if pixel_index < len(pixels) and pixels[pixel_index] == color:
                            sixel_char |= (1 << bit)
                
                if sixel_char > 0:
                    if color not in line_data:
                        line_data[color] = ['?'] * width
                    line_data[color][x] = chr(63 + sixel_char)
        
        for color, chars in line_data.items():
            sixel_output.append(f"#{color}")
            
            result = []
            i = 0
            while i < len(chars):
                char = chars[i]
                count = 1
                while i + count < len(chars) and chars[i + count] == char:
                    count += 1
                
                if count >= 4:
                    result.append(f"!{count}{char}")
                else:
                    result.append(char * count)
                i += count
            
            sixel_output.append(''.join(result))
            sixel_output.append("$")
        
        sixel_output.append("-")
    
    sixel_output.append("\033\\")
    return ''.join(sixel_output)


def display_image(image_path: str, max_width: int = 800, max_height: int = 600, colors: int = 256):
    """Display an image file in the terminal using SIXEL."""
    try:
        with Image.open(image_path) as img:
            print(f"Displaying: {image_path}")
            print(f"Settings: {max_width}x{max_height}, {colors} colors")
            sixel_data = image_to_sixel(img, max_width, max_height, colors)
            sys.stdout.write(sixel_data)
            sys.stdout.flush()
            print()
    except FileNotFoundError:
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)


def display_video(video_path: str, max_width: int = 320, max_height: int = 240, 
                  fps: float = None, colors: int = 32, loop: bool = False):
    """Display a video file in the terminal using SIXEL (frame by frame)."""
    
    # Try PyAV first (better handling of videos with audio)
    try:
        import av
        use_pyav = True
    except ImportError:
        use_pyav = False
    
    if not use_pyav:
        try:
            import cv2
            use_cv2 = True
        except ImportError:
            use_cv2 = False
    else:
        use_cv2 = False
    
    if not use_pyav and not use_cv2:
        print("Error: Video playback requires either:")
        print("  pip install av   (recommended)")
        print("  pip install opencv-python")
        sys.exit(1)
    
    if use_pyav:
        _play_video_pyav(video_path, max_width, max_height, fps, colors, loop)
    else:
        _play_video_cv2(video_path, max_width, max_height, fps, colors, loop)


def _get_ffplay_path():
    """Get the path to ffplay, using imageio-ffmpeg's bundled ffmpeg if available."""
    import shutil
    
    # First try system ffplay
    if shutil.which('ffplay'):
        return 'ffplay'
    
    # Try to get ffmpeg from imageio-ffmpeg and derive ffplay path
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        # ffplay is usually in the same directory as ffmpeg
        import os
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        ffplay_candidates = [
            os.path.join(ffmpeg_dir, 'ffplay'),
            os.path.join(ffmpeg_dir, 'ffplay.exe'),
        ]
        for candidate in ffplay_candidates:
            if os.path.exists(candidate):
                return candidate
    except ImportError:
        pass
    
    return None


def _play_video_pyav(video_path: str, max_width: int, max_height: int,
                     fps: float, colors: int, loop: bool):
    """Play video using PyAV with audio support and A/V sync."""
    import av
    
    container = av.open(video_path)
    video_stream = None
    has_audio = False
    
    for stream in container.streams:
        if stream.type == 'video' and video_stream is None:
            video_stream = stream
        if stream.type == 'audio':
            has_audio = True
    
    if video_stream is None:
        print("Error: No video stream found in file")
        container.close()
        sys.exit(1)
    
    video_fps = float(video_stream.average_rate) if video_stream.average_rate else 30
    if video_stream.duration and video_stream.time_base:
        duration = float(video_stream.duration * video_stream.time_base)
    else:
        duration = 0
    
    effective_fps = fps if fps is not None else video_fps
    
    container.close()
    
    # Initialize audio player
    audio_player = None
    audio_available = False
    
    if has_audio:
        audio_player = AudioPlayer(video_path)
        if audio_player.is_available():
            #print("Extracting audio...")
            if audio_player.initialize():
                audio_available = True
            else:
                print("Note: Could not initialize audio.")
        else:
            print("Note: Install pygame for audio support: pip install pygame")
    
    # Store info for display
    info_lines = [
        f"Playing: {video_path}",
        f"Video FPS: {video_fps:.2f}, Duration: {duration:.1f}s",
        f"Settings: {max_width}x{max_height}, {colors} colors, {effective_fps:.1f} fps{', loop' if loop else ''}",
        f"Audio: {'Yes' if audio_available else 'No'}",
        "",
        "Controls: [Space] Pause  [Q/Esc] Quit  [M] Mute",
        ""
    ]
    
    frame_count = 0
    is_paused = False
    is_muted = False
    
    container = av.open(video_path)
    stream = container.streams.video[0]
    orig_width = stream.width
    orig_height = stream.height
    
    scale = min(max_width / orig_width, max_height / orig_height, 1.0)
    target_width = int(orig_width * scale)
    target_height = int(orig_height * scale)
    
    frame_skip = max(1, int(video_fps / effective_fps))
    
    container.close()
    
    # Print info before switching to alternate screen
    for line in info_lines:
        if line:
            print(line)
    
    # Switch to alternate screen buffer (like vim/less)
    sys.stdout.write("\033[?1049h")  # Enter alternate screen
    sys.stdout.write("\033[?25l")    # Hide cursor
    sys.stdout.write("\033[H")       # Move cursor to home
    sys.stdout.flush()
    
    # Start audio playback
    if audio_available:
        audio_player.play()
    
    playback_start = time.perf_counter()
    pause_start = 0
    
    quit_requested = False
    
    try:
        while not quit_requested:
            container = av.open(video_path)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            
            frame_index = 0
            
            for frame in container.decode(stream):
                # Check for keyboard input
                key = get_key_non_blocking()
                if key:
                    if key in ('q', 'esc'):
                        quit_requested = True
                        break
                    elif key == 'space':
                        is_paused = not is_paused
                        if audio_available:
                            audio_player.toggle_pause()
                        if is_paused:
                            pause_start = time.perf_counter()
                    elif key == 'm' and audio_available:
                        audio_player.toggle_mute()
                        is_muted = audio_player.is_muted
                
                # Handle pause
                while is_paused and not quit_requested:
                    key = get_key_non_blocking()
                    if key == 'space':
                        is_paused = False
                        if audio_available:
                            audio_player.toggle_pause()
                        # Adjust playback timing to account for pause duration
                        pause_duration = time.perf_counter() - pause_start
                        playback_start += pause_duration
                    elif key in ('q', 'esc'):
                        quit_requested = True
                        break
                    elif key == 'm' and audio_available:
                        audio_player.toggle_mute()
                        is_muted = audio_player.is_muted
                    time.sleep(0.05)
                
                if quit_requested:
                    break
                
                frame_index += 1
                
                if frame_index % frame_skip != 0:
                    continue
                
                expected_time = frame_count / effective_fps
                actual_time = time.perf_counter() - playback_start
                
                if actual_time > expected_time + 0.1:
                    frame_count += 1
                    continue
                
                # Position cursor at home for each frame (SIXEL overwrites in place)
                sys.stdout.write("\033[H")
                
                pil_image = frame.to_image()
                if pil_image.size != (target_width, target_height):
                    pil_image = pil_image.resize((target_width, target_height), Image.Resampling.NEAREST)
                
                sixel_data = image_to_sixel_fast(pil_image, max_width, max_height, colors)
                
                sys.stdout.write(sixel_data)
                sys.stdout.flush()
                
                frame_count += 1
                
                expected_next = frame_count / effective_fps
                actual_now = time.perf_counter() - playback_start
                sleep_time = expected_next - actual_now
                
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
            
            container.close()
            
            if not loop or quit_requested:
                break
            
            # Restart audio for looping
            if audio_available:
                audio_player.stop()
                audio_player.initialize()
                audio_player.play()
                if is_muted:
                    audio_player.mute()
            
            playback_start = time.perf_counter()
            frame_count = 0
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        pass
    finally:
        sys.stdout.write("\033[?25h")    # Show cursor
        sys.stdout.write("\033[?1049l")  # Exit alternate screen (restore main screen)
        sys.stdout.flush()
        if audio_player:
            audio_player.stop()
        try:
            container.close()
        except:
            pass
        print(f"Playback finished. ({frame_count} frames)")


def _play_video_cv2(video_path: str, max_width: int, max_height: int,
                    fps: float, colors: int, loop: bool):
    """Play video using OpenCV (fallback)."""
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    effective_fps = fps if fps is not None else video_fps
    if effective_fps <= 0:
        effective_fps = 30
    
    # Store info for display
    info_lines = [
        f"Playing: {video_path}",
        f"Video FPS: {video_fps:.2f}, Total frames: {total_frames}",
        f"Settings: {max_width}x{max_height}, {colors} colors, {effective_fps:.1f} fps{', loop' if loop else ''}",
        "",
        "Controls: [Space] Pause  [Q/Esc] Quit",
        ""
    ]
    
    # Print info before switching to alternate screen
    for line in info_lines:
        if line:
            print(line)
    
    frame_delay = 1.0 / effective_fps
    is_paused = False
    quit_requested = False
    
    # Switch to alternate screen buffer (like vim/less)
    sys.stdout.write("\033[?1049h")  # Enter alternate screen
    sys.stdout.write("\033[?25l")    # Hide cursor
    sys.stdout.write("\033[H")       # Move cursor to home
    sys.stdout.flush()
    
    try:
        while not quit_requested:
            ret, frame = cap.read()
            
            if not ret:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            # Check for keyboard input
            key = get_key_non_blocking()
            if key:
                if key in ('q', 'esc'):
                    quit_requested = True
                    break
                elif key == 'space':
                    is_paused = not is_paused
            
            # Handle pause
            while is_paused and not quit_requested:
                key = get_key_non_blocking()
                if key == 'space':
                    is_paused = False
                elif key in ('q', 'esc'):
                    quit_requested = True
                    break
                time.sleep(0.05)
            
            if quit_requested:
                break
            
            # Position cursor at home for each frame (SIXEL overwrites in place)
            sys.stdout.write("\033[H")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            sixel_data = image_to_sixel(pil_image, max_width, max_height, colors)
            
            sys.stdout.write(sixel_data)
            sys.stdout.flush()
            
            time.sleep(frame_delay)
            
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h")    # Show cursor
        sys.stdout.write("\033[?1049l")  # Exit alternate screen (restore main screen)
        sys.stdout.flush()
        cap.release()
        print("Playback stopped.")


def display_gif(gif_path: str, max_width: int = 400, max_height: int = 300, 
                colors: int = 64, loop: bool = True):
    """Display an animated GIF in the terminal using SIXEL."""
    try:
        img = Image.open(gif_path)
    except FileNotFoundError:
        print(f"Error: File not found: {gif_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading GIF: {e}")
        sys.exit(1)
    
    if not getattr(img, 'is_animated', False):
        display_image(gif_path, max_width, max_height, colors)
        return
    
    # Store info for display
    info_lines = [
        f"Playing animated GIF: {gif_path}",
        f"Frames: {img.n_frames}",
        f"Settings: {max_width}x{max_height}, {colors} colors{', loop' if loop else ''}",
        "",
        "Controls: [Space] Pause  [Q/Esc] Quit",
        ""
    ]
    
    print("Pre-processing frames...")
    
    frames = []
    durations = []
    
    for frame_num in range(img.n_frames):
        img.seek(frame_num)
        duration = img.info.get('duration', 100) / 1000.0
        durations.append(duration)
        
        frame = img.convert('RGB')
        frame.thumbnail((max_width, max_height), Image.Resampling.NEAREST)
        
        sixel_data = image_to_sixel_fast(frame, max_width, max_height, colors)
        frames.append(sixel_data)
    
    print(f"Ready!")
    
    # Print info before switching to alternate screen
    for line in info_lines:
        if line:
            print(line)
    
    # Playback state
    is_paused = False
    quit_requested = False
    
    # Switch to alternate screen buffer (like vim/less)
    sys.stdout.write("\033[?1049h")  # Enter alternate screen
    sys.stdout.write("\033[?25l")    # Hide cursor
    sys.stdout.write("\033[H")       # Move cursor to home
    sys.stdout.flush()
    
    try:
        while not quit_requested:
            for i, sixel_data in enumerate(frames):
                # Check for keyboard input
                key = get_key_non_blocking()
                if key:
                    if key in ('q', 'esc'):
                        quit_requested = True
                        break
                    elif key == 'space':
                        is_paused = not is_paused
                
                # Handle pause
                while is_paused and not quit_requested:
                    key = get_key_non_blocking()
                    if key == 'space':
                        is_paused = False
                    elif key in ('q', 'esc'):
                        quit_requested = True
                        break
                    time.sleep(0.05)
                
                if quit_requested:
                    break
                
                start_time = time.perf_counter()
                
                # Position cursor at home for each frame (SIXEL overwrites in place)
                sys.stdout.write("\033[H")
                sys.stdout.write(sixel_data)
                sys.stdout.flush()
                
                elapsed = time.perf_counter() - start_time
                sleep_time = durations[i] - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            if not loop or quit_requested:
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h")    # Show cursor
        sys.stdout.write("\033[?1049l")  # Exit alternate screen (restore main screen)
        sys.stdout.flush()
        print("Playback stopped.")


def save_as_sixel(image_path: str, output_path: str = None, max_width: int = 800, 
                   max_height: int = 600, colors: int = 256):
    """Convert an image to a .sixel file that can be viewed with 'type' command."""
    try:
        with Image.open(image_path) as img:
            sixel_data = image_to_sixel(img, max_width, max_height, colors)
            
            if output_path is None:
                output_path = Path(image_path).stem + '.sixel'
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(sixel_data)
            
            print(f"Saved: {output_path}")
            print(f"View with: type \"{output_path}\"")
            
    except FileNotFoundError:
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def check_sixel_support():
    """Check if terminal supports SIXEL (basic check)."""
    term = os.environ.get('TERM', '')
    term_program = os.environ.get('TERM_PROGRAM', '')
    wt_session = os.environ.get('WT_SESSION', '')
    
    if wt_session:
        return True
    
    sixel_terms = ['mlterm', 'xterm-256color', 'mintty', 'wezterm', 'contour', 'windows-terminal']
    supported = any(t in term.lower() or t in term_program.lower() for t in sixel_terms)
    
    if not supported:
        print("Warning: Your terminal may not support SIXEL graphics.")
        print(f"TERM={term}, TERM_PROGRAM={term_program}")
        print("Supported terminals: Windows Terminal, mlterm, xterm (with sixel), mintty, WezTerm, Contour")
        print("Continuing anyway...\n")
    
    return supported


def get_file_type(filepath: str) -> str:
    """Determine file type based on extension."""
    ext = Path(filepath).suffix.lower()
    
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif', '.ico'}
    video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    gif_ext = {'.gif'}
    
    if ext in image_exts:
        return 'image'
    elif ext in video_exts:
        return 'video'
    elif ext in gif_ext:
        return 'gif'
    else:
        return 'unknown'
