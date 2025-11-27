#!/usr/bin/env python3
"""
TermPlay - SIXEL Image/Video Viewer for Terminal
Core functionality for displaying images, GIFs, and videos using SIXEL graphics protocol.
"""

import sys
import os
import time
from pathlib import Path

# Set OpenCV environment variables BEFORE importing cv2
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '100000'

from PIL import Image


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
    import subprocess
    
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
    
    print(f"Playing: {video_path}")
    print(f"Video FPS: {video_fps:.2f}, Duration: {duration:.1f}s")
    print(f"Settings: {max_width}x{max_height}, {colors} colors, {effective_fps:.1f} fps{', loop' if loop else ''}")
    print(f"Audio: {'Yes' if has_audio else 'No'}")
    print("Press Ctrl+C to stop")
    
    frame_count = 0
    audio_process = None
    
    container.close()
    
    if has_audio:
        ffplay_path = _get_ffplay_path()
        if ffplay_path:
            try:
                audio_process = subprocess.Popen(
                    [ffplay_path, '-nodisp', '-autoexit', '-loglevel', 'quiet', video_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception:
                print("Note: Could not start audio playback.")
                has_audio = False
        else:
            print("Note: ffplay not found. Audio disabled.")
            has_audio = False
    
    container = av.open(video_path)
    stream = container.streams.video[0]
    orig_width = stream.width
    orig_height = stream.height
    
    scale = min(max_width / orig_width, max_height / orig_height, 1.0)
    target_width = int(orig_width * scale)
    target_height = int(orig_height * scale)
    
    frame_skip = max(1, int(video_fps / effective_fps))
    
    container.close()
    
    frame_terminal_lines = (target_height // 16) + 2
    
    sys.stdout.write("\n" * frame_terminal_lines)
    sys.stdout.write(f"\033[{frame_terminal_lines}A")
    sys.stdout.write("\033[s")
    sys.stdout.flush()
    
    playback_start = time.perf_counter()
    
    try:
        while True:
            container = av.open(video_path)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            
            frame_index = 0
            
            for frame in container.decode(stream):
                frame_index += 1
                
                if frame_index % frame_skip != 0:
                    continue
                
                expected_time = frame_count / effective_fps
                actual_time = time.perf_counter() - playback_start
                
                if actual_time > expected_time + 0.1:
                    frame_count += 1
                    continue
                
                sys.stdout.write("\033[u\033[J")
                
                pil_image = frame.to_image()
                if pil_image.size != (target_width, target_height):
                    pil_image = pil_image.resize((target_width, target_height), Image.Resampling.NEAREST)
                
                sixel_data = image_to_sixel_fast(pil_image, max_width, max_height, colors)
                
                sys.stdout.write(sixel_data)
                sys.stdout.flush()
                
                frame_count += 1
                
                expected_next = (frame_count) / effective_fps
                actual_now = time.perf_counter() - playback_start
                sleep_time = expected_next - actual_now
                
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
            
            container.close()
            
            if not loop:
                break
            
            playback_start = time.perf_counter()
            frame_count = 0
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nError during playback: {e}")
    finally:
        if audio_process:
            audio_process.terminate()
            try:
                audio_process.wait(timeout=1)
            except:
                audio_process.kill()
        try:
            container.close()
        except:
            pass
        print(f"\nPlayback finished. ({frame_count} frames)")


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
    
    print(f"Playing: {video_path}")
    print(f"Video FPS: {video_fps:.2f}, Total frames: {total_frames}")
    print(f"Settings: {max_width}x{max_height}, {colors} colors, {effective_fps:.1f} fps{', loop' if loop else ''}")
    print("Press Ctrl+C to stop\n")
    
    frame_delay = 1.0 / effective_fps
    last_height_lines = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            if last_height_lines > 0:
                sys.stdout.write(f"\033[{last_height_lines}A")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            img_height = pil_image.size[1]
            last_height_lines = (img_height // 15) + 1
            
            sixel_data = image_to_sixel(pil_image, max_width, max_height, colors)
            
            sys.stdout.write(sixel_data)
            sys.stdout.write("\n")
            sys.stdout.flush()
            
            time.sleep(frame_delay)
            
    except KeyboardInterrupt:
        print("\n\nPlayback stopped.")
    finally:
        cap.release()


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
    
    print(f"Playing animated GIF: {gif_path}")
    print(f"Frames: {img.n_frames}")
    print(f"Settings: {max_width}x{max_height}, {colors} colors{', loop' if loop else ''}")
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
    
    print(f"Ready! Press Ctrl+C to stop")
    
    img.seek(0)
    thumb = img.convert('RGB')
    thumb.thumbnail((max_width, max_height), Image.Resampling.NEAREST)
    frame_terminal_lines = (thumb.size[1] // 16) + 2
    
    sys.stdout.write("\n" * frame_terminal_lines)
    sys.stdout.write(f"\033[{frame_terminal_lines}A")
    sys.stdout.write("\033[s")
    sys.stdout.flush()
    
    try:
        while True:
            for i, sixel_data in enumerate(frames):
                start_time = time.perf_counter()
                
                sys.stdout.write("\033[u\033[J")
                sys.stdout.write(sixel_data)
                sys.stdout.flush()
                
                elapsed = time.perf_counter() - start_time
                sleep_time = durations[i] - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            if not loop:
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        print("\nPlayback stopped.")


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
