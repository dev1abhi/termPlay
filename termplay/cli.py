#!/usr/bin/env python3
"""
TermPlay CLI - Command line interface for displaying images and videos in terminal.
"""

import argparse
import sys
import os

from .core import (
    display_image,
    display_video,
    display_gif,
    save_as_sixel,
    check_sixel_support,
    get_file_type,
)


def main():
    parser = argparse.ArgumentParser(
        prog='termplay',
        description='Display images and videos in terminal using SIXEL graphics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  termplay image.png                    # Display an image
  termplay animation.gif --loop         # Play animated GIF in loop
  termplay video.mp4 --fps 15           # Play video at 15 FPS
  termplay photo.jpg --width 400        # Display image with max width 400px
  termplay image.png --save             # Save as .sixel file
  termplay image.png --save -o out.sixel # Save with custom filename

Supported formats:
  Images: PNG, JPG, JPEG, BMP, WebP, TIFF, ICO
  Videos: MP4, AVI, MKV, MOV, WMV, FLV, WebM (requires av or opencv-python)
  Animated: GIF

Note: Requires a SIXEL-compatible terminal such as:
  - Windows Terminal (native support)
  - mlterm
  - xterm (compiled with --enable-sixel-graphics)
  - mintty (Windows)
  - WezTerm
  - Contour

Tip: Use --save to create .sixel files, then view with: type filename.sixel (only for images)
        '''
    )
    
    parser.add_argument('file', help='Path to image or video file')
    parser.add_argument('-w', '--width', type=int, default=800,
                        help='Maximum width in pixels (default: 800)')
    parser.add_argument('-H', '--height', type=int, default=600,
                        help='Maximum height in pixels (default: 600)')
    parser.add_argument('-c', '--colors', type=int, default=256,
                        help='Number of colors (2-256, default: 256)')
    parser.add_argument('-f', '--fps', type=float, default=None,
                        help='Frames per second for video playback (default: native video FPS)')
    parser.add_argument('-l', '--loop', action='store_true',
                        help='Loop video/GIF playback')
    parser.add_argument('-s', '--save', action='store_true',
                        help='Save as .sixel file instead of displaying (images only)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output filename for --save (default: <input>.sixel)')
    parser.add_argument('--no-check', action='store_true',
                        help='Skip terminal SIXEL support check')
    parser.add_argument('-v', '--version', action='version', 
                        version='%(prog)s 1.0.0')
    
    args = parser.parse_args()
    
    try:
        # Validate arguments
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        
        args.colors = max(2, min(256, args.colors))
        
        # Check terminal support
        if not args.no_check and not args.save:
            check_sixel_support()
        
        # Determine file type and display
        file_type = get_file_type(args.file)
        
        # Save mode
        if args.save:
            if file_type == 'video':
                print("Error: Cannot save video as .sixel file. Only images are supported.")
                sys.exit(1)
            save_as_sixel(args.file, args.output, args.width, args.height, args.colors)
            return
        
        if file_type == 'image':
            display_image(args.file, args.width, args.height, args.colors)
        elif file_type == 'gif':
            display_gif(args.file, args.width, args.height, args.colors, args.loop)
        elif file_type == 'video':
            # Use smaller size and fewer colors for video performance
            video_width = min(args.width, 320)
            video_height = min(args.height, 240)
            video_colors = min(args.colors, 32)
            display_video(args.file, video_width, video_height, args.fps, 
                          video_colors, args.loop)
        else:
            # Try as image anyway
            print(f"Unknown file type, attempting to display as image...")
            display_image(args.file, args.width, args.height, args.colors)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)


if __name__ == '__main__':
    main()
