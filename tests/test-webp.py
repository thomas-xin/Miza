import pyvips
import glob
import os

def create_animated_webp(input_dir, output_path, fps=30, loop=0, lossless=True):
    """
    Create animated WebP from PNG sequence using pyvips/libvips.
    
    Args:
        input_dir (str): Directory containing PNG files
        output_path (str): Output WebP file path
        fps (int): Frames per second for animation
        loop (int): Number of loops (0 = infinite)
        lossless (bool): Use lossless compression
    """
    # Calculate delay per frame in milliseconds
    delay_per_frame = int(1000 / fps)
    
    # Get sorted list of PNG files (ensure proper numerical sorting)
    png_files = sorted(glob.glob(os.path.join(input_dir, "*.png")),
                       key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    # Load images and store in memory (adjust access mode for efficiency)
    images = [pyvips.Image.new_from_file(png, access="sequential") 
             for png in png_files]
    
    # Create parameters for WebP save
    save_args = {
        # "delay": [delay_per_frame] * len(images),
        # "loop": loop,
        "lossless": lossless,
        # "minimize_size": False,  # Disable for faster encoding
        "strip": True,          # Remove metadata
        "smart_subsample": True,
        "reduction_effort": 0    # Faster encoding (0-6)
    }
    
    # Create multi-page image and save as WebP
    joined = pyvips.Image.arrayjoin(images, across=1)
    joined.webpsave(output_path, **save_args)
    
    print(f"Created animated WebP: {output_path}")
    print(f"Total frames: {len(images)} | Dimensions: {images[0].width}x{images[0].height}")

# Example usage
create_animated_webp(
    input_dir="1739334505915778200",
    output_path="animation.webp",
    fps=30,
    loop=0,
    lossless=True
)