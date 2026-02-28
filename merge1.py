import os
import time
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool, cpu_count
import textwrap

# --- CONFIGURATION ---
OUTPUT_DIR = "frames"
WIDTH = 1800
HEIGHT = 1800
SUB_SIZE = 600
TOTAL_FRAMES = 1080
# TEXT_COLOR = (255, 255, 255)
TEXT_COLOR = '#4169E1' # RoyalBlue
BG_COLOR = (0, 0, 0) # Midnight Black context

# Input Matrix
INPUTS = [
    {"dir": "frames_1D", "label": "Line", 'pos': (0, 0)},
    {"dir": "frames_2D", "label": "Square", 'pos': (600, 0)},
    {"dir": "frames_3D", "label": "Cube", 'pos': (1200, 0)},
    {"dir": "frames_4D", "label": "Tesseract", 'pos': (0, 600)},
    {"dir": "frames_5D", "label": "Penteract", 'pos': (600, 600)},
    {"dir": "frames_6D", "label": "Hexeract", 'pos': (1200, 600)},
    {"dir": "frames_7D", "label": "Hepteract", 'pos': (0, 1200)},
    {"dir": "frames_8D", "label": "Octeract", 'pos': (600, 1200)},
    {"dir": "frames_9D", "label": "Enneract", 'pos': (1200, 1200)},
]

def get_font():
    """Industrial fallback for font loading."""
    try:
        # Try a standard sans-serif font
        return ImageFont.truetype("DejaVuSans.ttf", 30)
    except IOError:
        try:
            return ImageFont.truetype("arial.ttf", 30)
        except IOError:
            # Fallback to default if system constraints prevent custom fonts
            return ImageFont.load_default()

def process_frame(frame_idx):
    """
    Worker node function to process a single composite frame.
    Self-contained to ensure process isolation.
    """
    filename = f"frame_{frame_idx:04d}.png"
    
    # Initialize Canvas (Midnight Black)
    canvas = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    
    # Load Font per process (avoids pickling issues)
    font = get_font()

    for item in INPUTS:
        input_path = os.path.join(item["dir"], filename)
        x_offset, y_offset = item["pos"]
        
        try:
            # Open source Frame
            with Image.open(input_path) as src:
                # Resize if not exactly 600x600 (Safety Check)
                if src.size != (SUB_SIZE, SUB_SIZE):
                    src = src.resize((SUB_SIZE, SUB_SIZE))
                
                # Paste into Grid
                canvas.paste(src, (x_offset, y_offset))
                
                # Overlay Label (Bottom Left of sub-frame)
                # Position: x + 10 padding, y + height - 40 padding
                text_pos = (x_offset + 15, y_offset + SUB_SIZE - 40)
                draw.text(text_pos, item["label"], fill=TEXT_COLOR, font=font)
                
        except FileNotFoundError:
            # Error Handling: Fill with placeholder or log error
            print(f"[!] Phantom Entity: Missing {input_path}")
            
    # Save Output
    output_path = os.path.join(OUTPUT_DIR, filename)
    canvas.save(output_path, "PNG")
    return frame_idx

def main():
    # 1. Setup Infrastructure
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"[*] Initializing Industrial Merge Protocol...")
    print(f"[*] Target: {TOTAL_FRAMES} frames | Resolution: {WIDTH}x{HEIGHT}")
    print(f"[*] Cores Available: {cpu_count()}")

    # 2. Batch Execution
    start_time = time.time()
    
    # Create iterable for frame indices
    frames = range(TOTAL_FRAMES)
    
    with Pool(processes=cpu_count()) as pool:
        # Map worker function to frames
        results = pool.map(process_frame, frames)

    # 3. Final metrics
    elapsed = time.time() - start_time
    print(f"[*] Protocol Complete. Processed {len(results)} frames.")
    print(f"[*] Throughput: {len(results)/elapsed:.2f} fps")
    print(f"[*] Output Directory: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
