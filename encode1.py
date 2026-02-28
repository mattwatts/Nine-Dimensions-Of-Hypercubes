import subprocess
import os
import sys

# --- CONFIGURATION ---
INPUT_DIR = "frames"
OUTPUT_FILE = "Hypercubes.mp4"
FRAME_RATE = 24  # 1080 frames / 45 seconds = 24 fps
RESOLUTION = "1800x1800"

def run_ffmpeg():
    """
    Executes ffmpeg as a subprocess to ensure memory efficiency 
    and leverage system-level optimization.
    """
    
    # Check for frames
    if not os.path.exists(INPUT_DIR):
        print(f"[!] Error: Directory '{INPUT_DIR}' not found. Run Task 1 first.")
        sys.exit(1)

    input_pattern = os.path.join(INPUT_DIR, "frame_%04d.png")
    
    cmd = [
        "ffmpeg",
        "-y",                      # Overwrite output without asking
        "-framerate", str(FRAME_RATE),
        "-i", input_pattern,       # Input sequence
        "-c:v", "libx264",         # Industrial standard codec
        "-pix_fmt", "yuv420p",     # Widest compatibility
        "-crf", "18",              # Visually lossless
        "-vf", f"scale={RESOLUTION}", # Force resolution constraints
        OUTPUT_FILE
    ]
    
    print(f"[*] Initiating Encoding Sequence: {' '.join(cmd)}")
    
    try:
        # Run subprocess and stream output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Capture standard error (ffmpeg writes progress to stderr)
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print(f"[*] Success: Video exported to {os.path.abspath(OUTPUT_FILE)}")
        else:
            print(f"[!] Encoding Failed:\n{stderr}")
            
    except OSError as e:
        print(f"[!] OS Error: ffmpeg not found or not executable. Ensure ffmpeg is in PATH.")
        print(f"[!] Details: {e}")

if __name__ == "__main__":
    run_ffmpeg()
