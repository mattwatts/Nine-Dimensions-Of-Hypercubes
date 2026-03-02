"""
UNE DEEP RESEARCH PROTOCOL v2.2 - SOVEREIGN CODE
SCRIPT: logic_garden_v67_loop.py
MODE:   Nursery (Galactic Palette)
TARGET: Milky Way (Seamless Loop)
STYLE:  "The Galactic City Loop" | 40s Deep Time | 4K Ready

AUTHOR: Matt Watts / Assistant Protocol
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# --- 1. THE GALACTIC PALETTE ---
BG_VOID = "#020205"
CORE_GOLD = "#FDB813"
ARM_BLUE = "#00BFFF"     # Deep Sky Blue
ARM_CYAN = "#E0FFFF"     # Light Cyan
DUST_RED = "#8B0000"     # Dark lanes

# --- 2. CONFIGURATION ---
FPS = 30
DURATION = 40
TOTAL_FRAMES = FPS * DURATION

class MilkyWayLoopSim:
    def __init__(self):
        self.num_stars = 15000
        
        # 1. INITIALIZE GEOMETRY (BASE)
        # We store base coordinates and rotate them every frame
        # This prevents floating point drift and ensures perfect loop
        
        # BULGE (30%)
        n_bulge = int(self.num_stars * 0.3)
        r_bulge = np.random.normal(0, 1.5, n_bulge)
        theta_bulge = np.random.uniform(0, 2*np.pi, n_bulge)
        
        # Elongated Bar
        bx = r_bulge * np.cos(theta_bulge) * 2.0 
        by = r_bulge * np.sin(theta_bulge) * 1.0
        
        # ARMS (70%)
        n_arms = self.num_stars - n_bulge
        arm_idx = np.random.randint(0, 2, n_arms) # 2 Main Arms
        t = np.random.uniform(0, 4*np.pi, n_arms) # Distance along arm
        
        # Log Spiral: r = a * e^(b*theta) -> Visual Approximation
        r_arm = 2.0 + t * 1.5
        spread = np.random.normal(0, 0.5 + t*0.1, n_arms) # Thicker at edges
        
        theta_base = t + (arm_idx * np.pi) # 180 deg offset
        
        ax_pts = (r_arm) * np.cos(theta_base + spread/r_arm)
        ay_pts = (r_arm) * np.sin(theta_base + spread/r_arm)
        
        # COMBINE
        self.x_base = np.concatenate([bx, ax_pts])
        self.y_base = np.concatenate([by, ay_pts])
        
        # COLORS & SIZES
        self.colors = np.zeros((self.num_stars, 4))
        self.sizes = np.zeros(self.num_stars)
        
        # Bulge Colors
        c_gold = matplotlib.colors.to_rgb(CORE_GOLD)
        self.colors[:n_bulge, :3] = c_gold
        self.colors[:n_bulge, 3] = 0.5
        self.sizes[:n_bulge] = np.random.uniform(1, 3, n_bulge)
        
        # Arm Colors (Blue/Cyan Mix)
        c_blue = matplotlib.colors.to_rgb(ARM_BLUE)
        c_cyan = matplotlib.colors.to_rgb(ARM_CYAN)
        mix = np.random.rand(n_arms, 1)
        
        self.colors[n_bulge:, :3] = c_blue * mix + c_cyan * (1-mix)
        self.colors[n_bulge:, 3] = 0.6
        self.sizes[n_bulge:] = np.random.uniform(0.5, 2.5, n_arms)
        
        # THE SUN (Base Position)
        self.sun_r = 12.0
        self.sun_theta_base = -0.5 

    def get_rotated_coords(self, angle):
        # Rotation Matrix
        c, s = np.cos(angle), np.sin(angle)
        x_new = self.x_base * c - self.y_base * s
        y_new = self.x_base * s + self.y_base * c
        return x_new, y_new

    def render(self, frame_idx, fig):
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG_VOID)
        
        # 1. CALCULATE LOOP ANGLES
        # Exact 2*PI rotation over TOTAL_FRAMES
        prog = frame_idx / float(TOTAL_FRAMES)
        angle = prog * 2 * np.pi
        
        # Rotate Galaxy (Clockwise visually, so negative angle)
        x_rot, y_rot = self.get_rotated_coords(-angle)
        
        # Limits
        lim = 25
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_axis_off()
        
        # 2. DRAW STARS
        ax.scatter(x_rot, y_rot, c=self.colors, s=self.sizes)
        
        # 3. DRAW SUN
        # Rotate Sun
        sun_angle = self.sun_theta_base - angle
        sx = self.sun_r * np.cos(sun_angle)
        sy = self.sun_r * np.sin(sun_angle)
        
        # Pulse: Exact integer cycles (5) for resonance
        pulse_phase = prog * 2 * np.pi * 5
        pulse = 60 + 20 * np.sin(pulse_phase)
        
        # Draw Crosshair
        ax.scatter([sx], [sy], color="white", s=pulse, marker='+', zorder=10, linewidth=1.5)
        # Small Dot
        ax.scatter([sx], [sy], color="white", s=15, zorder=11)
        
        # 4. HUD
        fig.text(0.5, 0.92, "LOGIC GARDEN 67: THE GALACTIC CITY", color="white", ha='center', fontsize=16, fontweight='bold', fontfamily='monospace')
        
        # We rotate the label too? No, keep label static but near sun? 
        # Or just keep labels fixed on screen.
        fig.text(0.5, 0.05, "ROTATION: SEAMLESS", color=ARM_BLUE, ha='center', fontsize=10, fontfamily='monospace', alpha=0.6)
        
        # Dynamic Label for Sun?
        # Offset text slightly
        ax.text(sx + 1.5, sy + 1.5, "YOU ARE HERE", color="white", fontsize=8, fontfamily='monospace', alpha=0.9)

        # Save
        out_dir = "logic_garden_milkyway_loop_frames"
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"galaxy_loop_{frame_idx:04d}.png")
        plt.savefig(filename, facecolor=BG_VOID)
        plt.close()

# --- 3. EXECUTION ---
if __name__ == "__main__":
    print(f"[NURSERY] Spinning the Void...")
    
    sim = MilkyWayLoopSim()
    
    # Square aspect for loop
    for i in range(TOTAL_FRAMES):
        fig = plt.figure(figsize=(10, 10), dpi=100)
        
        sim.render(i, fig)
        
        if i % 60 == 0:
            print(f"Frame {i}/{TOTAL_FRAMES}")
