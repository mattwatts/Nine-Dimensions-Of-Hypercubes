"""
UNE DEEP RESEARCH PROTOCOL v2.2 - SOVEREIGN CODE
SCRIPT: logic_garden_v68_turret_fixed.py
MODE:   Nursery (Combat Palette)
TARGET: Active Protection System (Slewing Turret)
STYLE:  "The Hard Kill: Barrel Sync" | 20s | 5 Shots | 4K Ready
STATUS: PATCHED (Restored CALC_GOLD)

AUTHOR: Matt Watts / Assistant Protocol
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import os

# --- 1. THE COMBAT PALETTE ---
BG_VOID = "#050505"
RADAR_SWEEP = "#004444"
RADAR_LOCK = "#00FF00"
THREAT_RED = "#FF0000"
THREAT_TRAIL = "#880000"
BARREL_GREY = "#606060"
INTERCEPT_WHITE = "#FFFFFF"
CALC_GOLD = "#FFD700"  # Restored

# --- 2. CONFIGURATION ---
FPS = 30
DURATION = 20
TOTAL_FRAMES = FPS * DURATION

class APSTurretSim:
    def __init__(self):
        # ASSET
        self.asset_pos = np.array([0.0, 0.0])
        self.turret_angle = np.pi / 2 # Start facing Up
        self.barrel_len = 1.8 
        
        # LOGIC
        self.threats = [] 
        self.countermeasures = [] 
        self.debris = [] 
        
        self.kills = 0
        self.shots_planned = 5
        self.shots_generated = 0
        self.cooldown = 45
        
        self.radar_angle = 0.0

    def spawn_threat(self):
        angle = np.random.uniform(0, 2*np.pi)
        dist = 14.0 
        pos = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        
        speed = np.random.uniform(0.30, 0.45)
        
        # Incoming Vector
        error_angle = np.random.uniform(-0.02, 0.02)
        aim_angle = angle + np.pi + error_angle
        vel = np.array([speed * np.cos(aim_angle), speed * np.sin(aim_angle)])
        
        self.threats.append({
            "pos": pos,
            "vel": vel,
            "id": np.random.randint(1000, 9999), 
            "locked": False,
            "intercepted": False,
            "cm_fired": False,
            "aim_point": None 
        })

    def get_intercept_solution(self, threat):
        # Solve for Aim Vector
        dist = np.linalg.norm(threat["pos"])
        speed_t = np.linalg.norm(threat["vel"])
        speed_cm = 0.9 
        
        t_est = dist / (speed_cm + speed_t)
        intercept_pt = threat["pos"] + threat["vel"] * t_est
        
        aim_vec = intercept_pt - self.asset_pos
        return aim_vec, intercept_pt

    def update_turret(self):
        # Look at the closest active threat
        active_threats = [t for t in self.threats if not t["intercepted"]]
        
        target_angle = self.turret_angle # Default: Hold position
        
        if active_threats:
            # Sort by distance
            active_threats.sort(key=lambda t: np.sum(t["pos"]**2))
            target = active_threats[0]
            
            # Aim at INTERCEPT location
            aim_vec, _ = self.get_intercept_solution(target)
            target_angle = np.arctan2(aim_vec[1], aim_vec[0])
            
        # Slew towards target
        diff = target_angle - self.turret_angle
        diff = (diff + np.pi) % (2*np.pi) - np.pi
        
        slew_rate = 0.35 # Rad/frame
        
        if abs(diff) < slew_rate:
            self.turret_angle = target_angle
        else:
            self.turret_angle += np.sign(diff) * slew_rate

    def fire_cm(self, threat):
        aim_vec, _ = self.get_intercept_solution(threat)
        aim_norm = aim_vec / np.linalg.norm(aim_vec)
        
        speed_cm = 0.9
        vel_cm = aim_norm * speed_cm
        
        # SPAWN AT BARREL TIP
        start_pos = np.array([
            np.cos(self.turret_angle) * self.barrel_len,
            np.sin(self.turret_angle) * self.barrel_len
        ])
        
        self.countermeasures.append({
            "pos": start_pos,
            "vel": vel_cm,
            "target_id": threat["id"],
            "life": 40
        })

    def update(self, frame_idx):
        # 1. RADAR & TURRET
        self.radar_angle -= 0.15
        self.update_turret()
        
        # 2. SPAWN LOGIC (5 shots / 20s)
        if self.shots_generated < self.shots_planned:
            if self.cooldown <= 0:
                self.spawn_threat()
                self.shots_generated += 1
                self.cooldown = np.random.randint(60, 150)
            else:
                self.cooldown -= 1
                
        # 3. THREATS
        for t in self.threats:
            if t["intercepted"]: continue
            
            t["pos"] += t["vel"]
            dist_sq = np.sum(t["pos"]**2)
            
            if dist_sq < 100.0: 
                t["locked"] = True
                
                # Check Turret Alignment
                aim_vec, _ = self.get_intercept_solution(t)
                req_angle = np.arctan2(aim_vec[1], aim_vec[0])
                curr = self.turret_angle
                diff = abs((req_angle - curr + np.pi) % (2*np.pi) - np.pi)
                
                aligned = diff < 0.1
                panic = dist_sq < 25.0
                
                if not t["cm_fired"] and (aligned or panic):
                    self.fire_cm(t)
                    t["cm_fired"] = True
            
            if dist_sq < 0.2:
                t["intercepted"] = True

        # 4. CMs
        active_cms = []
        for cm in self.countermeasures:
            cm["pos"] += cm["vel"]
            cm["life"] -= 1
            
            target = next((t for t in self.threats if t["id"] == cm["target_id"]), None)
            
            hit = False
            if target and not target["intercepted"]:
                d = np.linalg.norm(cm["pos"] - target["pos"])
                if d < 1.0: 
                    target["intercepted"] = True
                    self.kills += 1
                    self.spawn_debris(cm["pos"], target["vel"], cm["vel"])
                    hit = True
            if not hit and cm["life"] > 0:
                active_cms.append(cm)
        self.countermeasures = active_cms

        # 5. DEBRIS
        active_debris = []
        for p in self.debris:
            p["pos"] += p["vel"]
            p["vel"] *= 0.9
            p["life"] -= 0.03
            if p["life"] > 0:
                active_debris.append(p)
        self.debris = active_debris
        
        self.threats = [t for t in self.threats if not t["intercepted"]]

    def spawn_debris(self, pos, t_vel, cm_vel):
        for _ in range(20):
            vel = t_vel * 0.1 + np.random.normal(0, 0.2, 2)
            self.debris.append({"pos": pos.copy(), "vel": vel, "life": 1.0, "color": INTERCEPT_WHITE, "size": 12})
        for _ in range(10):
            vel = cm_vel * 0.1 + np.random.normal(0, 0.15, 2)
            self.debris.append({"pos": pos.copy(), "vel": vel, "life": 1.2, "color": THREAT_RED, "size": 8})

    def render(self, frame_idx, ax):
        limit = 12
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_facecolor(BG_VOID)
        
        # 1. RADAR
        ax.add_artist(plt.Circle((0,0), 10, color=RADAR_SWEEP, alpha=0.05))
        ax.add_artist(plt.Circle((0,0), 5, color=RADAR_SWEEP, alpha=0.05, linestyle=':'))
        
        # 2. TANK CHASSIS (Static)
        ax.add_artist(Rectangle((-0.75, -1.0), 1.5, 2.0, color="#252525", zorder=5))
        
        # 3. TURRET (Dynamic)
        bx = np.cos(self.turret_angle) * self.barrel_len
        by = np.sin(self.turret_angle) * self.barrel_len
        ax.plot([0, bx], [0, by], color=BARREL_GREY, linewidth=4, zorder=6)
        ax.add_artist(plt.Circle((0,0), 0.6, color="#454545", zorder=7))
        
        # 4. THREATS
        for t in self.threats:
            start_x = t["pos"][0] - t["vel"][0]*4
            start_y = t["pos"][1] - t["vel"][1]*4
            ax.plot([start_x, t["pos"][0]], [start_y, t["pos"][1]], 
                    color=THREAT_TRAIL, linewidth=1.5, alpha=0.8)
            ax.scatter(t["pos"][0], t["pos"][1], color=THREAT_RED, s=40, marker='v', zorder=10)
            
            if t["locked"]:
                # Line from Turret to Threat (Targeting Laser)
                ax.plot([0, t["pos"][0]], [0, t["pos"][1]], color=CALC_GOLD, alpha=0.2, linestyle="--", linewidth=0.5)

        # 5. PROJECTILES
        for cm in self.countermeasures:
            ax.scatter(cm["pos"][0], cm["pos"][1], color=INTERCEPT_WHITE, s=30, marker='o', zorder=12)

        # 6. DEBRIS
        for d in self.debris:
            # ALPHA CLAMP
            alpha_val = max(0.0, min(1.0, d["life"]))
            ax.scatter(d["pos"][0], d["pos"][1], color=d["color"], s=d["size"]*alpha_val, alpha=alpha_val, zorder=8)

        # 7. HUD
        fig.text(0.5, 0.92, "LOGIC GARDEN 68: THE HARD KILL [SLEW]", color="white", ha='center', fontsize=16, fontweight='bold', fontfamily='monospace')
        
        rem_shots = self.shots_planned - self.shots_generated
        status = "TARGETING: ACTIVE"
        col_stat = RADAR_LOCK
        
        if rem_shots == 0 and len(self.threats) == 0:
            status = "SECTOR SECURE"
            col_stat = INTERCEPT_WHITE
            
        ax.text(0, -11, f"{status} | KILLS: {self.kills}/5", color=col_stat, ha='center', fontsize=14, fontweight='bold', fontfamily='monospace',
                bbox=dict(facecolor='black', edgecolor=col_stat, pad=6, alpha=0.9))
        
        # Save
        out_dir = "logic_garden_turret_fixed_frames"
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"turret_{frame_idx:04d}.png")
        plt.savefig(filename, facecolor=BG_VOID)
        plt.close()

# --- 3. EXECUTION ---
if __name__ == "__main__":
    print(f"[NURSERY] Turret Servos Online (Retry)...")
    
    sim = APSTurretSim()
    
    for i in range(TOTAL_FRAMES):
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        
        sim.update(i)
        sim.render(i, ax)
        plt.close()
        
        if i % 60 == 0:
            print(f"Frame {i}/{TOTAL_FRAMES}")
