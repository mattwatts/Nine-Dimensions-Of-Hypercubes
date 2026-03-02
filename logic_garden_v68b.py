"""
UNE DEEP RESEARCH PROTOCOL v2.2 - SOVEREIGN CODE
SCRIPT: logic_garden_v68_turn.py
MODE:   Nursery (Combat Palette)
TARGET: Active Protection System (Evasive Turn)
STYLE:  "The Hard Kill: The Pivot" | 20s | 5 Shots | 4K Ready

AUTHOR: Matt Watts / Assistant Protocol
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import os

# --- 1. THE COMBAT PALETTE ---
BG_VOID = "#050505"
RADAR_SWEEP = "#004444"
RADAR_LOCK = "#00FF00"
THREAT_RED = "#FF0000"
THREAT_TRAIL = "#880000"
BARREL_GREY = "#606060"
INTERCEPT_WHITE = "#FFFFFF"
CALC_GOLD = "#FFD700"
TRACK_TRAIL = "#151515" 

# --- 2. CONFIGURATION ---
FPS = 30
DURATION = 20
TOTAL_FRAMES = FPS * DURATION

class APSTurnSim:
    def __init__(self):
        # WAYPOINTS
        self.p0 = np.array([0.0, -8.0])   # Start
        self.p1 = np.array([0.0, -2.0])   # Pivot Point (1/3 up)
        self.p2 = np.array([-8.0, 8.0])   # End (Top Left)
        
        self.asset_pos = self.p0.copy()
        self.chassis_angle = np.pi / 2    # Facing North initially
        
        # Turret
        self.turret_angle = np.pi / 2 
        self.barrel_len = 1.8 
        
        # LOGIC
        self.threats = [] 
        self.countermeasures = [] 
        self.debris = [] 
        self.tracks_l = [] 
        self.tracks_r = [] 
        
        self.kills = 0
        self.shots_planned = 5
        self.shots_generated = 0
        self.cooldown = 45 # Initial delay
        
        self.radar_angle = 0.0

    def get_rotated_point(self, point, center, angle):
        # Rotate a point around a center
        c, s = np.cos(angle), np.sin(angle)
        px, py = point
        cx, cy = center
        
        # Translate to origin
        x = px - cx
        y = py - cy
        
        # Rotate
        x_new = x * c - y * s
        y_new = x * s + y * c
        
        # Translate back
        return np.array([x_new + cx, y_new + cy])

    def spawn_threat(self):
        # Spawn relative to CURRENT tank pos
        angle = np.random.uniform(0, 2*np.pi)
        dist = 14.0 
        
        offset = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        pos = self.asset_pos + offset
        
        speed = np.random.uniform(0.35, 0.50)
        
        # Aim at TANK
        to_tank = self.asset_pos - pos
        target_angle = np.arctan2(to_tank[1], to_tank[0])
        error = np.random.uniform(-0.03, 0.03)
        aim_angle = target_angle + error
        
        vel = np.array([speed * np.cos(aim_angle), speed * np.sin(aim_angle)])
        
        self.threats.append({
            "pos": pos,
            "vel": vel,
            "id": np.random.randint(1000, 9999), 
            "locked": False,
            "intercepted": False,
            "cm_fired": False
        })

    def get_intercept_solution(self, threat):
        # Improve intercept logic to account for Tank Velocity?
        # For this sim, Tank is slow enough to ignore self-velocity for the firing solution.
        dist = np.linalg.norm(threat["pos"] - self.asset_pos)
        speed_t = np.linalg.norm(threat["vel"])
        speed_cm = 0.9 
        
        t_est = dist / (speed_cm + speed_t)
        intercept_pt = threat["pos"] + threat["vel"] * t_est
        
        aim_vec = intercept_pt - self.asset_pos
        return aim_vec, intercept_pt

    def update_turret(self):
        active = [t for t in self.threats if not t["intercepted"]]
        
        if active:
            active.sort(key=lambda t: np.sum((t["pos"] - self.asset_pos)**2))
            target = active[0]
            aim_vec, _ = self.get_intercept_solution(target)
            target_angle = np.arctan2(aim_vec[1], aim_vec[0])
        else:
            # Idle: Look where we are driving
            target_angle = self.chassis_angle
            
        diff = target_angle - self.turret_angle
        diff = (diff + np.pi) % (2*np.pi) - np.pi
        
        slew_rate = 0.45 # Fast slew
        if abs(diff) < slew_rate:
            self.turret_angle = target_angle
        else:
            self.turret_angle += np.sign(diff) * slew_rate

    def fire_cm(self, threat):
        aim_vec, _ = self.get_intercept_solution(threat)
        aim_norm = aim_vec / np.linalg.norm(aim_vec)
        speed_cm = 0.9
        vel_cm = aim_norm * speed_cm
        
        start_pos = self.asset_pos + np.array([
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
        # 1. MOVEMENT LOGIC (The Pivot)
        prog = frame_idx / float(TOTAL_FRAMES)
        pivot_time = 1.0 / 3.0
        
        if prog < pivot_time:
            # Phase 1: P0 to P1
            # Normalize t to 0-1 for this segment
            t = prog / pivot_time
            self.asset_pos = self.p0 * (1-t) + self.p1 * t
            
            # Heading: North
            self.chassis_angle = np.pi / 2
            
        else:
            # Phase 2: P1 to P2
            # Normalize t to 0-1 for this segment
            t = (prog - pivot_time) / (1.0 - pivot_time)
            self.asset_pos = self.p1 * (1-t) + self.p2 * t
            
            # Heading: P1 to P2 vector
            vec = self.p2 - self.p1
            self.chassis_angle = np.arctan2(vec[1], vec[0])

        # 2. GENERATE TRACKS
        # Back Left and Back Right corners relative to rotation
        # Tank Dimensions: Width 1.4, Length 2.0. Center at asset_pos.
        # Rear offsets: (-0.6, -0.9), (0.6, -0.9) approx
        
        # We need to rotate these offsets by chassis_angle relative to (0,1) (Forward)
        # Ah, chassis_angle is standard math angle (0 = East).
        # My offsets assume facing North?
        # Let's define offsets assuming heading is 0 (East).
        # Rear is x = -1.0. Left is y = 0.6.
        # Rear Left: (-1.0, 0.6). Rear Right: (-1.0, -0.6)
        
        # Rotation Matrix
        c, s = np.cos(self.chassis_angle), np.sin(self.chassis_angle)
        
        # Local offsets
        off_l = np.array([-1.0, 0.6])
        off_r = np.array([-1.0, -0.6])
        
        # Global offsets
        gl_l = np.array([off_l[0]*c - off_l[1]*s, off_l[0]*s + off_l[1]*c])
        gl_r = np.array([off_r[0]*c - off_r[1]*s, off_r[0]*s + off_r[1]*c])
        
        self.tracks_l.append(self.asset_pos + gl_l)
        self.tracks_r.append(self.asset_pos + gl_r)

        # 3. COMBAT LOGIC
        self.radar_angle -= 0.15
        self.update_turret()
        
        if self.shots_generated < self.shots_planned:
            if self.cooldown <= 0:
                self.spawn_threat()
                self.shots_generated += 1
                self.cooldown = np.random.randint(60, 150)
            else:
                self.cooldown -= 1
                
        # 4. THREAT UPDATE
        for t in self.threats:
            if t["intercepted"]: continue
            t["pos"] += t["vel"]
            dist_sq = np.sum((t["pos"] - self.asset_pos)**2)
            
            if dist_sq < 100.0:  
                t["locked"] = True
                
                # Check alignment
                aim_vec, _ = self.get_intercept_solution(t)
                req_angle = np.arctan2(aim_vec[1], aim_vec[0])
                diff = abs((req_angle - self.turret_angle + np.pi) % (2*np.pi) - np.pi)
                
                if not t["cm_fired"] and (diff < 0.15 or dist_sq < 25):
                    self.fire_cm(t)
                    t["cm_fired"] = True
            
            if dist_sq < 0.2:
                t["intercepted"] = True

        # 5. CM UPDATE
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

        # 6. DEBRIS
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
        for _ in range(8):
            vel = cm_vel * 0.1 + np.random.normal(0, 0.15, 2)
            self.debris.append({"pos": pos.copy(), "vel": vel, "life": 1.2, "color": THREAT_RED, "size": 8})

    def render(self, frame_idx, ax):
        limit = 12
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_facecolor(BG_VOID)
        
        # 1. TRACKS (Historical)
        if len(self.tracks_l) > 1:
            lx, ly = zip(*self.tracks_l)
            rx, ry = zip(*self.tracks_r)
            ax.plot(lx, ly, color=TRACK_TRAIL, linestyle="-", linewidth=2.5)
            ax.plot(rx, ry, color=TRACK_TRAIL, linestyle="-", linewidth=2.5)
        
        # 2. RADAR DOME (Moving)
        ax.add_artist(plt.Circle((self.asset_pos[0], self.asset_pos[1]), 10, color=RADAR_SWEEP, alpha=0.05))
        ax.add_artist(plt.Circle((self.asset_pos[0], self.asset_pos[1]), 5, color=RADAR_SWEEP, alpha=0.05, linestyle=':'))
        
        # 3. TANK CHASSIS (Rotated Polygon)
        # Define chassis as 2.0 long (x), 1.5 wide (y) centered at 0, facing East (Angle 0)
        # Corners: (1, 0.75), (-1, 0.75), (-1, -0.75), (1, -0.75)
        # Because we used angle 0 as East in math.
        
        corners_local = [
            np.array([ 1.0,  0.7]),
            np.array([-1.0,  0.7]),
            np.array([-1.0, -0.7]),
            np.array([ 1.0, -0.7])
        ]
        
        # Rotate and Translate
        c, s = np.cos(self.chassis_angle), np.sin(self.chassis_angle)
        corners_global = []
        for pt in corners_local:
            x_rot = pt[0]*c - pt[1]*s
            y_rot = pt[0]*s + pt[1]*c
            corners_global.append(self.asset_pos + np.array([x_rot, y_rot]))
            
        poly = Polygon(corners_global, closed=True, facecolor="#252525", edgecolor="none", zorder=5)
        ax.add_artist(poly)
        
        # 4. TURRET
        bx = self.asset_pos[0] + np.cos(self.turret_angle) * self.barrel_len
        by = self.asset_pos[1] + np.sin(self.turret_angle) * self.barrel_len
        ax.plot([self.asset_pos[0], bx], [self.asset_pos[1], by], color=BARREL_GREY, linewidth=4, zorder=6)
        ax.add_artist(plt.Circle((self.asset_pos[0], self.asset_pos[1]), 0.65, color="#454545", zorder=7))
        
        # 5. THREATS
        for t in self.threats:
            start_x = t["pos"][0] - t["vel"][0]*4
            start_y = t["pos"][1] - t["vel"][1]*4
            ax.plot([start_x, t["pos"][0]], [start_y, t["pos"][1]], 
                    color=THREAT_TRAIL, linewidth=1.5, alpha=0.8)
            ax.scatter(t["pos"][0], t["pos"][1], color=THREAT_RED, s=40, marker='v', zorder=10)
            
            if t["locked"]:
                ax.plot([self.asset_pos[0], t["pos"][0]], [self.asset_pos[1], t["pos"][1]], 
                        color=CALC_GOLD, alpha=0.2, linestyle="--", linewidth=0.5)

        # 6. CMs
        for cm in self.countermeasures:
            ax.scatter(cm["pos"][0], cm["pos"][1], color=INTERCEPT_WHITE, s=30, marker='o', zorder=12)

        # 7. DEBRIS
        for d in self.debris:
            alpha_val = max(0.0, min(1.0, d["life"]))
            ax.scatter(d["pos"][0], d["pos"][1], color=d["color"], s=d["size"]*alpha_val, alpha=alpha_val, zorder=8)

        # 8. HUD
        fig.text(0.5, 0.92, "LOGIC GARDEN 68: THE HARD KILL [TURN]", color="white", ha='center', fontsize=16, fontweight='bold', fontfamily='monospace')
        
        rem_shots = self.shots_planned - self.shots_generated
        status = "EVASIVE MANEUVER"
        col_stat = RADAR_LOCK
        
        if rem_shots == 0 and len(self.threats) == 0:
            status = "TARGET AREA REACHED"
            col_stat = INTERCEPT_WHITE
            
        ax.text(0, -11, f"{status} | KILLS: {self.kills}/5", color=col_stat, ha='center', fontsize=14, fontweight='bold', fontfamily='monospace',
                bbox=dict(facecolor='black', edgecolor=col_stat, pad=6, alpha=0.9))
        
        # Save
        out_dir = "logic_garden_turn_frames"
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"turn_{frame_idx:04d}.png")
        plt.savefig(filename, facecolor=BG_VOID)
        plt.close()

# --- 3. EXECUTION ---
if __name__ == "__main__":
    print(f"[NURSERY] Executing Pivot at T-140...")
    
    sim = APSTurnSim()
    
    for i in range(TOTAL_FRAMES):
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        
        sim.update(i)
        sim.render(i, ax)
        plt.close()
        
        if i % 60 == 0:
            print(f"Frame {i}/{TOTAL_FRAMES}")
