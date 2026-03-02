"""
UNE DEEP RESEARCH PROTOCOL v2.2 - SOVEREIGN CODE
SCRIPT: logic_garden_v68_drift.py
MODE:   Nursery (Combat Palette)
TARGET: Active Protection System (High Velocity Drift)
STYLE:  "The Hard Kill: Drift" | 20s | High Tempo | 4K Ready

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

class APSDriftSim:
    def __init__(self):
        # PHYSICS STATE
        self.pos = np.array([0.0, 0.0])
        self.vel = np.array([0.0, 0.0])
        self.heading = 0.0 # Radians
        self.target_heading = 0.0
        
        # NAVIGATION AI
        self.nav_timer = 0
        self.waypoint = np.array([0.0, 0.0])
        
        # TURRET STATE
        self.turret_angle = 0.0 
        self.barrel_len = 1.8 
        
        # COMBAT LOGIC
        self.threats = [] 
        self.countermeasures = [] 
        self.debris = [] 
        self.tracks_l = [] 
        self.tracks_r = [] 
        
        self.kills = 0
        self.shots_planned = 8 # More aggro
        self.shots_generated = 0
        self.cooldown = 30
        
        self.radar_angle = 0.0
        
        # Kickstart motion
        self.pick_new_waypoint()

    def pick_new_waypoint(self):
        # Pick a random point within bounds (-9, 9)
        self.waypoint = np.random.uniform(-9.0, 9.0, 2)
        self.nav_timer = np.random.randint(40, 80) # Change mind frequently

    def update_physics(self):
        # 1. NAVIGATION
        self.nav_timer -= 1
        if self.nav_timer <= 0:
            self.pick_new_waypoint()
            
        # Steer towards waypoint
        to_wp = self.waypoint - self.pos
        req_heading = np.arctan2(to_wp[1], to_wp[0])
        
        # Smooth Turn
        diff = req_heading - self.heading
        diff = (diff + np.pi) % (2*np.pi) - np.pi
        
        turn_rate = 0.15 # Fast turn
        if abs(diff) < turn_rate:
            self.heading = req_heading
        else:
            self.heading += np.sign(diff) * turn_rate
            
        # 2. PROPULSION (Drift Physics)
        # Apply force in direction of HEADING
        thrust = 0.04 # Engine power
        accel = np.array([np.cos(self.heading), np.sin(self.heading)]) * thrust
        
        # Add to velocity
        self.vel += accel
        
        # Friction/Drag
        self.vel *= 0.92 # Slide factor (higher = more slide)
        
        # Update Position
        self.pos += self.vel
        
        # 3. BOUNDARY CONTAINMENT (Bounce)
        limit = 11.0
        if abs(self.pos[0]) > limit:
            self.pos[0] = np.sign(self.pos[0]) * limit
            self.vel[0] *= -0.5 # Bounce energy loss
            self.pick_new_waypoint() # Panic steer away
            
        if abs(self.pos[1]) > limit:
            self.pos[1] = np.sign(self.pos[1]) * limit
            self.vel[1] *= -0.5
            self.pick_new_waypoint()

        # 4. TRACK GENERATION (Drift Marks)
        # Chassis corners rotating with heading
        c, s = np.cos(self.heading), np.sin(self.heading)
        
        # Local offsets for rear sprockets
        # Tank Center origin. Rear is x=-1.0. Spread y=0.6.
        off_l = np.array([-1.0, 0.6])
        off_r = np.array([-1.0, -0.6])
        
        gl_l = np.array([off_l[0]*c - off_l[1]*s, off_l[0]*s + off_l[1]*c]) + self.pos
        gl_r = np.array([off_r[0]*c - off_r[1]*s, off_r[0]*s + off_r[1]*c]) + self.pos
        
        self.tracks_l.append(gl_l)
        self.tracks_r.append(gl_r)
        
        # Fade tracks (Perf optimization)
        max_tracks = 300
        if len(self.tracks_l) > max_tracks:
            self.tracks_l.pop(0)
            self.tracks_r.pop(0)

    def spawn_threat(self):
        # Spawn FAR out because we are moving fast
        angle = np.random.uniform(0, 2*np.pi)
        dist = 18.0 
        
        offset = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        pos = self.pos + offset
        
        # Aim at Predicted Position? Or just current?
        # Let's aim at current, requiring APS to work hard
        speed = np.random.uniform(0.6, 0.9) # FAST incoming
        
        to_tank = self.pos - pos
        target_angle = np.arctan2(to_tank[1], to_tank[0])
        aim_angle = target_angle + np.random.uniform(-0.05, 0.05)
        
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
        # Intercept math must account for tank velocity now?
        # Actually firing from a moving platform adds velocity to projectile.
        # Let's assume the APS compensates.
        # Relative Position
        rel_pos = threat["pos"] - self.pos
        
        dist = np.linalg.norm(rel_pos)
        speed_t = np.linalg.norm(threat["vel"])
        speed_cm = 1.2 # Faster CM
        
        t_est = dist / (speed_cm + speed_t)
        intercept_pt = threat["pos"] + threat["vel"] * t_est
        
        aim_vec = intercept_pt - self.pos
        return aim_vec, intercept_pt

    def update_turret(self):
        active = [t for t in self.threats if not t["intercepted"]]
        
        if active:
            active.sort(key=lambda t: np.sum((t["pos"] - self.pos)**2))
            target = active[0]
            aim_vec, _ = self.get_intercept_solution(target)
            target_angle = np.arctan2(aim_vec[1], aim_vec[0])
        else:
            # Look Forward if idle
            target_angle = self.heading 
            
        diff = target_angle - self.turret_angle
        diff = (diff + np.pi) % (2*np.pi) - np.pi
        
        slew_rate = 0.5 # Hyper fast slew
        if abs(diff) < slew_rate:
            self.turret_angle = target_angle
        else:
            self.turret_angle += np.sign(diff) * slew_rate

    def fire_cm(self, threat):
        aim_vec, _ = self.get_intercept_solution(threat)
        aim_norm = aim_vec / np.linalg.norm(aim_vec)
        speed_cm = 1.2
        vel_cm = aim_norm * speed_cm
        
        # Muzzle Velocity = Physics Velocity + Muzzle Boost?
        # Let's add platform velocity for realism
        total_vel = vel_cm # + self.vel * 0.5 (optional)
        
        start_pos = self.pos + np.array([
            np.cos(self.turret_angle) * self.barrel_len,
            np.sin(self.turret_angle) * self.barrel_len
        ])
        
        self.countermeasures.append({
            "pos": start_pos,
            "vel": total_vel,
            "target_id": threat["id"],
            "life": 40
        })

    def update(self, frame_idx):
        # 1. PHYSICS
        self.update_physics()
        self.radar_angle -= 0.3
        self.update_turret()
        
        # 2. SPAWNER
        if self.shots_generated < self.shots_planned:
            if self.cooldown <= 0:
                self.spawn_threat()
                self.shots_generated += 1
                self.cooldown = np.random.randint(30, 90) # Faster pace
            else:
                self.cooldown -= 1
                
        # 3. THREATS
        for t in self.threats:
            if t["intercepted"]: continue
            t["pos"] += t["vel"]
            
            dist_sq = np.sum((t["pos"] - self.pos)**2)
            
            if dist_sq < 144.0: # 12m radar range
                t["locked"] = True
                
                # Check alignment
                aim_vec, _ = self.get_intercept_solution(t)
                req_angle = np.arctan2(aim_vec[1], aim_vec[0])
                diff = abs((req_angle - self.turret_angle + np.pi) % (2*np.pi) - np.pi)
                
                # Fire if aligned OR PANIC
                if not t["cm_fired"] and (diff < 0.2 or dist_sq < 30):
                    self.fire_cm(t)
                    t["cm_fired"] = True
            
            if dist_sq < 0.5: # Hit chassis
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
                if d < 1.2: 
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
            p["vel"] *= 0.85 # Drag
            p["life"] -= 0.05
            if p["life"] > 0:
                active_debris.append(p)
        self.debris = active_debris
        
        self.threats = [t for t in self.threats if not t["intercepted"]]

    def spawn_debris(self, pos, t_vel, cm_vel):
        for _ in range(15):
            vel = t_vel * 0.2 + np.random.normal(0, 0.4, 2)
            self.debris.append({"pos": pos.copy(), "vel": vel, "life": 1.0, "color": INTERCEPT_WHITE, "size": 12})
        for _ in range(8):
            vel = cm_vel * 0.2 + np.random.normal(0, 0.3, 2)
            self.debris.append({"pos": pos.copy(), "vel": vel, "life": 1.0, "color": THREAT_RED, "size": 8})

    def render(self, frame_idx, ax):
        limit = 12
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_facecolor(BG_VOID)
        
        # 1. TRACKS
        if len(self.tracks_l) > 1:
            lx, ly = zip(*self.tracks_l)
            rx, ry = zip(*self.tracks_r)
            # Use 'Track Trail' color
            ax.plot(lx, ly, color=TRACK_TRAIL, linestyle="-", linewidth=2, alpha=0.6)
            ax.plot(rx, ry, color=TRACK_TRAIL, linestyle="-", linewidth=2, alpha=0.6)
        
        # 2. RADAR
        ax.add_artist(plt.Circle((self.pos[0], self.pos[1]), 10, color=RADAR_SWEEP, alpha=0.05))
        
        # 3. TANK CHASSIS
        # Rotated Rectangle
        corners_local = [
            np.array([ 1.0,  0.7]),
            np.array([-1.0,  0.7]),
            np.array([-1.0, -0.7]),
            np.array([ 1.0, -0.7])
        ]
        c, s = np.cos(self.heading), np.sin(self.heading)
        corners_global = []
        for pt in corners_local:
            x_rot = pt[0]*c - pt[1]*s
            y_rot = pt[0]*s + pt[1]*c
            corners_global.append(self.pos + np.array([x_rot, y_rot]))
            
        poly = Polygon(corners_global, closed=True, facecolor="#252525", edgecolor="#404040", zorder=5)
        ax.add_artist(poly)
        
        # 4. TURRET
        bx = self.pos[0] + np.cos(self.turret_angle) * self.barrel_len
        by = self.pos[1] + np.sin(self.turret_angle) * self.barrel_len
        ax.plot([self.pos[0], bx], [self.pos[1], by], color=BARREL_GREY, linewidth=4, zorder=6)
        ax.add_artist(plt.Circle((self.pos[0], self.pos[1]), 0.65, color="#454545", zorder=7))
        
        # 5. THREATS
        for t in self.threats:
            start_x = t["pos"][0] - t["vel"][0]*3
            start_y = t["pos"][1] - t["vel"][1]*3
            ax.plot([start_x, t["pos"][0]], [start_y, t["pos"][1]], 
                    color=THREAT_TRAIL, linewidth=1.5, alpha=0.8)
            ax.scatter(t["pos"][0], t["pos"][1], color=THREAT_RED, s=40, marker='v', zorder=10)
            
            if t["locked"]:
                ax.plot([self.pos[0], t["pos"][0]], [self.pos[1], t["pos"][1]], 
                        color=CALC_GOLD, alpha=0.15, linestyle="--", linewidth=0.5)

        # 6. CMs
        for cm in self.countermeasures:
            ax.scatter(cm["pos"][0], cm["pos"][1], color=INTERCEPT_WHITE, s=30, marker='o', zorder=12)

        # 7. DEBRIS
        for d in self.debris:
            alpha_val = np.clip(d["life"], 0.0, 1.0)
            ax.scatter(d["pos"][0], d["pos"][1], color=d["color"], s=d["size"]*alpha_val, alpha=alpha_val, zorder=8)

        # 8. HUD
        fig.text(0.5, 0.92, "LOGIC GARDEN 68: THE HARD KILL [MADMAN]", color="white", ha='center', fontsize=16, fontweight='bold', fontfamily='monospace')
        
        rem = self.shots_planned - self.shots_generated
        status = f"THREATS: {len(self.threats) + rem}"
        col_stat = THREAT_RED
        if rem == 0 and len(self.threats) == 0:
            status = "SECTOR CLEARED"
            col_stat = INTERCEPT_WHITE

        ax.text(0, -11, f"{status} | KILLS: {self.kills}", color=col_stat, ha='center', fontsize=14, fontweight='bold', fontfamily='monospace',
                bbox=dict(facecolor='black', edgecolor=col_stat, pad=6, alpha=0.9))
        
        # Save
        out_dir = "logic_garden_drift_frames"
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"drift_{frame_idx:04d}.png")
        plt.savefig(filename, facecolor=BG_VOID)
        plt.close()

# --- 3. EXECUTION ---
if __name__ == "__main__":
    print(f"[NURSERY] Disengaging Safety Governors...")
    
    sim = APSDriftSim()
    
    for i in range(TOTAL_FRAMES):
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        
        sim.update(i)
        sim.render(i, ax)
        plt.close()
        
        if i % 60 == 0:
            print(f"Frame {i}/{TOTAL_FRAMES}")
