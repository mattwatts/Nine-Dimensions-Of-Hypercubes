"""
UNE DEEP RESEARCH PROTOCOL v2.2 - SOVEREIGN CODE
SCRIPT: logic_garden_v68_race.py
MODE:   Nursery (Combat Palette)
TARGET: Active Protection System (Dual Tank Race)
STYLE:  "The Hard Kill: Twin Motors" | 20s | High Tempo | 4K Ready

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
THREAT_RED = "#FF0000"
THREAT_TRAIL = "#880000"
BARREL_GREY = "#606060"
INTERCEPT_WHITE = "#FFFFFF"

# TEAMS
TEAM_A_COLOR = "#00FFFF" # Cyan (Alpha)
TEAM_B_COLOR = "#FFA500" # Orange (Bravo)
TEAM_A_TRACK = "#004444"
TEAM_B_TRACK = "#442200"

# --- 2. CONFIGURATION ---
FPS = 30
DURATION = 20
TOTAL_FRAMES = FPS * DURATION

class Tank:
    def __init__(self, start_angle, color, track_color, label):
        self.angle = start_angle # Orbital angle
        self.radius = 8.0
        self.color = color
        self.track_color = track_color
        self.label = label
        
        # Physics
        self.pos = np.array([self.radius * np.cos(self.angle), self.radius * np.sin(self.angle)])
        self.heading = self.angle + np.pi/2 # Tangent
        self.speed = 0.35 # Base speed
        
        # Turret
        self.turret_angle = self.heading
        self.barrel_len = 1.8
        
        # Tracks
        self.tracks_l = []
        self.tracks_r = []
        
        # Stats
        self.kills = 0
        self.cm_ready = True
        self.cooldown = 0
    
    def update(self, frame_idx):
        # 1. ORBITAL MOTION
        # Add slight speed variation for "racing" feel
        noise = np.sin(frame_idx * 0.1 + (0 if self.label=="A" else 3)) * 0.02
        current_speed = self.speed + noise
        
        self.angle += current_speed / self.radius # v = w*r -> w = v/r
        
        # Update Pos
        self.pos = np.array([self.radius * np.cos(self.angle), self.radius * np.sin(self.angle)])
        
        # Update Heading (Drift Style)
        # Perfect tangent is angle + pi/2. 
        # Drift means we oversteer or understeer. 
        # Let's point slightly inward (oversteer drift)
        self.heading = self.angle + np.pi/2 + 0.3 # 0.3 rad drift angle
        
        # 2. TRACK GENERATION
        c, s = np.cos(self.heading), np.sin(self.heading)
        off_l = np.array([-1.0, 0.6])
        off_r = np.array([-1.0, -0.6])
        
        gl_l = np.array([off_l[0]*c - off_l[1]*s, off_l[0]*s + off_l[1]*c]) + self.pos
        gl_r = np.array([off_r[0]*c - off_r[1]*s, off_r[0]*s + off_r[1]*c]) + self.pos
        
        self.tracks_l.append(gl_l)
        self.tracks_r.append(gl_r)
        
        if len(self.tracks_l) > 150:
            self.tracks_l.pop(0)
            self.tracks_r.pop(0)

class RaceSim:
    def __init__(self):
        # TANKS
        self.tanks = [
            Tank(0.0, TEAM_A_COLOR, TEAM_A_TRACK, "ALPHA"),
            Tank(np.pi, TEAM_B_COLOR, TEAM_B_TRACK, "BRAVO")
        ]
        
        # COMBAT OBJECTS
        self.threats = [] 
        self.countermeasures = [] 
        self.debris = [] 
        
        # SPAWN LOGIC
        self.global_cooldown = 20
        self.shots_planned = 12
        self.shots_generated = 0
        
        self.radar_angle = 0.0

    def spawn_threat(self):
        # Spawn outside, aiming at a specific tank
        target_tank = self.tanks[np.random.randint(0, 2)]
        
        # Spawn Point: Random edge
        spawn_angle = np.random.uniform(0, 2*np.pi)
        dist = 16.0
        pos = np.array([dist * np.cos(spawn_angle), dist * np.sin(spawn_angle)])
        
        # Aim roughly at tank's FUTURE position (lead)
        # Tank moves ~0.35 rad/frame. 
        # Distance ~ 10 units. Speed proj ~ 0.5. Time ~ 20 frames.
        # Future angle ~ current + 20 * (0.35/8) ~ +0.8 rads
        
        future_angle = target_tank.angle + 0.8
        future_pos = np.array([8.0 * np.cos(future_angle), 8.0 * np.sin(future_angle)])
        
        to_target = future_pos - pos
        aim_angle = np.arctan2(to_target[1], to_target[0])
        error = np.random.uniform(-0.1, 0.1)
        
        speed = 0.6 # Fast
        vel = np.array([speed * np.cos(aim_angle+error), speed * np.sin(aim_angle+error)])
        
        self.threats.append({
            "pos": pos,
            "vel": vel,
            "target_tank": target_tank,
            "id": np.random.randint(1000, 9999), 
            "locked": False,
            "intercepted": False,
            "cm_fired": False
        })

    def fire_cm(self, tank, threat):
        # Solution logic
        rel_pos = threat["pos"] - tank.pos
        dist = np.linalg.norm(rel_pos)
        speed_t = np.linalg.norm(threat["vel"])
        speed_cm = 1.0
        
        t_est = dist / (speed_cm + speed_t)
        intercept_pt = threat["pos"] + threat["vel"] * t_est
        
        aim_vec = intercept_pt - tank.pos
        aim_norm = aim_vec / np.linalg.norm(aim_vec)
        
        vel_cm = aim_norm * speed_cm
        
        # Spawn at muzzle
        start_pos = tank.pos + np.array([
            np.cos(tank.turret_angle) * tank.barrel_len,
            np.sin(tank.turret_angle) * tank.barrel_len
        ])
        
        self.countermeasures.append({
            "pos": start_pos,
            "vel": vel_cm,
            "target_tank": tank, # Owner
            "target_id": threat["id"],
            "life": 40
        })

    def update(self, frame_idx):
        # 1. UPDATE TANKS
        for t in self.tanks:
            t.update(frame_idx)
            
            # Turret Logic (Targeting)
            # Find threats directed at me or close to me
            my_threats = [th for th in self.threats if not th["intercepted"]]
            # Sort by proximity
            my_threats.sort(key=lambda x: np.sum((x["pos"] - t.pos)**2))
            
            target_angle = t.heading # Default forward
            
            if my_threats:
                target = my_threats[0]
                # Aim at predicted
                dist = np.linalg.norm(target["pos"] - t.pos)
                t_est = dist / (1.0 + 0.6)
                pred_pt = target["pos"] + target["vel"] * t_est
                aim = pred_pt - t.pos
                target_angle = np.arctan2(aim[1], aim[0])
            
            # Slew
            diff = target_angle - t.turret_angle
            diff = (diff + np.pi) % (2*np.pi) - np.pi
            slew = 0.4
            if abs(diff) < slew:
                t.turret_angle = target_angle
            else:
                t.turret_angle += np.sign(diff) * slew

        # 2. SPAWN THREATS
        if self.shots_generated < self.shots_planned:
            if self.global_cooldown <= 0:
                self.spawn_threat()
                self.shots_generated += 1
                self.global_cooldown = np.random.randint(30, 60)
            else:
                self.global_cooldown -= 1
                
        # 3. THREAT UPDATE
        for t in self.threats:
            if t["intercepted"]: continue
            t["pos"] += t["vel"]
            
            # Check distance to BOTH tanks (Cooperative defense? No, Selfish)
            # Only the targeted tank fires? Or whoever is closest?
            # Let's say whoever has lock fires.
            
            for tank in self.tanks:
                dist_sq = np.sum((t["pos"] - tank.pos)**2)
                
                if dist_sq < 100.0: # Radar
                    t["locked"] = True
                    
                    # Fire Condition
                    # Check angle alignment
                    aim_vec = t["pos"] - tank.pos # Raw vector approx
                    req_angle = np.arctan2(aim_vec[1], aim_vec[0]) # Approx
                    diff = abs((req_angle - tank.turret_angle + np.pi) % (2*np.pi) - np.pi)
                    
                    if not t["cm_fired"] and diff < 0.2:
                        self.fire_cm(tank, t)
                        t["cm_fired"] = True # Only one intercepts per threat
                        break # Stop checking other tanks

            if np.linalg.norm(t["pos"]) > 25: # Despawn if missed
                t["intercepted"] = True

        # 4. CM UPDATE
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
                    cm["target_tank"].kills += 1
                    self.spawn_debris(cm["pos"], target["vel"], cm["vel"], cm["target_tank"].color)
                    hit = True
            
            if not hit and cm["life"] > 0:
                active_cms.append(cm)
        self.countermeasures = active_cms

        # 5. DEBRIS
        active_debris = []
        for p in self.debris:
            p["pos"] += p["vel"]
            p["vel"] *= 0.85
            p["life"] -= 0.05
            if p["life"] > 0:
                active_debris.append(p)
        self.debris = active_debris
        
        self.threats = [t for t in self.threats if not t["intercepted"]]

    def spawn_debris(self, pos, t_vel, cm_vel, color):
        for _ in range(12):
            vel = t_vel * 0.2 + np.random.normal(0, 0.4, 2)
            self.debris.append({"pos": pos.copy(), "vel": vel, "life": 1.0, "color": color, "size": 12})
        for _ in range(6):
            vel = cm_vel * 0.2 + np.random.normal(0, 0.3, 2)
            self.debris.append({"pos": pos.copy(), "vel": vel, "life": 1.0, "color": THREAT_RED, "size": 8})

    def render(self, frame_idx, ax):
        limit = 12
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_facecolor(BG_VOID)
        
        # 1. TRACKS
        for tank in self.tanks:
            if len(tank.tracks_l) > 1:
                lx, ly = zip(*tank.tracks_l)
                rx, ry = zip(*tank.tracks_r)
                ax.plot(lx, ly, color=tank.track_color, linestyle="-", linewidth=2, alpha=0.8)
                ax.plot(rx, ry, color=tank.track_color, linestyle="-", linewidth=2, alpha=0.8)
        
        # 2. RADAR RINGS
        ax.add_artist(plt.Circle((0,0), 6, color=RADAR_SWEEP, alpha=0.03))
        ax.add_artist(plt.Circle((0,0), 10, color=RADAR_SWEEP, alpha=0.03))
        
        # 3. TANKS
        for tank in self.tanks:
            # Chassis
            corners_local = [np.array([1.0, 0.7]), np.array([-1.0, 0.7]), np.array([-1.0, -0.7]), np.array([1.0, -0.7])]
            c, s = np.cos(tank.heading), np.sin(tank.heading)
            corners_global = []
            for pt in corners_local:
                x_rot = pt[0]*c - pt[1]*s
                y_rot = pt[0]*s + pt[1]*c
                corners_global.append(tank.pos + np.array([x_rot, y_rot]))
            
            poly = Polygon(corners_global, closed=True, facecolor="#202020", edgecolor=tank.color, linewidth=1.5, zorder=5)
            ax.add_artist(poly)
            
            # Turret
            bx = tank.pos[0] + np.cos(tank.turret_angle) * tank.barrel_len
            by = tank.pos[1] + np.sin(tank.turret_angle) * tank.barrel_len
            ax.plot([tank.pos[0], bx], [tank.pos[1], by], color=BARREL_GREY, linewidth=4, zorder=6)
            ax.add_artist(plt.Circle((tank.pos[0], tank.pos[1]), 0.65, color="#454545", zorder=7))
        
        # 4. THREATS
        for t in self.threats:
            start_x = t["pos"][0] - t["vel"][0]*3
            start_y = t["pos"][1] - t["vel"][1]*3
            ax.plot([start_x, t["pos"][0]], [start_y, t["pos"][1]], 
                    color=THREAT_TRAIL, linewidth=1.5, alpha=0.8)
            ax.scatter(t["pos"][0], t["pos"][1], color=THREAT_RED, s=40, marker='v', zorder=10)
            
            if t["locked"]:
                ax.plot([t["target_tank"].pos[0], t["pos"][0]], [t["target_tank"].pos[1], t["pos"][1]], 
                        color=t["target_tank"].color, alpha=0.2, linestyle="--", linewidth=0.5)

        # 5. CMs
        for cm in self.countermeasures:
            ax.scatter(cm["pos"][0], cm["pos"][1], color=cm["target_tank"].color, s=25, marker='o', zorder=12)

        # 6. DEBRIS
        for d in self.debris:
            alpha_val = np.clip(d["life"], 0.0, 1.0)
            ax.scatter(d["pos"][0], d["pos"][1], color=d["color"], s=d["size"]*alpha_val, alpha=alpha_val, zorder=8)

        # 7. HUD
        fig.text(0.5, 0.92, "LOGIC GARDEN 68: THE HARD KILL [TWIN MOTORS]", color="white", ha='center', fontsize=16, fontweight='bold', fontfamily='monospace')
        
        # Dual Stats
        t1, t2 = self.tanks
        stat_a = f"ALPHA: {t1.kills}"
        stat_b = f"BRAVO: {t2.kills}"
        
        fig.text(0.3, 0.05, stat_a, color=TEAM_A_COLOR, ha='center', fontsize=14, fontweight='bold', fontfamily='monospace',
                 bbox=dict(facecolor='black', edgecolor=TEAM_A_COLOR, pad=5, alpha=0.8))
        fig.text(0.7, 0.05, stat_b, color=TEAM_B_COLOR, ha='center', fontsize=14, fontweight='bold', fontfamily='monospace',
                 bbox=dict(facecolor='black', edgecolor=TEAM_B_COLOR, pad=5, alpha=0.8))
        
        # Save
        out_dir = "logic_garden_race_frames"
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"race_{frame_idx:04d}.png")
        plt.savefig(filename, facecolor=BG_VOID)
        plt.close()

# --- 3. EXECUTION ---
if __name__ == "__main__":
    print(f"[NURSERY] Starting Engines...")
    
    sim = RaceSim()
    
    for i in range(TOTAL_FRAMES):
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        
        sim.update(i)
        sim.render(i, ax)
        plt.close()
        
        if i % 60 == 0:
            print(f"Frame {i}/{TOTAL_FRAMES}")
