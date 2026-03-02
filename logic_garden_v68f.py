"""
UNE DEEP RESEARCH PROTOCOL v2.2 - SOVEREIGN CODE
SCRIPT: logic_garden_v68_war.py
MODE:   Nursery (Combat Palette)
TARGET: Active Protection System (10-Tank Deathmatch)
STYLE:  "The Hard Kill: Total War" | 30s | 10 Tanks | 4K Ready

AUTHOR: Matt Watts / Assistant Protocol
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import os

# --- 1. THE COMBAT PALETTE ---
BG_VOID = "#050505"
INTERCEPT_WHITE = "#FFFFFF"

# 10 UNIQUE FACTION COLORS
FACTIONS = [
    "#00FFFF", # Cyan
    "#FF00FF", # Magenta
    "#FFFF00", # Yellow
    "#00FF00", # Lime
    "#FFA500", # Orange
    "#FF1493", # Deep Pink
    "#1E90FF", # Dodger Blue
    "#FF0000", # Red
    "#9D00FF", # Electric Purple
    "#FFFFFF"  # White (Ghost)
]

# --- 2. CONFIGURATION ---
FPS = 30
DURATION = 30
TOTAL_FRAMES = FPS * DURATION

class WarTank:
    def __init__(self, uid, start_pos, heading, color):
        self.uid = uid
        self.color = color
        self.alive = True
        self.hp = 100.0
        
        self.pos = np.array(start_pos, dtype=float)
        self.vel = np.array([0.0, 0.0])
        self.heading = heading
        
        self.turret_angle = heading
        self.barrel_len = 1.6
        self.gun_cooldown = np.random.randint(0, 50) # Staggered starts
        self.aps_cooldown = 0
        self.aps_charges = 3
        
        self.tracks_l = []
        self.tracks_r = []

    def update_ai(self, all_tanks, projectiles):
        if not self.alive: return None, None
        
        # 1. TARGETING
        target = None
        min_dist = 999.0
        
        for other in all_tanks:
            if other.uid == self.uid or not other.alive: continue
            d = np.linalg.norm(other.pos - self.pos)
            if d < min_dist:
                min_dist = d
                target = other
        
        if target:
            # 2. MOVEMENT (Chaos Dodge)
            to_target = target.pos - self.pos
            dist = np.linalg.norm(to_target)
            norm = to_target / (dist + 0.01)
            
            # Strafe
            strafe = np.array([-norm[1], norm[0]])
            # If chaotic (many neighbors), move away from cluster center?
            # Simple heuristic: Keep moving.
            
            desired_vel = strafe * 0.7 + norm * (0.5 if dist > 8 else -0.5)
            # Add noise
            desired_vel += np.random.uniform(-0.15, 0.15, 2)
            
            desired_heading = np.arctan2(desired_vel[1], desired_vel[0])
            diff = (desired_heading - self.heading + np.pi) % (2*np.pi) - np.pi
            self.heading += np.clip(diff, -0.12, 0.12)
            
            thrust = 0.09
            self.vel += np.array([np.cos(self.heading), np.sin(self.heading)]) * thrust
            
            # 3. TURRET & FIRE
            lead = target.pos + target.vel * (dist / 1.1)
            aim = lead - self.pos
            req_turret = np.arctan2(aim[1], aim[0])
            t_diff = (req_turret - self.turret_angle + np.pi) % (2*np.pi) - np.pi
            self.turret_angle += np.clip(t_diff, -0.3, 0.3) # Fast turret
            
            if abs(t_diff) < 0.2 and self.gun_cooldown <= 0:
                self.gun_cooldown = np.random.randint(20, 50)
                return "FIRE", lead
                
        # Friction & Walls
        self.vel *= 0.94
        self.pos += self.vel
        
        lim = 12.0
        if abs(self.pos[0]) > lim: self.vel[0] *= -0.8; self.pos[0] = np.sign(self.pos[0])*lim
        if abs(self.pos[1]) > lim: self.vel[1] *= -0.8; self.pos[1] = np.sign(self.pos[1])*lim
        
        # APS RECHARGE
        if self.aps_charges < 3:
            self.aps_cooldown += 1
            if self.aps_cooldown > 45: 
                self.aps_charges += 1
                self.aps_cooldown = 0
                
        # DEFENSE
        for p in projectiles:
            if p["owner"] == self.uid: continue
            if p["type"] != "SHELL": continue
            
            if np.linalg.norm(p["pos"] - self.pos) < 6.0:
                rel = p["vel"] - self.vel
                if np.dot(rel, self.pos - p["pos"]) > 0: # Closing
                    if self.aps_charges > 0:
                        return "APS", p
                        
        return None, None

    def update_tracks(self):
        c, s = np.cos(self.heading), np.sin(self.heading)
        off_l = np.array([-0.9, 0.6])
        off_r = np.array([-0.9, -0.6])
        
        gl_l = np.array([off_l[0]*c - off_l[1]*s, off_l[0]*s + off_l[1]*c]) + self.pos
        gl_r = np.array([off_r[0]*c - off_r[1]*s, off_r[0]*s + off_r[1]*c]) + self.pos
        
        self.tracks_l.append(gl_l); self.tracks_r.append(gl_r)
        if len(self.tracks_l) > 50: self.tracks_l.pop(0); self.tracks_r.pop(0)

class WarSim:
    def __init__(self):
        self.tanks = []
        radius = 11.0
        
        # Spawn 10 Tanks in a circle
        for i in range(10):
            angle = (2 * np.pi / 10) * i
            pos = [radius * np.cos(angle), radius * np.sin(angle)]
            # Face inward (angle + pi)
            heading = angle + np.pi
            # Random jitter to heading so they don't all shoot center immediately
            heading += np.random.uniform(-0.5, 0.5)
            
            self.tanks.append(WarTank(i, pos, heading, FACTIONS[i]))
            
        self.projectiles = []
        self.debris = []
        self.alive_count = 10

    def spawn_shell(self, tank):
        speed = 1.2
        tip = tank.pos + np.array([np.cos(tank.turret_angle), np.sin(tank.turret_angle)]) * tank.barrel_len
        angle = tank.turret_angle + np.random.uniform(-0.04, 0.04)
        vel = np.array([np.cos(angle)*speed, np.sin(angle)*speed]) + tank.vel * 0.2
        
        self.projectiles.append({
            "pos": tip, "vel": vel, "owner": tank.uid, "type": "SHELL",
            "life": 50, "color": INTERCEPT_WHITE
        })
        self.spawn_debris(tip, vel*0.3, tank.color, 4)

    def fire_aps(self, tank, target):
        tank.aps_charges -= 1
        dist = np.linalg.norm(target["pos"] - tank.pos)
        int_pt = target["pos"] + target["vel"] * (dist/2.5)
        vel = (int_pt - tank.pos)
        vel = vel / np.linalg.norm(vel) * 1.6
        
        self.projectiles.append({
            "pos": tank.pos.copy(), "vel": vel, "owner": tank.uid, "type": "APS",
            "target_id": id(target), "life": 12, "color": tank.color
        })

    def spawn_debris(self, pos, vel, color, count=10):
        for _ in range(count):
            v = vel * 0.5 + np.random.normal(0, 0.4, 2)
            self.debris.append({
                "pos": pos.copy(), "vel": v, "life": 1.0, "color": color, "size": np.random.uniform(4, 10)
            })

    def update(self, frame_idx):
        alive = 0
        for t in self.tanks:
            if t.alive:
                alive += 1
                t.gun_cooldown -= 1
                t.update_tracks()
                act, data = t.update_ai(self.tanks, self.projectiles)
                if act == "FIRE": self.spawn_shell(t)
                elif act == "APS": self.fire_aps(t, data)
            else:
                t.vel *= 0.9; t.pos += t.vel
        self.alive_count = alive
        
        active_p = []
        for p in self.projectiles:
            p["pos"] += p["vel"]; p["life"] -= 1
            hit = False
            
            if p["type"] == "SHELL":
                for t in self.tanks:
                    if t.uid != p["owner"] and t.alive:
                        if np.linalg.norm(p["pos"] - t.pos) < 1.05:
                            t.hp -= 34; hit = True # 3 hits
                            self.spawn_debris(p["pos"], p["vel"]*0.2, t.color, 12)
                            if t.hp <= 0:
                                t.alive = False
                                self.spawn_debris(t.pos, t.vel, t.color, 40)
                            break
            elif p["type"] == "APS":
                for s in self.projectiles:
                    if s["type"] == "SHELL" and s["owner"] != p["owner"]:
                        if np.linalg.norm(p["pos"] - s["pos"]) < 1.0:
                            s["life"] = 0; hit = True
                            self.spawn_debris(s["pos"], s["vel"]*0.1, INTERCEPT_WHITE, 6)
                            break
            if not hit and p["life"] > 0: active_p.append(p)
        self.projectiles = active_p
        
        active_d = []
        for d in self.debris:
            d["pos"] += d["vel"]; d["vel"] *= 0.88; d["life"] -= 0.05
            if d["life"] > 0: active_d.append(d)
        self.debris = active_d

    def draw_tank(self, ax, t):
        col = t.color if t.alive else "#333333"
        # Chassis
        c, s = np.cos(t.heading), np.sin(t.heading)
        pts = [np.array([1, 0.7]), np.array([-1, 0.7]), np.array([-1, -0.7]), np.array([1, -0.7])]
        g_pts = [t.pos + np.array([p[0]*c - p[1]*s, p[0]*s + p[1]*c]) for p in pts]
        
        ax.add_artist(Polygon(g_pts, closed=True, facecolor="#151515", edgecolor=col, linewidth=1.5, zorder=5))
        
        # Turret
        bx, by = t.pos[0] + np.cos(t.turret_angle)*t.barrel_len, t.pos[1] + np.sin(t.turret_angle)*t.barrel_len
        ax.plot([t.pos[0], bx], [t.pos[1], by], color=col, linewidth=2.5, zorder=6)
        ax.add_artist(Circle((t.pos[0], t.pos[1]), 0.6, color="#444444", zorder=7))
        
        # Status
        if t.alive:
            # HP DOTS
            hp_rem = int(t.hp / 34) + 1
            for i in range(hp_rem):
                ax.scatter(t.pos[0] - 0.5 + i*0.5, t.pos[1], color=col, s=8, zorder=8)

    def render(self, frame_idx, ax):
        lim = 13
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal'); ax.set_axis_off(); ax.set_facecolor(BG_VOID)
        
        # Tracks
        for t in self.tanks:
            if len(t.tracks_l) > 1:
                col = t.color if t.alive else "#222222"
                lx, ly = zip(*t.tracks_l); rx, ry = zip(*t.tracks_r)
                ax.plot(lx, ly, color=col, lw=1, alpha=0.3)
                ax.plot(rx, ry, color=col, lw=1, alpha=0.3)
        
        # Objects
        for t in self.tanks: self.draw_tank(ax, t)
        
        for p in self.projectiles:
            ax.scatter(p["pos"][0], p["pos"][1], color=p["color"], s=30 if p["type"]=="SHELL" else 10, zorder=10)
            
        for d in self.debris:
            ax.scatter(d["pos"][0], d["pos"][1], color=d["color"], s=d["size"], alpha=np.clip(d["life"],0,1), zorder=8)
            
        # HUD
        fig.text(0.5, 0.92, "LOGIC GARDEN 68: TOTAL WAR", color="white", ha='center', fontsize=16, fontweight='bold', fontfamily='monospace')
        
        if self.alive_count > 1:
            st, col = f"ALIVE: {self.alive_count}/10", INTERCEPT_WHITE
        elif self.alive_count == 1:
            win = next((t for t in self.tanks if t.alive), None)
            st, col = "CHAMPION", win.color
        else:
            st, col = "SQUAD WIPED", "#888888"
            
        ax.text(0, -12, st, color=col, ha='center', fontsize=14, fontweight='bold', fontfamily='monospace',
                bbox=dict(facecolor='black', edgecolor=col, pad=6, alpha=0.9))

        # I/O
        out_dir = "logic_garden_war_frames"
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"war_{frame_idx:04d}.png"), facecolor=BG_VOID)
        plt.close()

# --- 3. EXECUTION ---
if __name__ == "__main__":
    print(f"[NURSERY] Dropping 10 Units...")
    sim = WarSim()
    for i in range(TOTAL_FRAMES):
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        sim.update(i)
        sim.render(i, ax)
        if i % 60 == 0: print(f"Frame {i}/{TOTAL_FRAMES}")
