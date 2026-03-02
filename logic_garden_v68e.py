"""
UNE DEEP RESEARCH PROTOCOL v2.2 - SOVEREIGN CODE
SCRIPT: logic_garden_v68_battle_fixed.py
MODE:   Nursery (Combat Palette)
TARGET: Active Protection System (4-Way Deathmatch)
STYLE:  "The Hard Kill: Battle Royale" | 30s | 4 Tanks | 4K Ready
STATUS: PATCHED (Restored Disk I/O)

AUTHOR: Matt Watts / Assistant Protocol
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
import os

# --- 1. THE COMBAT PALETTE ---
BG_VOID = "#050505"
INTERCEPT_WHITE = "#FFFFFF"

# FACTIONS
COLORS = [
    "#00FFFF", # CYAN
    "#FF00FF", # MAGENTA
    "#FFFF00", # YELLOW
    "#00FF00"  # LIME
]

# --- 2. CONFIGURATION ---
FPS = 30
DURATION = 30
TOTAL_FRAMES = FPS * DURATION

class CombatTank:
    def __init__(self, uid, start_pos, color):
        self.uid = uid
        self.color = color
        self.alive = True
        self.hp = 100.0
        
        # Physics
        self.pos = np.array(start_pos, dtype=float)
        self.vel = np.array([0.0, 0.0])
        self.heading = np.random.uniform(0, 2*np.pi)
        
        # Systems
        self.turret_angle = self.heading
        self.barrel_len = 1.6
        self.gun_cooldown = np.random.randint(0, 30)
        self.aps_cooldown = 0
        self.aps_charges = 3 
        
        # Visuals
        self.tracks_l = []
        self.tracks_r = []

    def update_ai(self, all_tanks, projectiles):
        if not self.alive: return None, None
        
        # 1. ACQUIRE TARGET
        target = None
        min_dist = 999.0
        
        for other in all_tanks:
            if other.uid == self.uid or not other.alive: continue
            d = np.linalg.norm(other.pos - self.pos)
            if d < min_dist:
                min_dist = d
                target = other
        
        # 2. NAVIGATION (Drift & Dodge)
        if target:
            to_target = target.pos - self.pos
            dist = np.linalg.norm(to_target)
            to_target_norm = to_target / (dist + 0.01)
            
            # Strafe
            strafe = np.array([-to_target_norm[1], to_target_norm[0]])
            
            # Approach/Retreat
            approach = 0.5 if dist > 10 else (-0.5 if dist < 5 else 0.0)
            
            desired_vel = strafe * 0.6 + to_target_norm * approach + np.random.uniform(-0.1, 0.1, 2)
            desired_heading = np.arctan2(desired_vel[1], desired_vel[0])
            
            # Turn Chassis
            diff = (desired_heading - self.heading + np.pi) % (2*np.pi) - np.pi
            self.heading += np.clip(diff, -0.1, 0.1)
            
            # Thrust
            self.vel += np.array([np.cos(self.heading), np.sin(self.heading)]) * 0.08
            
            # Turret Aiming (Lead)
            lead_pos = target.pos + target.vel * (dist / 1.1)
            aim_vec = lead_pos - self.pos
            req_turret = np.arctan2(aim_vec[1], aim_vec[0])
            
            t_diff = (req_turret - self.turret_angle + np.pi) % (2*np.pi) - np.pi
            self.turret_angle += np.clip(t_diff, -0.25, 0.25)
            
            # FIRE
            if abs(t_diff) < 0.15 and self.gun_cooldown <= 0:
                self.gun_cooldown = np.random.randint(25, 45)
                return "FIRE", lead_pos
        else:
            # Idle spin
            self.heading += 0.05
        
        # Friction
        self.vel *= 0.94
        self.pos += self.vel
        
        # Wall Boundary
        lim = 11.5
        if abs(self.pos[0]) > lim: self.vel[0] *= -0.8; self.pos[0] = np.sign(self.pos[0])*lim
        if abs(self.pos[1]) > lim: self.vel[1] *= -0.8; self.pos[1] = np.sign(self.pos[1])*lim
        
        # APS RECHARGE
        if self.aps_charges < 3:
            self.aps_cooldown += 1
            if self.aps_cooldown > 40: 
                self.aps_charges += 1
                self.aps_cooldown = 0
                
        # 3. DEFENSE
        for p in projectiles:
            if p["owner"] == self.uid: continue
            if p["type"] != "SHELL": continue
            
            d = np.linalg.norm(p["pos"] - self.pos)
            if d < 5.0: 
                rel_vel = p["vel"] - self.vel
                closing = np.dot(rel_vel, (self.pos - p["pos"])) > 0
                if closing and self.aps_charges > 0:
                    return "APS", p
                    
        return None, None

    def update_tracks(self):
        c, s = np.cos(self.heading), np.sin(self.heading)
        off_l = np.array([-0.9, 0.6])
        off_r = np.array([-0.9, -0.6])
        
        gl_l = np.array([off_l[0]*c - off_l[1]*s, off_l[0]*s + off_l[1]*c]) + self.pos
        gl_r = np.array([off_r[0]*c - off_r[1]*s, off_r[0]*s + off_r[1]*c]) + self.pos
        
        self.tracks_l.append(gl_l)
        self.tracks_r.append(gl_r)
        
        if len(self.tracks_l) > 60:
            self.tracks_l.pop(0)
            self.tracks_r.pop(0)

class BattleSim:
    def __init__(self):
        self.tanks = [
            CombatTank(0, [-8, -8], COLORS[0]),
            CombatTank(1, [8, -8], COLORS[1]),
            CombatTank(2, [8, 8], COLORS[2]),
            CombatTank(3, [-8, 8], COLORS[3])
        ]
        self.projectiles = []
        self.debris = []
        self.alive_count = 4

    def spawn_shell(self, tank):
        speed = 1.1
        tip = tank.pos + np.array([np.cos(tank.turret_angle), np.sin(tank.turret_angle)]) * tank.barrel_len
        angle = tank.turret_angle + np.random.uniform(-0.03, 0.03)
        vel = np.array([np.cos(angle)*speed, np.sin(angle)*speed]) + tank.vel * 0.2
        
        self.projectiles.append({
            "pos": tip, "vel": vel, "owner": tank.uid, "type": "SHELL",
            "life": 60, "color": INTERCEPT_WHITE
        })
        self.spawn_debris(tip, vel*0.3, tank.color, 5)

    def fire_aps(self, tank, target):
        tank.aps_charges -= 1
        dist = np.linalg.norm(target["pos"] - tank.pos)
        int_pt = target["pos"] + target["vel"] * (dist/2.5)
        vel = (int_pt - tank.pos)
        vel = vel / np.linalg.norm(vel) * 1.5
        
        self.projectiles.append({
            "pos": tank.pos.copy(), "vel": vel, "owner": tank.uid, "type": "APS",
            "target_id": id(target), "life": 15, "color": tank.color
        })

    def spawn_debris(self, pos, vel, color, count=10):
        for _ in range(count):
            v = vel * 0.5 + np.random.normal(0, 0.3, 2)
            self.debris.append({
                "pos": pos.copy(), "vel": v, "life": 1.0, "color": color, "size": np.random.uniform(5, 12)
            })

    def update(self, frame_idx):
        alive_c = 0
        for t in self.tanks:
            if t.alive:
                alive_c += 1
                t.gun_cooldown -= 1
                t.update_tracks()
                act, data = t.update_ai(self.tanks, self.projectiles)
                if act == "FIRE": self.spawn_shell(t)
                elif act == "APS": self.fire_aps(t, data)
            else:
                t.vel *= 0.9; t.pos += t.vel
        self.alive_count = alive_c
        
        active_p = []
        for p in self.projectiles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            hit = False
            
            if p["type"] == "SHELL":
                for t in self.tanks:
                    if t.uid != p["owner"] and t.alive:
                        if np.linalg.norm(p["pos"] - t.pos) < 1.0:
                            t.hp -= 35; hit = True
                            self.spawn_debris(p["pos"], p["vel"]*0.2, t.color, 15)
                            if t.hp <= 0: 
                                t.alive = False
                                self.spawn_debris(t.pos, t.vel, t.color, 40)
                            break
            elif p["type"] == "APS":
                for s in self.projectiles:
                    if s["type"] == "SHELL" and s["owner"] != p["owner"]:
                        if np.linalg.norm(p["pos"] - s["pos"]) < 1.0:
                            s["life"] = 0; hit = True
                            self.spawn_debris(s["pos"], s["vel"]*0.1, INTERCEPT_WHITE, 8)
                            break
                            
            if not hit and p["life"] > 0: active_p.append(p)
        self.projectiles = active_p
        
        active_d = []
        for d in self.debris:
            d["pos"] += d["vel"]; d["vel"] *= 0.9; d["life"] -= 0.04
            if d["life"] > 0: active_d.append(d)
        self.debris = active_d

    def draw_tank(self, ax, t, color):
        c, s = np.cos(t.heading), np.sin(t.heading)
        pts = [np.array([1, 0.7]), np.array([-1, 0.7]), np.array([-1, -0.7]), np.array([1, -0.7])]
        g_pts = [t.pos + np.array([p[0]*c - p[1]*s, p[0]*s + p[1]*c]) for p in pts]
        ax.add_artist(Polygon(g_pts, closed=True, facecolor="#202020", edgecolor=color, linewidth=2, zorder=5))
        
        bx, by = t.pos[0] + np.cos(t.turret_angle)*t.barrel_len, t.pos[1] + np.sin(t.turret_angle)*t.barrel_len
        ax.plot([t.pos[0], bx], [t.pos[1], by], color=color, linewidth=3, zorder=6)
        ax.add_artist(Circle((t.pos[0], t.pos[1]), 0.6, color="#444444", zorder=7))
        
        if t.alive:
            for i in range(t.aps_charges):
                ax.scatter(t.pos[0] + 0.5 + i*0.2, t.pos[1] - 1.2, color=INTERCEPT_WHITE, s=10, zorder=8)

    def render(self, frame_idx, ax):
        lim = 13
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal'); ax.set_axis_off(); ax.set_facecolor(BG_VOID)
        
        # Tracks
        for t in self.tanks:
            if len(t.tracks_l) > 1:
                col = t.color if t.alive else "#333333"
                lx, ly = zip(*t.tracks_l); rx, ry = zip(*t.tracks_r)
                ax.plot(lx, ly, color=col, lw=1, alpha=0.3); ax.plot(rx, ry, color=col, lw=1, alpha=0.3)
        
        # Tanks
        for t in self.tanks: 
            self.draw_tank(ax, t, t.color if t.alive else "#333333")
            
        # Projectiles
        for p in self.projectiles:
            ax.scatter(p["pos"][0], p["pos"][1], color=p["color"], s=40 if p["type"]=="SHELL" else 15, zorder=10)
            
        # Debris
        for d in self.debris:
            alp = np.clip(d["life"], 0, 1)
            ax.scatter(d["pos"][0], d["pos"][1], color=d["color"], s=d["size"], alpha=alp, zorder=8)
            
        # HUD
        fig.text(0.5, 0.92, "LOGIC GARDEN 68: THE ARENA", color="white", ha='center', fontsize=16, fontweight='bold', fontfamily='monospace')
        
        if self.alive_count > 1:
            st, col = f"COMBATANTS: {self.alive_count}", INTERCEPT_WHITE
        elif self.alive_count == 1:
            win = next((t for t in self.tanks if t.alive), None)
            st, col = "VICTORY", win.color
        else:
            st, col = "MUTUAL DESTRUCTION", "#888888"
            
        ax.text(0, -12, st, color=col, ha='center', fontsize=14, fontweight='bold', fontfamily='monospace',
                bbox=dict(facecolor='black', edgecolor=col, pad=6, alpha=0.9))

        # --- RESTORED DISK I/O ---
        out_dir = "logic_garden_battle_frames"
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"battle_{frame_idx:04d}.png")
        plt.savefig(filename, facecolor=BG_VOID)
        plt.close()

# --- 3. EXECUTION ---
if __name__ == "__main__":
    print(f"[NURSERY] Initialization: Free For All (Patched)...")
    sim = BattleSim()
    for i in range(TOTAL_FRAMES):
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        sim.update(i)
        sim.render(i, ax)
        if i % 60 == 0: print(f"Frame {i}/{TOTAL_FRAMES}")
