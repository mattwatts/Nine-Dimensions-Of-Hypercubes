"""
UNE DEEP RESEARCH PROTOCOL v2.2 - SOVEREIGN CODE
SCRIPT: logic_garden_v68_titan_patched.py
MODE:   Nursery (Combat Palette)
TARGET: Active Protection System (Boss Battle - Hero Buffed)
STYLE:  "The Titan" | 40s | Hero Wins | 4K Ready
STATUS: PATCHED (Hero APS Online + Damage Tuning)

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
HERO_CYAN = "#00FFFF"
BOSS_RED = "#8B0000"     # Dark Crimson
BOSS_CORE = "#FF4500"    # Orange Core
BOSS_SHELL = "#FFCC00"   # Heavy Shells
INTERCEPT_WHITE = "#FFFFFF" 

# --- 2. CONFIGURATION ---
FPS = 30
DURATION = 40
TOTAL_FRAMES = FPS * DURATION

class HeroTank:
    def __init__(self):
        self.pos = np.array([-10.0, 0.0])
        self.vel = np.array([0.0, 0.0])
        self.heading = 0.0
        self.alive = True
        self.hp = 200.0  # BUFF: 2x Health
        self.max_hp = 200.0
        
        self.turret_angle = 0.0
        self.barrel_len = 1.8
        self.cooldown = 0
        
        # UPGRADE: APS
        self.aps_charge = 150.0 
        
        self.tracks_l = []
        self.tracks_r = []

    def update(self, boss, projectiles):
        if not self.alive: 
             self.vel *= 0.9; self.pos += self.vel
             return []

        actions = []

        # 1. MOVEMENT AI (Aggressive)
        to_boss = boss.pos - self.pos
        dist = np.linalg.norm(to_boss)
        norm = to_boss / (dist + 0.01)
        
        # Circle Strafe
        strafe_dir = np.array([-norm[1], norm[0]])
        
        # Maintain optimal range (8-12m)
        approach = 0.0
        if dist > 12: approach = 0.5
        if dist < 8: approach = -0.5
        
        # Avoid Walls
        wall_repel = np.array([0.0, 0.0])
        if abs(self.pos[0]) > 12: wall_repel[0] = -np.sign(self.pos[0]) * 1.0
        if abs(self.pos[1]) > 12: wall_repel[1] = -np.sign(self.pos[1]) * 1.0

        desired_vel = strafe_dir * 0.9 + norm * approach + wall_repel
        accel = desired_vel - self.vel
        self.vel += accel * 0.12 # BUFF: Higher Agility
        
        # Heading
        desired_heading = np.arctan2(self.vel[1], self.vel[0])
        diff = (desired_heading - self.heading + np.pi) % (2*np.pi) - np.pi
        self.heading += diff * 0.2
        self.pos += self.vel
        
        # 2. GUNNERY (Target Boss)
        lead_pos = boss.pos 
        aim = lead_pos - self.pos
        req_angle = np.arctan2(aim[1], aim[0])
        t_diff = (req_angle - self.turret_angle + np.pi) % (2*np.pi) - np.pi
        self.turret_angle += t_diff * 0.5
        
        self.cooldown -= 1
        if abs(t_diff) < 0.2 and self.cooldown <= 0:
            self.cooldown = 5 # BUFF: Faster Fire Rate
            actions.append("FIRE")

        # 3. APS DEFENSE (NEW)
        self.aps_charge += 2.0 # Recharge
        if self.aps_charge > 100: self.aps_charge = 100
        
        for p in projectiles:
            if p["owner"] == "BOSS" and p["type"] == "SHELL":
                d = np.linalg.norm(p["pos"] - self.pos)
                if d < 6.0: # Close range
                    # Check closing velocity
                    rel_vel = p["vel"] - self.vel
                    closing = np.dot(rel_vel, (self.pos - p["pos"])) > 0
                    if closing and self.aps_charge > 20:
                        actions.append(("APS", p))
                        self.aps_charge -= 20
                        break # One per frame max
                        
        return actions

    def record_tracks(self):
        c, s = np.cos(self.heading), np.sin(self.heading)
        off_l = np.array([-0.9, 0.6]); off_r = np.array([-0.9, -0.6])
        gl_l = self.pos + np.array([off_l[0]*c - off_l[1]*s, off_l[0]*s + off_l[1]*c])
        gl_r = self.pos + np.array([off_r[0]*c - off_r[1]*s, off_r[0]*s + off_r[1]*c])
        self.tracks_l.append(gl_l); self.tracks_r.append(gl_r)
        if len(self.tracks_l) > 100: self.tracks_l.pop(0); self.tracks_r.pop(0)

class BossTank:
    def __init__(self):
        self.pos = np.array([5.0, 0.0])
        self.vel = np.array([0.0, 0.0])
        self.heading = np.pi
        self.alive = True
        self.hp = 2000.0
        self.max_hp = 2000.0
        self.turret_angle = np.pi
        self.gun1_cd = 0; self.gun2_cd = 20
        self.aps_charge = 100.0
        self.tracks_l = []; self.tracks_r = []

    def update(self, hero, projectiles):
        if not self.alive: return []
        actions = []

        # MOVEMENT
        to_hero = hero.pos - self.pos
        dist = np.linalg.norm(to_hero)
        desired_heading = np.arctan2(to_hero[1], to_hero[0])
        diff = (desired_heading - self.heading + np.pi) % (2*np.pi) - np.pi
        self.heading += np.clip(diff, -0.015, 0.015) 
        
        throttle = 0.0
        if dist > 9.0: throttle = 0.03
        if dist < 5.0: throttle = -0.02
        self.vel = np.array([np.cos(self.heading), np.sin(self.heading)]) * throttle
        self.pos += self.vel
        
        # TURRET
        t_diff = (desired_heading - self.turret_angle + np.pi) % (2*np.pi) - np.pi
        self.turret_angle += np.clip(t_diff, -0.06, 0.06) 
        
        # FIRE
        aligned = abs(t_diff) < 0.3
        self.gun1_cd -= 1; self.gun2_cd -= 1
        
        if aligned:
            if self.gun1_cd <= 0:
                actions.append("FIRE_L"); self.gun1_cd = 50 
            if self.gun2_cd <= 0:
                actions.append("FIRE_R"); self.gun2_cd = 50
        
        # APS
        self.aps_charge += 0.5
        if self.aps_charge > 100.0: self.aps_charge = 100.0
        for p in projectiles:
            if p["owner"] == "HERO" and p["type"] == "SHELL":
                d = np.linalg.norm(p["pos"] - self.pos)
                if d < 6.0:
                    rel_vel = p["vel"] - self.vel
                    closing = np.dot(rel_vel, (self.pos - p["pos"])) > 0
                    if closing and self.aps_charge >= 20.0:
                        actions.append(("APS", p))
                        self.aps_charge -= 20.0
                        
        return actions

    def record_tracks(self):
        c, s = np.cos(self.heading), np.sin(self.heading)
        off_l = np.array([-2.5, 1.8]); off_r = np.array([-2.5, -1.8]); 
        p = self.pos
        gl_l = p + np.array([off_l[0]*c - off_l[1]*s, off_l[0]*s + off_l[1]*c])
        gl_r = p + np.array([off_r[0]*c - off_r[1]*s, off_r[0]*s + off_r[1]*c])
        self.tracks_l.append(gl_l); self.tracks_r.append(gl_r)
        if len(self.tracks_l) > 150: self.tracks_l.pop(0); self.tracks_r.pop(0)

class BossSim:
    def __init__(self):
        self.hero = HeroTank()
        self.boss = BossTank()
        self.projectiles = []
        self.debris = []
        self.shake = 0.0

    def spawn_shell(self, origin, angle, speed, owner, size, color):
        vel = np.array([np.cos(angle)*speed, np.sin(angle)*speed])
        if owner == "BOSS": vel += self.boss.vel * 0.5
        else: vel += self.hero.vel * 0.5
        self.projectiles.append({
            "pos": origin.copy(), "vel": vel, "owner": owner, "type": "SHELL",
            "life": 90, "color": color, "size": size
        })
        self.spawn_debris(origin, vel*0.3, color, 5)

    def spawn_aps(self, origin, target, color):
        dist = np.linalg.norm(target["pos"] - origin)
        int_pt = target["pos"] + target["vel"] * (dist/2.0)
        vel = (int_pt - origin)
        vel = vel / np.linalg.norm(vel) * 1.8 
        self.projectiles.append({
            "pos": origin.copy(), "vel": vel, "owner": "APS", "type": "APS",
            "target_id": id(target), "life": 15, "color": color, "size": 10
        })

    def spawn_debris(self, pos, vel, color, count):
        for _ in range(count):
            v = vel * 0.5 + np.random.normal(0, 0.3, 2)
            self.debris.append({
                "pos": pos.copy(), "vel": v, "life": 1.0, "color": color, "size": np.random.uniform(3, 10)
            })

    def update(self, frame_idx):
        h_acts = self.hero.update(self.boss, self.projectiles)
        b_acts = self.boss.update(self.hero, self.projectiles)
        self.hero.record_tracks()
        self.boss.record_tracks()
        
        # HERO ACTIONS
        for act in h_acts:
            if isinstance(act, str) and act == "FIRE":
                tip = self.hero.pos + np.array([np.cos(self.hero.turret_angle), np.sin(self.hero.turret_angle)]) * 1.8
                self.spawn_shell(tip, self.hero.turret_angle, 1.25, "HERO", 20, HERO_CYAN)
            elif isinstance(act, tuple) and act[0] == "APS":
                self.spawn_aps(self.hero.pos, act[1], HERO_CYAN)

        # BOSS ACTIONS
        for act in b_acts:
            if isinstance(act, str):
                c, s = np.cos(self.boss.turret_angle), np.sin(self.boss.turret_angle)
                offset_perp = np.array([-s, c]) * 0.8
                if act == "FIRE_L":
                    tip = self.boss.pos + np.array([c, s])*3.5 + offset_perp
                    self.spawn_shell(tip, self.boss.turret_angle, 0.9, "BOSS", 60, BOSS_SHELL)
                if act == "FIRE_R":
                    tip = self.boss.pos + np.array([c, s])*3.5 - offset_perp
                    self.spawn_shell(tip, self.boss.turret_angle, 0.9, "BOSS", 60, BOSS_SHELL)
            elif isinstance(act, tuple) and act[0] == "APS":
                self.spawn_aps(self.boss.pos, act[1], BOSS_CORE)
                    
        # PHYSICS
        active_p = []
        for p in self.projectiles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            hit = False
            
            # SHELL HITS
            if p["type"] == "SHELL":
                # HERO HITS BOSS
                if p["owner"] == "HERO" and self.boss.alive:
                    if np.linalg.norm(p["pos"] - self.boss.pos) < 2.5: 
                        self.boss.hp -= 20 # BUFF: Harder Hitting
                        hit = True
                        self.spawn_debris(p["pos"], p["vel"]*(-0.2), BOSS_RED, 8)
                        if self.boss.hp <= 0:
                            self.boss.alive = False
                            self.spawn_debris(self.boss.pos, np.zeros(2), BOSS_CORE, 250)
                # BOSS HITS HERO
                if p["owner"] == "BOSS" and self.hero.alive:
                    if np.linalg.norm(p["pos"] - self.hero.pos) < 1.0: 
                        self.hero.hp -= 35; hit = True
                        self.spawn_debris(p["pos"], self.hero.vel, HERO_CYAN, 30)
                        if self.hero.hp <= 0:
                            self.hero.alive = False
                            self.spawn_debris(self.hero.pos, self.hero.vel, HERO_CYAN, 80)

            # APS INTERCEPT (Bidirectional)
            if p["type"] == "APS":
                for s in self.projectiles:
                    if s["type"] == "SHELL" and id(s) == p["target_id"]:
                         if np.linalg.norm(p["pos"] - s["pos"]) < 2.0:
                            s["life"] = 0; hit = True; p["life"] = 0
                            self.spawn_debris(s["pos"], s["vel"]*0.1, INTERCEPT_WHITE, 6)
                            break
            
            if not hit and p["life"] > 0: active_p.append(p)
        self.projectiles = active_p
        
        active_d = []
        for d in self.debris:
            d["pos"] += d["vel"]; d["vel"] *= 0.9; d["life"] -= 0.05
            if d["life"] > 0: active_d.append(d)
        self.debris = active_d

    def render(self, frame_idx, ax):
        lim = 14
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal'); ax.set_axis_off()
        ax.set_facecolor(BG_VOID)
        
        # TRACKS
        if len(self.hero.tracks_l) > 1:
            lx, ly = zip(*self.hero.tracks_l); rx, ry = zip(*self.hero.tracks_r)
            ax.plot(lx, ly, color=HERO_CYAN, lw=2, alpha=0.2)
            ax.plot(rx, ry, color=HERO_CYAN, lw=2, alpha=0.2)
        if len(self.boss.tracks_l) > 1:
            lx, ly = zip(*self.boss.tracks_l); rx, ry = zip(*self.boss.tracks_r)
            ax.plot(lx, ly, color="#330000", lw=6, alpha=0.3)
            ax.plot(rx, ry, color="#330000", lw=6, alpha=0.3)

        # BOSS
        b = self.boss; col = BOSS_RED if b.alive else "#221111"
        c, s = np.cos(b.heading), np.sin(b.heading)
        pts = [[3, 1], [2, 2.5], [-2, 2.5], [-3, 1], [-3, -1], [-2, -2.5], [2, -2.5], [3, -1]]
        g_pts = [b.pos + np.array([p[0]*c - p[1]*s, p[0]*s + p[1]*c]) for p in pts]
        ax.add_artist(Polygon(g_pts, closed=True, facecolor="#1a0505", edgecolor=col, linewidth=3, zorder=4))
        
        tc, ts = np.cos(b.turret_angle), np.sin(b.turret_angle)
        perp = np.array([-ts, tc]) * 0.8
        tip = np.array([tc, ts]) * 3.5
        p1 = b.pos + tip + perp; p2 = b.pos + tip - perp
        ax.plot([b.pos[0]+perp[0], p1[0]], [b.pos[1]+perp[1], p1[1]], color="#555", lw=5, zorder=5)
        ax.plot([b.pos[0]-perp[0], p2[0]], [b.pos[1]-perp[1], p2[1]], color="#555", lw=5, zorder=5)
        ax.add_artist(Circle((b.pos[0], b.pos[1]), 1.5, color=col, zorder=6))
        
        # BOSS APS RINGS
        if b.alive:
             for i in range(int(b.aps_charge / 20)):
                 ax.add_artist(Circle((b.pos[0], b.pos[1]), 2.0 + i*0.3, color=BOSS_CORE, fill=False, lw=1, alpha=0.5))

        # HERO
        if self.hero.alive:
            h = self.hero
            c, s = np.cos(h.heading), np.sin(h.heading)
            pts = [[1, 0.7], [-1, 0.7], [-1, -0.7], [1, -0.7]]
            g_pts = [h.pos + np.array([p[0]*c - p[1]*s, p[0]*s + p[1]*c]) for p in pts]
            ax.add_artist(Polygon(g_pts, closed=True, facecolor="#001111", edgecolor=HERO_CYAN, linewidth=2, zorder=7))
            tip = h.pos + np.array([np.cos(h.turret_angle), np.sin(h.turret_angle)])*1.8
            ax.plot([h.pos[0], tip[0]], [h.pos[1], tip[1]], color=HERO_CYAN, lw=2, zorder=8)
            # HERO APS
            for i in range(int(h.aps_charge / 20)):
                 ax.add_artist(Circle((h.pos[0], h.pos[1]), 1.0 + i*0.2, color=INTERCEPT_WHITE, fill=False, lw=1, alpha=0.5))

        # OBJS
        for p in self.projectiles:
            ax.scatter(p["pos"][0], p["pos"][1], color=p["color"], s=p["size"], zorder=10)
        for d in self.debris:
            ax.scatter(d["pos"][0], d["pos"][1], color=d["color"], s=d["size"], alpha=np.clip(d["life"],0,1), zorder=9)

        # HUD
        bar_w = 16.0
        pct = max(0, self.boss.hp / self.boss.max_hp)
        ax.plot([-8, -8 + 16*pct], [12, 12], color=BOSS_RED, lw=8, zorder=20)
        ax.text(0, 10.5, "LOGIC GARDEN 68: THE TITAN (FINALE)", color=BOSS_RED, ha='center', fontsize=10, fontweight='bold', fontfamily='monospace')
        
        pct_h = max(0, self.hero.hp / self.hero.max_hp)
        ax.plot([-12, -12 + 6*pct_h], [-12, -12], color=HERO_CYAN, lw=4, zorder=20)

        if not self.boss.alive:
             ax.text(0, 0, "TARGET NEUTRALIZED", color=HERO_CYAN, ha='center', fontsize=26, fontweight='bold', fontfamily='monospace', bbox=dict(facecolor='black', alpha=0.7))
        elif not self.hero.alive:
             ax.text(0, 0, "M.I.A.", color=BOSS_RED, ha='center', fontsize=30, fontweight='bold', fontfamily='monospace')

        # I/O
        out_dir = "logic_garden_titan_frames"
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"titan_{frame_idx:04d}.png"), facecolor=BG_VOID)
        plt.close()

# --- 3. EXECUTION ---
if __name__ == "__main__":
    print(f"[NURSERY] Boss Fight (Patched)...")
    sim = BossSim()
    for i in range(TOTAL_FRAMES):
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        sim.update(i)
        sim.render(i, ax)
        if i % 60 == 0: print(f"Frame {i}/{TOTAL_FRAMES}")
