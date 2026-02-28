import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import math
import gc  # Import garbage collector
from multiprocessing import Pool, cpu_count

# Force matplotlib to use the 'Agg' backend.
matplotlib.use('Agg')

# --- 1. Hypercube Generation Functions ---
def generate_hypercube_vertices(dim):
    """
    Generates all vertices of a d-dimensional hypercube.
    Vertices are represented as vectors with components -1 or 1.
    """
    vertices = []
    for i in range(2**dim):
        binary = bin(i)[2:].zfill(dim)
        vertex = np.array([float(bit)*2-1 for bit in binary])
        vertices.append(vertex)
    return np.array(vertices)

def generate_hypercube_edges(vertices, dim):
    """
    Generates all edges of a d-dimensional hypercube.
    Edges connect vertices that differ in exactly one coordinate.
    """
    edges = []
    num_vertices = len(vertices)
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            if np.sum(vertices[i] != vertices[j]) == 1:
                edges.append((i, j))
    return edges

# --- 2. Projection and Rotation Functions ---
def project_9d_to_3d(vertices_9d):
    """
    Custom linear projection from 9D vertices to 3D space.
    This matrix maps the 9 dimensions into 3D for visualization.
    """
    P_9_3 = np.array([
        [1.0, 0.0, 0.0],  # dim 0 -> X
        [0.0, 1.0, 0.0],  # dim 1 -> Y
        [0.0, 0.0, 1.0],  # dim 2 -> Z
        [0.5, 0.5, 0.0],  # dim 3 -> X, Y
        [0.0, 0.5, 0.5],  # dim 4 -> Y, Z
        [0.5, 0.0, 0.5],  # dim 5 -> X, Z
        [0.3, 0.3, 0.3],  # dim 6 -> X, Y, Z
        [0.2, 0.2, 0.0],  # dim 7 -> X, Y
        [0.0, 0.2, 0.2]   # dim 8 -> Y, Z
    ])
    return np.dot(vertices_9d, P_9_3)

def project_3d_to_2d_isometric(vertices_3d):
    """
    Projects 3D points to 2D using an isometric projection,
    consistent with the 'Isometric 3D projections' preference [so-100].
    Optimized to use vectorized numpy operations.
    """
    cos_30 = np.cos(np.radians(30))
    sin_30 = np.sin(np.radians(30))
    proj_x = (vertices_3d[:, 0] - vertices_3d[:, 1]) * cos_30
    proj_y = (vertices_3d[:, 0] + vertices_3d[:, 1]) * sin_30 - vertices_3d[:, 2]
    return np.column_stack((proj_x, proj_y))

# --- Worker Function for Parallel Processing ---
def generate_single_frame(
    frame_idx,
    dim,
    total_frames,
    output_folder,
    background_color,
    node_color,
    node_edge_color,
    edge_color,
    edge_alpha,
    edge_width,
    base_node_size,
    padding_factor,
    vertices_initial,
    edges_list,
    node_colors_map,
    xlim_min_val,
    xlim_max_val,
    ylim_min_val,
    ylim_max_val,
    rotation_planes,
    rotation_speeds
):
    """Generates and saves a single frame for the hypercube animation."""
    theta_base = (frame_idx / total_frames) * 2 * np.pi
    current_rotated_vertices = np.copy(vertices_initial)
    
    for i, (d1, d2) in enumerate(rotation_planes):
        theta = theta_base * rotation_speeds[i]
        R_plane = np.identity(dim)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_plane[d1, d1] = cos_t
        R_plane[d1, d2] = -sin_t
        R_plane[d2, d1] = sin_t
        R_plane[d2, d2] = cos_t
        current_rotated_vertices = np.dot(current_rotated_vertices, R_plane)
        
    projected_3d = project_9d_to_3d(current_rotated_vertices)
    projected_2d = project_3d_to_2d_isometric(projected_3d)

    # --- Plotting for the Current Frame ---
    fig, ax = plt.subplots(figsize=(10, 10))  # Figure size in inches (10*60 dpi = 600 pixels)
    ax.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)
    ax.set_xlim(xlim_min_val, xlim_max_val)
    ax.set_ylim(ylim_min_val, ylim_max_val)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # Draw edges first (behind nodes)
    for edge in edges_list:
        p1_idx, p2_idx = edge
        p1 = projected_2d[p1_idx]
        p2 = projected_2d[p2_idx]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=edge_color, alpha=edge_alpha, linewidth=edge_width, zorder=1)

    # Draw nodes with fixed size (no sparkle effect) and toned-down blue boundaries
    for i, node_pos in enumerate(projected_2d):
        current_node_size = base_node_size
        ax.scatter(node_pos[0], node_pos[1],
                   s=current_node_size,
                   color=node_colors_map[i],
                   zorder=2,
                   edgecolors=node_edge_color,
                   linewidths=0.5)

    frame_filename = os.path.join(output_folder, f"frame_{frame_idx:04d}.png")
    plt.savefig(frame_filename, dpi=60, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    del fig, ax
    gc.collect()

    # Progress reporting for each process
    if (frame_idx + 1) % 100 == 0 or frame_idx == total_frames - 1:
        print(f"Generated frame {frame_idx + 1}/{total_frames} by process {os.getpid()}")

if __name__ == '__main__':
    # --- Configuration for Animation ---
    DIM = 9
    FPS = 24
    OUTPUT_FOLDER = "frames_9D"
    ROTATION_TIME_SECONDS = 45  # Faster execution
    TOTAL_FRAMES = ROTATION_TIME_SECONDS * FPS
    OUTPUT_VIDEO_FILENAME = "Enneract.mp4"  # Output filename

    # --- Matt's Aesthetic Preferences (Blue-on-Black Theme) ---
    BACKGROUND_COLOR = '#000000'  # Black
    NODE_COLOR = '#4169E1'        # RoyalBlue
    NODE_EDGE_COLOR = '#B0C4DE'   # LightSteelBlue (Toned-down blue for boundaries)
    EDGE_COLOR = '#1E90FF'        # DodgerBlue
    EDGE_ALPHA = 0.6
    EDGE_WIDTH = 1.0
    BASE_NODE_SIZE = 50           # Smaller nodes
    SPARKLE_MODULATION = 0.0
    PADDING_FACTOR = 1.05

    # --- Prepare Output Directory ---
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # --- Initialize Hypercube Structure (once in main process) ---
    vertices_initial = generate_hypercube_vertices(DIM)
    edges_list = generate_hypercube_edges(vertices_initial, DIM)
    node_colors_map = [NODE_COLOR for _ in range(len(vertices_initial))]

    # --- Determine Plot Limits for Frame Filling (once in main process) ---
    temp_3d_points = project_9d_to_3d(vertices_initial)
    temp_2d_points = project_3d_to_2d_isometric(temp_3d_points)
    
    min_x, max_x = np.min(temp_2d_points[:, 0]), np.max(temp_2d_points[:, 0])
    min_y, max_y = np.min(temp_2d_points[:, 1]), np.max(temp_2d_points[:, 1])
    
    axis_max = max(abs(min_x), abs(max_x), abs(min_y), abs(max_y)) * PADDING_FACTOR
    xlim_min_val, xlim_max_val = -axis_max, axis_max
    ylim_min_val, ylim_max_val = -axis_max, axis_max

    # --- Rotation Parameters for 9D ---
    rotation_planes = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    rotation_speeds = [2, 3, 4, 5, 6, 7, 8, 9]

    print(f"Starting parallel generation of {TOTAL_FRAMES} frames using {cpu_count()} processes.")
    print(f"Each frame will be saved as {60*10}x{60*10} pixels.")
    print(f"Animation: 9D hypercube, {ROTATION_TIME_SECONDS}-second continuous rotation, blue-on-black color scheme with toned-down node boundaries and smaller nodes (optimized for speed and memory).")

    # --- Prepare arguments for starmap ---
    args_list = [
        (
            frame_idx,
            DIM,
            TOTAL_FRAMES,
            OUTPUT_FOLDER,
            BACKGROUND_COLOR,
            NODE_COLOR,
            NODE_EDGE_COLOR,
            EDGE_COLOR,
            EDGE_ALPHA,
            EDGE_WIDTH,
            BASE_NODE_SIZE,
            PADDING_FACTOR,
            vertices_initial,
            edges_list,
            node_colors_map,
            xlim_min_val,
            xlim_max_val,
            ylim_min_val,
            ylim_max_val,
            rotation_planes,
            rotation_speeds
        )
        for frame_idx in range(TOTAL_FRAMES)
    ]

    # --- Animation Loop: Generate Each Frame in Parallel ---
    with Pool(processes=cpu_count()) as pool:
        list(pool.starmap(generate_single_frame, args_list))

    print(f"\nAll {TOTAL_FRAMES} frames generated in the '{OUTPUT_FOLDER}' directory.")

    # --- FFmpeg Command for Video Generation ---
    ffmpeg_command = (
        f"ffmpeg -y -framerate {FPS} -i {OUTPUT_FOLDER}/frame_%04d.png "
        f"-c:v libx264 -vf \"fps={FPS},format=yuv420p,scale=600:600\" -crf 18 "
        f"{OUTPUT_VIDEO_FILENAME}"
    )

    print("\nTo generate the MP4 video, please ensure FFmpeg is installed and accessible in your system's PATH.")
    print("Then, navigate to the directory where you ran this Python script and execute the following command in your terminal:")
    print(f"```bash\n{ffmpeg_command}\n```")
    print(f"\nThis will create a '{OUTPUT_VIDEO_FILENAME}' file, approximately {ROTATION_TIME_SECONDS} seconds long, with continuous rotation and a blue-on-black color scheme.")
