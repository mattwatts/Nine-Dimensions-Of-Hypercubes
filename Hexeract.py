import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import math
import gc # Import garbage collector
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
        vertex = np.array([float(bit) * 2 - 1 for bit in binary])
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
        for j in range(i + 1, num_vertices):
            if np.sum(vertices[i] != vertices[j]) == 1:
                edges.append((i, j))
    return edges

# --- 2. Projection and Rotation Functions ---
def project_1d_to_3d(vertices_1d):
    """
    Projects 1D vertices into 3D space by adding zero Y and Z coordinates.
    """
    return np.column_stack((vertices_1d, np.zeros((vertices_1d.shape[0], 2))))

def project_2d_to_3d(vertices_2d):
    """
    Projects 2D vertices into a 3D space by adding a zero Z-coordinate.
    """
    return np.column_stack((vertices_2d, np.zeros(vertices_2d.shape[0])))

def project_3d_to_3d(vertices_3d):
    """
    For a 3D hypercube, the 'projection' to 3D is just the vertices themselves.
    This function simply returns the input 3D vertices without further transformation.
    """
    return vertices_3d

def project_4d_to_3d(vertices_4d):
    """
    Custom linear projection from 4D vertices to 3D space.
    This matrix maps the 4 dimensions into 3D for visualization.
    """
    P_4_3 = np.array([
        [1.0, 0.0, 0.0],   # dim 0 -> X
        [0.0, 1.0, 0.0],   # dim 1 -> Y
        [0.0, 0.0, 1.0],   # dim 2 -> Z
        [0.5, 0.5, 0.5]    # dim 3 -> X, Y, Z (composite contribution)
    ])
    return np.dot(vertices_4d, P_4_3)

def project_5d_to_3d(vertices_5d):
    """
    Custom linear projection from 5D vertices to 3D space.
    This matrix maps the 5 dimensions into 3D for visualization.
    """
    P_5_3 = np.array([
        [1.0, 0.0, 0.0],   # dim 0 -> X
        [0.0, 1.0, 0.0],   # dim 1 -> Y
        [0.0, 0.0, 1.0],   # dim 2 -> Z
        [0.5, 0.5, 0.0],   # dim 3 -> X, Y
        [0.0, 0.5, 0.5]    # dim 4 -> Y, Z
    ])
    return np.dot(vertices_5d, P_5_3)

def project_6d_to_3d(vertices_6d):
    """
    Custom linear projection from 6D vertices to 3D space.
    This matrix maps the 6 dimensions into 3D for visualization.
    """
    P_6_3 = np.array([
        [1.0, 0.0, 0.0],   # dim 0 -> X
        [0.0, 1.0, 0.0],   # dim 1 -> Y
        [0.0, 0.0, 1.0],   # dim 2 -> Z
        [0.5, 0.5, 0.0],   # dim 3 -> X, Y
        [0.0, 0.5, 0.5],   # dim 4 -> Y, Z
        [0.5, 0.0, 0.5]    # dim 5 -> X, Z
    ])
    return np.dot(vertices_6d, P_6_3)

def project_12d_to_3d(vertices_12d):
    """
    Custom linear projection from 12D vertices to 3D space.
    This matrix maps the 12 dimensions into 3D for visualization.
    """
    P_12_3 = np.array([
        [1.0, 0.0, 0.0],   # dim 0 -> X
        [0.0, 1.0, 0.0],   # dim 1 -> Y
        [0.0, 0.0, 1.0],   # dim 2 -> Z
        [0.5, 0.5, 0.0],   # dim 3 -> X, Y
        [0.0, 0.5, 0.5],   # dim 4 -> Y, Z
        [0.5, 0.0, 0.5],   # dim 5 -> X, Z
        [0.3, 0.3, 0.3],   # dim 6 -> X, Y, Z
        [0.2, 0.2, 0.0],   # dim 7 -> X, Y
        [0.0, 0.2, 0.2],   # dim 8 -> Y, Z
        [0.2, 0.0, 0.2],   # dim 9 -> X, Z
        [0.1, 0.1, 0.0],   # dim 10 -> X, Y
        [0.05, 0.05, 0.05] # dim 11 -> X, Y, Z (composite, lighter contribution)
    ])
    return np.dot(vertices_12d, P_12_3)

def project_13d_to_3d(vertices_13d):
    """
    Custom linear projection from 13D vertices to 3D space.
    This matrix maps the 13 dimensions into 3D for visualization.
    """
    P_13_3 = np.array([
        [1.0, 0.0, 0.0],   # dim 0 -> X
        [0.0, 1.0, 0.0],   # dim 1 -> Y
        [0.0, 0.0, 1.0],   # dim 2 -> Z
        [0.5, 0.5, 0.0],   # dim 3 -> X, Y
        [0.0, 0.5, 0.5],   # dim 4 -> Y, Z
        [0.5, 0.0, 0.5],   # dim 5 -> X, Z
        [0.3, 0.3, 0.3],   # dim 6 -> X, Y, Z
        [0.2, 0.2, 0.0],   # dim 7 -> X, Y
        [0.0, 0.2, 0.2],   # dim 8 -> Y, Z
        [0.2, 0.0, 0.2],   # dim 9 -> X, Z
        [0.1, 0.1, 0.0],   # dim 10 -> X, Y
        [0.05, 0.05, 0.05], # dim 11 -> X, Y, Z
        [0.05, 0.0, 0.05]   # dim 12 -> X, Z (lighter contribution)
    ])
    return np.dot(vertices_13d, P_13_3)

def project_14d_to_3d(vertices_14d):
    """
    Custom linear projection from 14D vertices to 3D space.
    This matrix maps the 14 dimensions into 3D for visualization.
    """
    P_14_3 = np.array([
        [1.0, 0.0, 0.0],   # dim 0 -> X
        [0.0, 1.0, 0.0],   # dim 1 -> Y
        [0.0, 0.0, 1.0],   # dim 2 -> Z
        [0.5, 0.5, 0.0],   # dim 3 -> X, Y
        [0.0, 0.5, 0.5],   # dim 4 -> Y, Z
        [0.5, 0.0, 0.5],   # dim 5 -> X, Z
        [0.3, 0.3, 0.3],   # dim 6 -> X, Y, Z
        [0.2, 0.2, 0.0],   # dim 7 -> X, Y
        [0.0, 0.2, 0.2],   # dim 8 -> Y, Z
        [0.2, 0.0, 0.2],   # dim 9 -> X, Z
        [0.1, 0.1, 0.0],   # dim 10 -> X, Y
        [0.05, 0.05, 0.05], # dim 11 -> X, Y, Z
        [0.05, 0.0, 0.05],  # dim 12 -> X, Z
        [0.02, 0.02, 0.02]  # dim 13 -> X, Y, Z (even lighter contribution)
    ])
    return np.dot(vertices_14d, P_14_3)

def project_15d_to_3d(vertices_15d):
    """
    Custom linear projection from 15D vertices to 3D space.
    This matrix maps the 15 dimensions into 3D for visualization.
    """
    P_15_3 = np.array([
        [1.0, 0.0, 0.0],    # dim 0 -> X
        [0.0, 1.0, 0.0],    # dim 1 -> Y
        [0.0, 0.0, 1.0],    # dim 2 -> Z
        [0.5, 0.5, 0.0],    # dim 3 -> X, Y
        [0.0, 0.5, 0.5],    # dim 4 -> Y, Z
        [0.5, 0.0, 0.5],    # dim 5 -> X, Z
        [0.3, 0.3, 0.3],    # dim 6 -> X, Y, Z
        [0.2, 0.2, 0.0],    # dim 7 -> X, Y
        [0.0, 0.2, 0.2],    # dim 8 -> Y, Z
        [0.2, 0.0, 0.2],    # dim 9 -> X, Z
        [0.1, 0.1, 0.0],    # dim 10 -> X, Y
        [0.05, 0.05, 0.05], # dim 11 -> X, Y, Z
        [0.05, 0.0, 0.05],  # dim 12 -> X, Z
        [0.02, 0.02, 0.02], # dim 13 -> X, Y, Z
        [0.01, 0.01, 0.0]   # dim 14 -> X, Y (even lighter contribution)
    ])
    return np.dot(vertices_15d, P_15_3)

def project_16d_to_3d(vertices_16d):
    """
    Custom linear projection from 16D vertices to 3D space.
    This matrix maps the 16 dimensions into 3D for visualization.
    """
    P_16_3 = np.array([
        [1.0, 0.0, 0.0],    # dim 0 -> X
        [0.0, 1.0, 0.0],    # dim 1 -> Y
        [0.0, 0.0, 1.0],    # dim 2 -> Z
        [0.5, 0.5, 0.0],    # dim 3 -> X, Y
        [0.0, 0.5, 0.5],    # dim 4 -> Y, Z
        [0.5, 0.0, 0.5],    # dim 5 -> X, Z
        [0.3, 0.3, 0.3],    # dim 6 -> X, Y, Z
        [0.2, 0.2, 0.0],    # dim 7 -> X, Y
        [0.0, 0.2, 0.2],    # dim 8 -> Y, Z
        [0.2, 0.0, 0.2],    # dim 9 -> X, Z
        [0.1, 0.1, 0.0],    # dim 10 -> X, Y
        [0.05, 0.05, 0.05], # dim 11 -> X, Y, Z
        [0.05, 0.0, 0.05],  # dim 12 -> X, Z
        [0.02, 0.02, 0.02], # dim 13 -> X, Y, Z
        [0.01, 0.01, 0.0],  # dim 14 -> X, Y
        [0.01, 0.0, 0.01]   # dim 15 -> X, Z (even lighter contribution)
    ])
    return np.dot(vertices_16d, P_16_3)

def project_17d_to_3d(vertices_17d):
    """
    Custom linear projection from 17D vertices to 3D space.
    This matrix maps the 17 dimensions into 3D for visualization.
    """
    P_17_3 = np.array([
        [1.0, 0.0, 0.0],    # dim 0 -> X
        [0.0, 1.0, 0.0],    # dim 1 -> Y
        [0.0, 0.0, 1.0],    # dim 2 -> Z
        [0.5, 0.5, 0.0],    # dim 3 -> X, Y
        [0.0, 0.5, 0.5],    # dim 4 -> Y, Z
        [0.5, 0.0, 0.5],    # dim 5 -> X, Z
        [0.3, 0.3, 0.3],    # dim 6 -> X, Y, Z
        [0.2, 0.2, 0.0],    # dim 7 -> X, Y
        [0.0, 0.2, 0.2],    # dim 8 -> Y, Z
        [0.2, 0.0, 0.2],    # dim 9 -> X, Z
        [0.1, 0.1, 0.0],    # dim 10 -> X, Y
        [0.05, 0.05, 0.05], # dim 11 -> X, Y, Z
        [0.05, 0.0, 0.05],  # dim 12 -> X, Z
        [0.02, 0.02, 0.02], # dim 13 -> X, Y, Z
        [0.01, 0.01, 0.0],  # dim 14 -> X, Y
        [0.01, 0.0, 0.01],  # dim 15 -> X, Z
        [0.005, 0.005, 0.005] # dim 16 -> X, Y, Z (even lighter contribution)
    ])
    return np.dot(vertices_17d, P_17_3)


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
    rotation_planes_main,
    rotation_speeds_main
):
    """Generates and saves a single frame for the hypercube animation."""
    theta_base = (frame_idx / total_frames) * 2 * np.pi

    projected_3d = None

    if dim <= 3:
        # For 1D, 2D, 3D hypercubes, we want to rotate them as 3D objects.
        raw_3d_vertices = None
        actual_rotation_planes = []
        actual_rotation_speeds = []
        rotation_space_dim = 3 # Always rotate in 3D space for these low dimensions

        if dim == 1:
            raw_3d_vertices = project_1d_to_3d(vertices_initial)
            actual_rotation_planes = [(0, 1)] # Rotate in XY plane of 3D space
            actual_rotation_speeds = [2]
        elif dim == 2:
            raw_3d_vertices = project_2d_to_3d(vertices_initial)
            actual_rotation_planes = [(0, 1), (0, 2), (1, 2)] # Full 3D rotation
            actual_rotation_speeds = [2, 3, 4]
        elif dim == 3:
            raw_3d_vertices = project_3d_to_3d(vertices_initial)
            actual_rotation_planes = rotation_planes_main # Use the 3D-specific rotations defined in main
            actual_rotation_speeds = rotation_speeds_main

        # Apply 3D rotation to the (conceptually) 3D vertices
        current_rotated_vertices_3d = np.copy(raw_3d_vertices)
        for i, (d1, d2) in enumerate(actual_rotation_planes):
            theta = theta_base * actual_rotation_speeds[i]
            R_plane_3d = np.identity(rotation_space_dim)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            R_plane_3d[d1, d1] = cos_t
            R_plane_3d[d1, d2] = -sin_t
            R_plane_3d[d2, d1] = sin_t
            R_plane_3d[d2, d2] = cos_t
            current_rotated_vertices_3d = np.dot(current_rotated_vertices_3d, R_plane_3d)
        projected_3d = current_rotated_vertices_3d

    else: # dim > 3
        # For higher dimensions, rotate in N-D space first, then project to 3D
        current_rotated_vertices = np.copy(vertices_initial)
        for i, (d1, d2) in enumerate(rotation_planes_main): # Use N-D rotations from main
            theta = theta_base * rotation_speeds_main[i]
            R_plane = np.identity(dim) # Rotation matrix for current dim
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            R_plane[d1, d1] = cos_t
            R_plane[d1, d2] = -sin_t
            R_plane[d2, d1] = sin_t
            R_plane[d2, d2] = cos_t
            current_rotated_vertices = np.dot(current_rotated_vertices, R_plane)

        # Call the correct N-D to 3D projection based on current DIM
        if dim == 4: projected_3d = project_4d_to_3d(current_rotated_vertices)
        elif dim == 5: projected_3d = project_5d_to_3d(current_rotated_vertices)
        elif dim == 6: projected_3d = project_6d_to_3d(current_rotated_vertices)
        elif dim == 12: projected_3d = project_12d_to_3d(current_rotated_vertices)
        elif dim == 13: projected_3d = project_13d_to_3d(current_rotated_vertices)
        elif dim == 14: projected_3d = project_14d_to_3d(current_rotated_vertices)
        elif dim == 15: projected_3d = project_15d_to_3d(current_rotated_vertices)
        elif dim == 16: projected_3d = project_16d_to_3d(current_rotated_vertices)
        elif dim == 17: projected_3d = project_17d_to_3d(current_rotated_vertices)
        else:
            raise ValueError(f"Unhandled dimension for N-D rotation and projection: {dim}")

    projected_2d = project_3d_to_2d_isometric(projected_3d)

    # --- Plotting for the Current Frame ---
    fig, ax = plt.subplots(figsize=(10, 10)) # Figure size in inches (10*60 dpi = 600 pixels)

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
    DIM = 6
    FPS = 24
    OUTPUT_FOLDER = "frames_6D"
    ROTATION_TIME_SECONDS = 45 # Faster execution
    TOTAL_FRAMES = ROTATION_TIME_SECONDS * FPS
    OUTPUT_VIDEO_FILENAME = "Hexeract.mp4" # <--- Updated output filename

    # --- Matt's Aesthetic Preferences (Blue-on-Black Theme) ---
    BACKGROUND_COLOR = '#000000' # Black
    NODE_COLOR = '#4169E1' # RoyalBlue
    NODE_EDGE_COLOR = '#B0C4DE' # LightSteelBlue (Toned-down blue for boundaries)
    EDGE_COLOR = '#1E90FF' # DodgerBlue

    EDGE_ALPHA = 0.6
    EDGE_WIDTH = 1.0
    BASE_NODE_SIZE = 50 # Smaller nodes
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
    # For 1D/2D/3D, project to 3D first for limit calculation
    temp_3d_points_for_limits = None
    if DIM == 1:
        temp_3d_points_for_limits = project_1d_to_3d(vertices_initial)
    elif DIM == 2:
        temp_3d_points_for_limits = project_2d_to_3d(vertices_initial)
    elif DIM == 3:
        temp_3d_points_for_limits = project_3d_to_3d(vertices_initial)
    elif DIM == 4:
        temp_3d_points_for_limits = project_4d_to_3d(vertices_initial)
    elif DIM == 5:
        temp_3d_points_for_limits = project_5d_to_3d(vertices_initial)
    elif DIM == 6:
        temp_3d_points_for_limits = project_6d_to_3d(vertices_initial)
    elif DIM == 12:
        temp_3d_points_for_limits = project_12d_to_3d(vertices_initial)
    elif DIM == 13:
        temp_3d_points_for_limits = project_13d_to_3d(vertices_initial)
    elif DIM == 14:
        temp_3d_points_for_limits = project_14d_to_3d(vertices_initial)
    elif DIM == 15:
        temp_3d_points_for_limits = project_15d_to_3d(vertices_initial)
    elif DIM == 16:
        temp_3d_points_for_limits = project_16d_to_3d(vertices_initial)
    elif DIM == 17:
        temp_3d_points_for_limits = project_17d_to_3d(vertices_initial)
    else:
        raise ValueError(f"Unsupported dimension for initial limit calculation: {DIM}")

    temp_2d_points = project_3d_to_2d_isometric(temp_3d_points_for_limits)
    min_x, max_x = np.min(temp_2d_points[:, 0]), np.max(temp_2d_points[:, 0])
    min_y, max_y = np.min(temp_2d_points[:, 1]), np.max(temp_2d_points[:, 1])

    axis_max = max(abs(min_x), abs(max_x), abs(min_y), abs(max_y)) * PADDING_FACTOR
    xlim_min_val, xlim_max_val = -axis_max, axis_max
    ylim_min_val, ylim_max_val = -axis_max, axis_max

    # --- Rotation Parameters for passing to generate_single_frame ---
    # For DIM <= 3, these lists are either empty (for 1D/2D handled internally) or actual 3D rotations.
    rotation_planes = []
    rotation_speeds = []
    if DIM == 3: # Only for actual 3D, use these from main
        rotation_planes = [(0, 1), (0, 2), (1, 2)] # X-Y, X-Z, Y-Z plane rotations
        rotation_speeds = [2, 3, 4]
    elif DIM > 3: # DIM > 3, use the N-D rotation parameters as before
        rotation_planes = [(i, i + 1) for i in range(DIM - 1)]
        rotation_speeds = [i + 2 for i in range(DIM - 1)]

    print(f"Starting parallel generation of {TOTAL_FRAMES} frames using {cpu_count()} processes.")
    print(f"Each frame will be saved as {60 * 10}x{60 * 10} pixels.")
    print(f"Animation: {DIM}D hypercube, {ROTATION_TIME_SECONDS}-second continuous rotation, blue-on-black color scheme with toned-down node boundaries and smaller nodes (optimized for speed and memory).")

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
            rotation_planes, # Pass the chosen rotation_planes
            rotation_speeds  # Pass the chosen rotation_speeds
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
