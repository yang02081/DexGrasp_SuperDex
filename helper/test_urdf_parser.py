import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.widgets import CheckButtons
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# In environments without Chinese fonts, matplotlib may repeatedly print "Glyph missing" warnings; this does not affect numerical results or image generation.
warnings.filterwarnings("ignore", message=r"Glyph .* missing from current font\.")

# ====================== Decoupled 3D visualization module ======================
class FrameVisualizer:
    """
    Generic 3D coordinate-frame visualizer (does not depend on URDF/robotics libraries).
    Input: a list of arbitrary frames, each containing:
        name: frame name (str)
        position: 3D position [x, y, z] (np.array)
        rotation: 3x3 rotation matrix (np.array)
    """
    
    # Predefined colors (consistent with robotics conventions)
    AXIS_COLORS = {
        'x': '#ff4444',  # red
        'y': '#44ff44',  # green
        'z': '#4444ff'   # blue
    }
    AXIS_LENGTH = 0.025  # Arrow length (meters), suitable for dexterous-hand scale
    
    def __init__(
        self,
        figsize=(10, 8),
        *,
        white_bg: bool = True,
        hide_axes: bool = True,
        frame_arrow_lw: float = 2.0,
        frame_arrow_ms: float = 15.0,
        axis_arrow_lw: float = 2.5,
        axis_arrow_ms: float = 18.0,
    ):
        """Initialize the 3D canvas."""
        fig_face = 'white' if white_bg else '#f0f0f0'
        ax_face = 'white' if white_bg else '#e0e0e0'

        self.fig = plt.figure(figsize=figsize, facecolor=fig_face)
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor=ax_face)

        # Arrow thickness / arrow head size (mutation_scale) are configurable
        self._frame_arrow_lw = float(frame_arrow_lw)
        self._frame_arrow_ms = float(frame_arrow_ms)
        self._axis_arrow_lw = float(axis_arrow_lw)
        self._axis_arrow_ms = float(axis_arrow_ms)

        # Store all toggleable artists (for interactive show/hide)
        self._frame_artists = []  # joint local frames (XYZ triads)
        self._axis_artists = []   # joint rotation axes (black arrows)
        self._mesh_artists = []   # hand mesh surfaces

        # Optimize the 3D view
        self.ax.view_init(elev=25, azim=45)
        self.ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

        # Hide the axis frame/ticks/labels/grid by default to keep the background clean
        self.ax.grid(False)
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')
        self.ax.set_zlabel('')

        if hide_axes:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_zticks([])

            # Make the mplot3d pane/grid transparent to avoid leftover gray backgrounds/borders
            try:
                for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
                    axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                    axis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)
            except Exception:
                pass

            try:
                self.ax.set_axis_off()
            except Exception:
                pass

    def _set_visible(self, artists, visible: bool):
        for a in artists:
            try:
                a.set_visible(visible)
            except Exception:
                pass

    def set_frames_visible(self, visible: bool):
        self._set_visible(self._frame_artists, visible)

    def set_axis_visible(self, visible: bool):
        self._set_visible(self._axis_artists, visible)

    def set_mesh_visible(self, visible: bool):
        self._set_visible(self._mesh_artists, visible)
        
    class _Arrow3D(FancyArrowPatch):
        """Custom 3D arrow class (matplotlib has no native 3D arrow support)."""
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs
            
        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

        def do_3d_projection(self, renderer=None):
            """
            Override this method to support 3D depth sorting.
            """
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            return np.min(zs)

    def add_frame(self, name, position, rotation, show_label=True):
        """
        Add a single coordinate frame to the visualization.
        :param name: frame name
        :param position: [x, y, z] position
        :param rotation: 3x3 rotation matrix (columns = axis directions)
        :param show_label: whether to display the frame name
        """
        # 1. Draw the origin (sphere effect)
        origin_artist = self.ax.scatter(
            position[0], position[1], position[2],
            s=80, c='k', alpha=0.8, edgecolor='w', zorder=100,
            label=f'Origin: {name}' if show_label else ""
        )
        self._frame_artists.append(origin_artist)
        
        # 2. Draw an arrow for each axis
        for i, axis in enumerate(['x', 'y', 'z']):
            # Compute the axis endpoint: start (origin) + direction * length
            direction = rotation[:, i]  # i-th column of the rotation matrix
            end_point = position + direction * self.AXIS_LENGTH
            
            # Create a 3D arrow
            arrow = self._Arrow3D(
                [position[0], end_point[0]],
                [position[1], end_point[1]],
                [position[2], end_point[2]],
                mutation_scale=self._frame_arrow_ms,
                lw=self._frame_arrow_lw,
                arrowstyle='-|>',
                color=self.AXIS_COLORS[axis],
                zorder=50 + i  # Ensure arrows appear in front of the point
            )
            self.ax.add_artist(arrow)
            self._frame_artists.append(arrow)
            
            # 3. Add axis labels (near the arrow tip)
            if show_label:
                label_pos = position + direction * (self.AXIS_LENGTH * 1.2)
                self.ax.text(
                    label_pos[0], label_pos[1], label_pos[2],
                    f"{name}_{axis.upper()}",
                    color=self.AXIS_COLORS[axis],
                    fontsize=9,
                    ha='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5)
                )
        
        # 4. Add the frame name (above the origin)
        if show_label:
            label_pos = position + np.array([0, 0, self.AXIS_LENGTH * 1.5])
            self.ax.text(
                label_pos[0], label_pos[1], label_pos[2],
                name,
                color='darkblue',
                fontsize=10,
                weight='bold',
                ha='center'
            )

    def add_axis_vector(self, name, position, direction, length=None, show_label=True, *, color='k'):
        """Draw a direction-vector arrow (used for joint rotation axes)."""
        direction = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm <= 0:
            return
        direction = direction / norm

        if length is None:
            length = self.AXIS_LENGTH
        end_point = position + direction * length

        arrow = self._Arrow3D(
            [position[0], end_point[0]],
            [position[1], end_point[1]],
            [position[2], end_point[2]],
            mutation_scale=self._axis_arrow_ms,
            lw=self._axis_arrow_lw,
            arrowstyle='-|>',
            color=color,
            zorder=120,
        )
        self.ax.add_artist(arrow)
        self._axis_artists.append(arrow)

        if show_label:
            label_pos = position + direction * (length * 1.15)
            self.ax.text(
                label_pos[0], label_pos[1], label_pos[2],
                f"{name}_AXIS",
                color=color,
                fontsize=9,
                ha='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5)
            )

    def add_mesh(self, vertices: np.ndarray, faces: np.ndarray, *, color=(0.75, 0.75, 0.75), alpha=0.18):
        """Add a triangular mesh surface to the 3D axes."""
        if vertices is None or faces is None:
            return
        if len(vertices) == 0 or len(faces) == 0:
            return
        tris = vertices[faces]
        coll = Poly3DCollection(tris, linewidths=0.0, edgecolors='none')
        coll.set_facecolor((*color, alpha))
        coll.set_zorder(1)
        self.ax.add_collection3d(coll)
        self._mesh_artists.append(coll)
    
    def add_frames(self, frames_data, show_labels=True):
        """
        Add multiple coordinate frames.
        :param frames_data: List where each element is a dict:
            {'name': str, 'position': np.array, 'rotation': np.array}
        :param show_labels: whether to show all labels
        """
        for frame in frames_data:
            self.add_frame(
                name=frame['name'],
                position=frame['position'],
                rotation=frame['rotation'],
                show_label=show_labels
            )
    
    def set_limits(self, center=None, radius=0.1):
        """
        Set a reasonable 3D view range (auto-adapts to dexterous-hand scale).
        :param center: view center [x, y, z]; defaults to the centroid of all points
        :param radius: view radius (meters)
        """
        if center is None:
            # Automatically compute the centroid of all points
            all_positions = [frame.get('position', np.zeros(3)) for frame in getattr(self, '_frames', [])]
            if all_positions:
                center = np.mean(all_positions, axis=0)
            else:
                center = np.zeros(3)
        
        self.ax.set_xlim(center[0]-radius, center[0]+radius)
        self.ax.set_ylim(center[1]-radius, center[1]+radius)
        self.ax.set_zlim(center[2]-radius, center[2]+radius)
    
    def show(self, title="Joint Coordinate Frames", legend=True):
        """Display the visualization result."""
        self.ax.set_title(title, fontsize=14, pad=20)
        if legend:
            self.ax.legend(loc='upper right', fontsize=9)

        # Interactive toggles: compare frames / axis / mesh inside the figure
        has_ctrl = False
        try:
            ax_ctrl = self.fig.add_axes([0.02, 0.02, 0.18, 0.10], facecolor='white')
            labels = ["frames", "axis", "mesh"]
            actives = [
                self._frame_artists and self._frame_artists[0].get_visible(),
                self._axis_artists and self._axis_artists[0].get_visible(),
                self._mesh_artists and self._mesh_artists[0].get_visible(),
            ]
            # If a list is empty, default to False to avoid get_visible errors
            actives = [
                bool(actives[0]) if isinstance(actives[0], bool) else False,
                bool(actives[1]) if isinstance(actives[1], bool) else False,
                bool(actives[2]) if isinstance(actives[2], bool) else False,
            ]
            check = CheckButtons(ax_ctrl, labels, actives)
            has_ctrl = True

            def _on_click(label):
                if label == "frames":
                    new_vis = not (self._frame_artists and self._frame_artists[0].get_visible())
                    self.set_frames_visible(new_vis)
                elif label == "axis":
                    new_vis = not (self._axis_artists and self._axis_artists[0].get_visible())
                    self.set_axis_visible(new_vis)
                elif label == "mesh":
                    new_vis = not (self._mesh_artists and self._mesh_artists[0].get_visible())
                    self.set_mesh_visible(new_vis)
                self.fig.canvas.draw_idle()

            check.on_clicked(_on_click)
        except Exception:
            pass

        # Do not call tight_layout here: it can conflict with CheckButtons / legend and trigger warnings
        plt.show()
    
    def save(self, filename="frames_visualization.png", dpi=150):
        """Save the visualization result."""
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Visualization saved to {filename}")


# ====================== Chain URDF parsing and visualization ======================
try:
    # Run as a package: python -m helper.test_urdf_parser
    from .urdf_parser import URDFJointPoseExtractor  # type: ignore
    from .constants import (  # type: ignore
        HandType,
        NodeTypeSemantic,
        FLEX,
        ABD,
        ROT,
        get_data_dict,
        build_node_to_joint_sequence,
        build_joint_semantic_map,
    )
except Exception:
    # Run as a script: python helper/test_urdf_parser.py
    from helper.urdf_parser import URDFJointPoseExtractor  # type: ignore
    from helper.constants import (  # type: ignore
        HandType,
        NodeTypeSemantic,
        FLEX,
        ABD,
        ROT,
        get_data_dict,
        build_node_to_joint_sequence,
        build_joint_semantic_map,
    )
import argparse
import os
TASK_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize URDF hand model.')
    parser.add_argument("--urdf_allegro", type=str, default=os.path.join(TASK_PATH, "rsc", "allegro", "allegro_low_friction.urdf"))
    parser.add_argument("--urdf_shadow", type=str, default=os.path.join(TASK_PATH, "rsc", "shadow", "shadowhand_large.urdf"))
    parser.add_argument("--urdf_leap", type=str, default=os.path.join(TASK_PATH, "rsc", "leap_hand", "leap_hand_right_envusing.urdf"))
    parser.add_argument("--urdf_inspire", type=str, default=os.path.join(TASK_PATH, "rsc", "inspire_hand", "inspire_hand_right_envusing.urdf"))
    parser.add_argument("--urdf_ability", type=str, default=os.path.join(TASK_PATH, "rsc", "ability_hand", "ability_hand_right.urdf"))
    parser.add_argument("--urdf_schunk", type=str, default=os.path.join(TASK_PATH, "rsc", "schunk_hand", "schunk_svh_hand_right_envusing.urdf"))
    parser.add_argument("--urdf_brainco", type=str, default=os.path.join(TASK_PATH, "rsc", "brainco", "revo2_right_hand_envusing.urdf"))
    parser.add_argument("--urdf_barrett", type=str, default=os.path.join(TASK_PATH, "rsc", "barrett_hand", "bhand_model_right_envusing.urdf"))

    parser.add_argument('--hand', type=str, default='LEAP',
                        help=f'Hand type to visualize. Available options: {[h.name for h in HandType]}')
    parser.add_argument('--save', action='store_true', help='Also save a PNG to disk')
    parser.add_argument('--show_frames', action='store_true', help='Also show joint local coordinate frames (XYZ triads)')
    parser.add_argument('--no_axis', dest='show_axis', action='store_false', help='Hide joint rotation axis arrows')
    parser.set_defaults(show_axis=True)
    # Compatibility with older Python versions: argparse.BooleanOptionalAction is unavailable
    parser.add_argument('--show_mesh', dest='show_mesh', action='store_true',
                        help='Enable rendering hand mesh (URDF <visual> geometry)')
    parser.add_argument('--no_show_mesh', dest='show_mesh', action='store_false',
                        help='Disable rendering hand mesh')
    parser.set_defaults(show_mesh=True)
    parser.add_argument('--show_topology', dest='show_topology', action='store_true',
                        help='Enable rendering hand topology graph')
    parser.add_argument('--no_show_topology', dest='show_topology', action='store_false',
                        help='Disable rendering hand topology graph')
    parser.set_defaults(show_topology=True)
    parser.add_argument('--mesh_max_faces', type=int, default=1000, help='Max faces per visual mesh for fast display')
    parser.add_argument('--mesh_alpha', type=float, default=0.25, help='Mesh transparency (0~1). Larger = more solid')

    # Arrow linewidth / head size
    parser.add_argument('--axis_arrow_lw', type=float, default=3.5, help='Joint axis arrow linewidth')
    parser.add_argument('--axis_arrow_ms', type=float, default=10.0, help='Joint axis arrow head size (mutation_scale)')
    parser.add_argument('--frame_arrow_lw', type=float, default=2.0, help='Frame triad arrow linewidth')
    parser.add_argument('--frame_arrow_ms', type=float, default=15.0, help='Frame triad arrow head size (mutation_scale)')

    # Randomize semantic axis length for visualization comparisons
    parser.add_argument('--random_semantic_axis_length', action='store_true',
                        help='Randomize semantic axis arrow length to a positive value per joint (visualization only)')
    parser.add_argument('--semantic_axis_length_min', type=float, default=0.4,
                        help='Min multiplier for semantic axis arrow length when randomization enabled (must be > 0)')
    parser.add_argument('--semantic_axis_length_max', type=float, default=2.0,
                        help='Max multiplier for semantic axis arrow length when randomization enabled (must be > 0)')
    parser.add_argument('--semantic_axis_length_seed', type=int, default=None,
                        help='Optional RNG seed for semantic axis arrow length randomization')

    # Node axis activation mask: color each node sphere in three bands (top/mid/bottom = FLEX/ROT/ABD)
    parser.add_argument('--node_axis_mask', action='store_true',
                        help='Color each node sphere by axis activation mask (top/mid/bottom = FLEX/ROT/ABD; inactive = gray)')
    
    
    args = parser.parse_args()
    urdf_allegro = args.urdf_allegro
    urdf_shadow = args.urdf_shadow
    urdf_leap = args.urdf_leap
    urdf_inspire = args.urdf_inspire
    urdf_schunk = args.urdf_schunk
    urdf_ability = args.urdf_ability
    urdf_brainco = args.urdf_brainco
    urdf_barrett = args.urdf_barrett
    try:
        hand_type_enum = HandType[args.hand.upper()]
    except KeyError:
        print(f"Error: Invalid hand type '{args.hand}'.")
        print(f"Available options: {[h.name for h in HandType]}")
        exit()

    # 1. Load configuration for the selected hand type
    data_dict = get_data_dict(
        TRAIN_LEAP=True,
        TRAIN_ALLEGRO=True,
        TRAIN_SHADOW=True,
        TRAIN_SCHUNK=True,
        TRAIN_INSPIRE=True,
        TRAIN_ABILITY=True,
        TRAIN_BRAINCO=True,
        TRAIN_BARRETT=True,
    )
    hand_config = data_dict.get(hand_type_enum.value)
    if not hand_config:
        print(f"Error: Configuration for hand type '{hand_type_enum.name}' not found.")
        exit()

    if hand_type_enum == HandType.ALLEGRO:
        urdf_file = urdf_allegro
    elif hand_type_enum == HandType.SHADOW:
        urdf_file = urdf_shadow
    elif hand_type_enum == HandType.LEAP:
        urdf_file = urdf_leap
    elif hand_type_enum == HandType.INSPIRE:
        urdf_file = urdf_inspire
    elif hand_type_enum == HandType.SCHUNK:
        urdf_file = urdf_schunk
    elif hand_type_enum == HandType.ABILITY:
        urdf_file = urdf_ability
    elif hand_type_enum == HandType.BRAINCO:
        urdf_file = urdf_brainco
    elif hand_type_enum == HandType.BARRETT:
        urdf_file = urdf_barrett
    else:
        print(f"Error: URDF file for hand type '{hand_type_enum.name}' not specified.")
        exit()
    # Get all joint names from build_node_to_joint_sequence
    all_hand_joints = build_node_to_joint_sequence().get(hand_type_enum.value, [])
    # Never show wrist/root (node 0)
    node_ids = list(range(1, len(all_hand_joints)))

    test_joint_names = [joint for node_id in node_ids for joint in all_hand_joints[node_id]]

    # ===== Semantic type -> color (per your specification) =====
    SEMANTIC_COLORS = {
        NodeTypeSemantic.WRIST: "#D30101",
        NodeTypeSemantic.PROXIMAL: "#FFC000",
        NodeTypeSemantic.MIDDLE: "#96D852",
        NodeTypeSemantic.DISTAL: "#11BB75",
        NodeTypeSemantic.FINGERTIP: "#0271C2",
    }

    other_key = getattr(NodeTypeSemantic, "METACARPAL", None)
    if other_key is not None:
        SEMANTIC_COLORS[other_key] = "#7f7f7f"

    # ===== Joint axis semantics (FLEX/ABD/ROT) -> color =====
    JOINT_AXIS_COLORS = {
        # More vivid, high-saturation colors (used for both axis arrows and node-mask spheres)
        FLEX: "#FF1F1F",   # vivid red
        ABD: "#FFD400",    # vivid yellow
        ROT: "#007BFF",    # vivid blue
    }

    joint_semantic_map = build_joint_semantic_map().get(hand_type_enum.value, {})

    node_semantic_mapping = hand_config.get("node_semantic_mapping")
    joint_to_semantic = {}
    if node_semantic_mapping is not None:
        if len(node_semantic_mapping) != len(all_hand_joints):
            raise ValueError(
                f"node_semantic_mapping length ({len(node_semantic_mapping)}) != node2seq length ({len(all_hand_joints)})"
            )
        for node_id, joint_names in enumerate(all_hand_joints):
            sem = node_semantic_mapping[node_id]
            if not isinstance(sem, NodeTypeSemantic):
                sem = NodeTypeSemantic(int(sem))
            for jn in joint_names:
                joint_to_semantic[jn] = sem

    print(f"--- Visualizing {hand_type_enum.name} Hand ---")
    print(f"URDF file: {urdf_file}")

    try:
        # Use "world" as the reference coordinate frame
        extractor = URDFJointPoseExtractor(urdf_path=urdf_file, base_link="world")
    except FileNotFoundError:
        print(f"Error: URDF file not found, please check the path: {urdf_file}")
        exit()
    except ValueError as e:
        print(f"Parser initialization failed: {e}")
        exit()

    # 3. Define a test hand pose
    joint_names_from_urdf = [j.name for j in extractor.robot.joints if j.type == 'revolute']
    test_joint_angles = {name: 0.0 for name in joint_names_from_urdf} 

    # ABD:
    # test_joint_angles["right_hand_Finger_Spread"]=-0.5
    
    # FLEX
    test_joint_angles["right_hand_Index_Finger_Proximal"] = 0
    
    # 4. Get all joint data at the given pose from the parser
    try:
        all_link_poses, all_joint_poses = extractor.get_all_poses(test_joint_angles, joint_pose_frame="child")
        all_joint_axes = extractor.get_all_joint_axes_in_base(test_joint_angles) if args.show_axis else {}
    except Exception as e:
        print(f"Error: an exception occurred while getting all joint poses: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # 5. Extract frames for visualization
    extracted_frames = []
    for joint_name in test_joint_names:
        if joint_name in all_joint_poses:
            pose_matrix = all_joint_poses[joint_name]
            extracted_frames.append({
                'name': joint_name,
                'position': pose_matrix[:3, 3],
                'rotation': pose_matrix[:3, :3]
            })
    
    if not extracted_frames:
        print("Error: failed to extract any joint data, visualization cannot continue.")
        exit()

    # 6. Create the visualization instance
    viz = FrameVisualizer(
        figsize=(12, 10),
        axis_arrow_lw=args.axis_arrow_lw,
        axis_arrow_ms=args.axis_arrow_ms,
        frame_arrow_lw=args.frame_arrow_lw,
        frame_arrow_ms=args.frame_arrow_ms,
    )

    # 6.1 Render the full hand mesh (URDF visual)
    # trimesh is relatively heavy, so we import it lazily to avoid blocking --help and similar flows
    trimesh = None
    if args.show_mesh:
        try:
            import trimesh as _trimesh  # type: ignore
            trimesh = _trimesh
        except Exception:
            trimesh = None

    def _safe_load_mesh(mesh_path: str):
        if trimesh is None:
            return None
        try:
            loaded = trimesh.load(mesh_path, force=None)
        except Exception:
            return None
        if isinstance(loaded, trimesh.Scene):
            geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geoms:
                return None
            return trimesh.util.concatenate(geoms)
        if isinstance(loaded, trimesh.Trimesh):
            return loaded
        return None

    def _origin_to_T(origin):
        if origin is None:
            return np.eye(4)
        xyz = np.array(origin.xyz) if getattr(origin, 'xyz', None) is not None else np.zeros(3)
        rpy = np.array(origin.rpy) if getattr(origin, 'rpy', None) is not None else np.zeros(3)
        T = np.eye(4)
        T[:3, 3] = xyz
        from scipy.spatial.transform import Rotation as _R
        T[:3, :3] = _R.from_euler('xyz', rpy, degrees=False).as_matrix()
        return T

    def _apply_T(vertices: np.ndarray, T: np.ndarray):
        v = np.asarray(vertices, dtype=np.float64)
        ones = np.ones((v.shape[0], 1), dtype=np.float64)
        vh = np.concatenate([v, ones], axis=1)
        out = (T @ vh.T).T
        return out[:, :3]

    if args.show_mesh and trimesh is None:
        print("[mesh] trimesh is unavailable, skipping mesh rendering")

    if args.show_mesh and trimesh is not None:
        urdf_dir = os.path.dirname(os.path.realpath(urdf_file))
        # Use link_poses to place each link's visual mesh under world/base
        for link in getattr(extractor.robot, 'links', []):
            T_base_link = all_link_poses.get(link.name)
            if T_base_link is None:
                continue
            visuals = getattr(link, 'visuals', None) or []
            for vis in visuals:
                geom = getattr(vis, 'geometry', None)
                if geom is None:
                    continue

                # Compute base->visual
                T_link_vis = _origin_to_T(getattr(vis, 'origin', None))
                T_base_vis = T_base_link @ T_link_vis

                mesh = None
                scale = None
                # urdf_parser_py geometry types: Mesh/Box/Cylinder/Sphere
                if geom.__class__.__name__.lower() == 'mesh':
                    filename = getattr(geom, 'filename', None)
                    scale = getattr(geom, 'scale', None)
                    if not filename:
                        continue
                    # Handle package:// and relative paths
                    if filename.startswith('package://'):
                        rel = filename.split('package://', 1)[1]
                        rel2 = rel.split('/', 1)[1] if '/' in rel else rel

                        candidates = [
                            os.path.join(urdf_dir, rel),
                            os.path.join(urdf_dir, rel2),
                            os.path.join(TASK_PATH, rel),
                            os.path.join(TASK_PATH, rel2),
                            os.path.join(TASK_PATH, 'rsc', rel),
                            os.path.join(TASK_PATH, 'rsc', rel2),
                        ]
                        mesh_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
                    elif os.path.isabs(filename):
                        mesh_path = filename
                    else:
                        mesh_path = os.path.join(urdf_dir, filename)
                    mesh = _safe_load_mesh(mesh_path)
                elif geom.__class__.__name__.lower() == 'box':
                    size = np.array(getattr(geom, 'size', [0.01, 0.01, 0.01]), dtype=np.float64)
                    mesh = trimesh.creation.box(extents=size)
                elif geom.__class__.__name__.lower() == 'cylinder':
                    radius = float(getattr(geom, 'radius', 0.01))
                    length = float(getattr(geom, 'length', 0.01))
                    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=24)
                elif geom.__class__.__name__.lower() == 'sphere':
                    radius = float(getattr(geom, 'radius', 0.01))
                    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)

                if mesh is None:
                    continue

                if scale is not None:
                    try:
                        s = np.array(scale, dtype=np.float64).reshape(3)
                        mesh = mesh.copy()
                        mesh.apply_scale(s)
                    except Exception:
                        pass

                # Downsample faces to keep interaction smooth
                faces = mesh.faces
                vertices = mesh.vertices
                if args.mesh_max_faces is not None and len(faces) > int(args.mesh_max_faces):
                    step = max(1, int(len(faces) // int(args.mesh_max_faces)))
                    faces = faces[::step]

                v_world = _apply_T(vertices, T_base_vis)
                viz.add_mesh(v_world, faces, color=(0.65, 0.65, 0.65), alpha=float(args.mesh_alpha))

        viz.set_mesh_visible(True)
    else:
        viz.set_mesh_visible(False)
    
    # 7. Joint local coordinate frames (XYZ triads): always build them for the interaction toggles; initial visibility is controlled by args
    viz.add_frames(extracted_frames, show_labels=False)
    viz.set_frames_visible(args.show_frames)

    # 7.1 Joint axis directions (black arrows): always build them for the interaction toggles; initial visibility is controlled by args
    # Randomize semantic axis length: sample a positive length multiplier for each axis arrow
    if args.random_semantic_axis_length:
        len_min = float(args.semantic_axis_length_min)
        len_max = float(args.semantic_axis_length_max)
        if len_min <= 0.0 or len_max <= 0.0:
            raise ValueError("semantic_axis_length_min/max must be > 0 when --random_semantic_axis_length is enabled")
        if len_max < len_min:
            len_min, len_max = len_max, len_min
        _axis_len_rng = np.random.default_rng(args.semantic_axis_length_seed)
    else:
        len_min = len_max = 1.0
        _axis_len_rng = None

    for joint_name in test_joint_names:
        info = all_joint_axes.get(joint_name)
        if info is None:
            continue

        # Color according to joint_semantic_map and use sign to correct the semantic positive direction
        axis_dir = np.asarray(info["axis"], dtype=np.float64)
        # No active semantic axis (missing from joint_semantic_map or mapped to None) => gray arrow
        axis_color = "#9E9E9E"
        sem = joint_semantic_map.get(joint_name)
        if isinstance(sem, (list, tuple)) and len(sem) == 2:
            axis_type = int(sem[0])
            axis_sign = float(sem[1])
            axis_dir = axis_dir * axis_sign
            axis_color = JOINT_AXIS_COLORS.get(axis_type, axis_color)
        else:
            # sem is None or not in the expected format: keep it gray
            axis_color = "#9E9E9E"

        viz.add_axis_vector(
            name=joint_name,
            position=info["position"],
            direction=axis_dir,
            length=(
                FrameVisualizer.AXIS_LENGTH
                * 1.2
                * (float(_axis_len_rng.uniform(len_min, len_max)) if _axis_len_rng is not None else 1.0)
            ),
            show_label=False,
            color=axis_color,
        )
    viz.set_axis_visible(args.show_axis)

    # 8. Draw hand nodes (one small sphere per node; colored by node_semantic_mapping; fingertip nodes supported)
    # Use plot_surface to draw a low-resolution sphere; it looks more like a real sphere than a single scatter point.
    node_sphere_radius = 0.008


    def _add_sphere(center_xyz: np.ndarray, *, radius: float, color: str, alpha: float = 0.9, resolution: int = 30):
        c = np.asarray(center_xyz, dtype=np.float64).reshape(3)
        u = np.linspace(0.0, 2.0 * np.pi, int(resolution))
        v = np.linspace(0.0, np.pi, int(resolution))
        x = c[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = c[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = c[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        viz.ax.plot_surface(
            x,
            y,
            z,
            color=color,
            alpha=float(alpha),
            linewidth=0.0,
            antialiased=True,
            shade=True,
            zorder=30,
        )

    def _add_sphere_band(
        center_xyz: np.ndarray,
        *,
        radius: float,
        color: str,
        v0: float,
        v1: float,
        alpha: float = 0.95,
        resolution_u: int = 24,
        resolution_v: int = 10,
    ):
        """Draw a latitude band of a sphere (used for segmented coloring).

        v is the polar angle: 0 at the north pole (top), pi at the south pole (bottom).
        """
        c = np.asarray(center_xyz, dtype=np.float64).reshape(3)
        v0 = float(max(0.0, min(np.pi, v0)))
        v1 = float(max(0.0, min(np.pi, v1)))
        if v1 <= v0:
            return
        u = np.linspace(0.0, 2.0 * np.pi, int(resolution_u))
        v = np.linspace(v0, v1, int(resolution_v))
        x = c[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = c[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = c[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        viz.ax.plot_surface(
            x,
            y,
            z,
            color=color,
            alpha=float(alpha),
            linewidth=0.0,
            antialiased=True,
            shade=True,
            zorder=31,
        )

    def _add_axis_mask_sphere(center_xyz: np.ndarray, *, radius: float, flex_on: bool, rot_on: bool, abd_on: bool):
        inactive_color = "#9E9E9E"
        flex_color = JOINT_AXIS_COLORS.get(FLEX, inactive_color) if flex_on else inactive_color
        rot_color = JOINT_AXIS_COLORS.get(ROT, inactive_color) if rot_on else inactive_color
        abd_color = JOINT_AXIS_COLORS.get(ABD, inactive_color) if abd_on else inactive_color

        # top/mid/bottom represent FLEX/ROT/ABD respectively.
        # To make the blue region smaller, narrow the middle ROT (blue) band.
        flex_ratio = 0.42
        rot_ratio = 0.16
        v1 = np.pi * flex_ratio
        v2 = np.pi * (flex_ratio + rot_ratio)

        _add_sphere_band(center_xyz, radius=radius, color=flex_color, v0=0.0, v1=v1)
        _add_sphere_band(center_xyz, radius=radius, color=rot_color, v0=v1, v1=v2)
        _add_sphere_band(center_xyz, radius=radius, color=abd_color, v0=v2, v1=np.pi)

    # Prefer body_parts from constants to find a representative pose for each node.
    # body_parts may contain either joint names or link names (for example, *tip* links).
    body_parts = hand_config.get("body_parts")
    if body_parts is not None and len(body_parts) != len(all_hand_joints):
        body_parts = None

    # Position of wrist/root (node 0): used for the wrist sphere and (0, *) topology edges
    wrist_pos = None
    if body_parts is not None and len(body_parts) > 0:
        wrist_name = body_parts[0]
        T0 = all_joint_poses.get(wrist_name)
        if T0 is None:
            T0 = all_link_poses.get(wrist_name)
        if T0 is not None:
            wrist_pos = T0[:3, 3]

    node_positions = []
    for node_id in node_ids:
        joint_names = all_hand_joints[node_id]
        # if joint_names == "z_joint":
        # print(joint_names)
        T = None

        if body_parts is not None:
            name = body_parts[node_id]
            if name in all_joint_poses:
                T = all_joint_poses[name]
            elif name in all_link_poses:
                T = all_link_poses[name]

        # Fallback: use the last joint pose of this node
        if T is None and joint_names:
            for name in reversed(joint_names):
                if name in all_joint_poses:
                    T = all_joint_poses[name]
                    break

        if T is None:
            continue

        pos = T[:3, 3]
        
        if args.hand.upper()=='LEAP':
            pos[2]=-0.02
            if node_semantic_mapping[node_id] in[ NodeTypeSemantic.FINGERTIP,NodeTypeSemantic.MIDDLE] and node_id!=14:
                pos[1]+=0.015
            if node_semantic_mapping[node_id] in[NodeTypeSemantic.MIDDLE] and node_id!=14:
                pos[0]+=0.015
        node_positions.append(pos)

        if args.node_axis_mask:
            flex_on = False
            rot_on = False
            abd_on = False
            for jn in all_hand_joints[node_id]:
                sem = joint_semantic_map.get(jn)
                if isinstance(sem, (list, tuple)) and len(sem) == 2:
                    axis_type = int(sem[0])
                    if axis_type == FLEX:
                        flex_on = True
                    elif axis_type == ROT:
                        rot_on = True
                    elif axis_type == ABD:
                        abd_on = True
            _add_axis_mask_sphere(pos, radius=node_sphere_radius, flex_on=flex_on, rot_on=rot_on, abd_on=abd_on)
        else:
            sem = node_semantic_mapping[node_id] if node_semantic_mapping is not None else None
            if sem is not None and not isinstance(sem, NodeTypeSemantic):
                sem = NodeTypeSemantic(int(sem))
            color = SEMANTIC_COLORS.get(sem, "#7f7f7f")
            _add_sphere(pos, radius=node_sphere_radius, color=color, alpha=0.95, resolution=10)

    # 8.1 Display the wrist/root node (node 0)
    if wrist_pos is None:
        wrist_pos = np.mean(node_positions, axis=0) if node_positions else np.zeros(3)
    if args.hand.upper()=='LEAP':
        wrist_pos[2]=0
        wrist_pos[1]-=0.03
        wrist_pos[0]-=0.07
    if args.node_axis_mask:
        _add_axis_mask_sphere(wrist_pos, radius=node_sphere_radius, flex_on=False, rot_on=False, abd_on=False)
    else:
        _add_sphere(
            wrist_pos,
            radius=node_sphere_radius,
            color=SEMANTIC_COLORS.get(NodeTypeSemantic.WRIST, "#D30101"),
            alpha=0.95,
            resolution=10,
        )

    # 9. Automatically set the view limits
    _pos_for_center = list(node_positions)
    if wrist_pos is not None:
        _pos_for_center.append(wrist_pos)
    center_point = np.mean(_pos_for_center, axis=0) if _pos_for_center else np.zeros(3)
    viz.set_limits(center=center_point, radius=0.15)

    # 9.1 Draw topology connections (add edges between adjacent nodes)
    # Note: the field name used by each hand type in constants.get_data_dict() is connection, not parent_map.
    if args.show_topology:
        topo_edges = hand_config.get("connection")
        if topo_edges:
            # node_ids start from 1, and node_positions correspond to them one-to-one.
            node_id_to_pos_map = {node_id: pos for node_id, pos in zip(node_ids, node_positions)}

            # Add the wrist/root (node 0) position so we can draw (0, *) edges
            if 0 not in node_id_to_pos_map:
                node_id_to_pos_map[0] = wrist_pos if wrist_pos is not None else center_point

            for a, b in topo_edges:
                if a in node_id_to_pos_map and b in node_id_to_pos_map:
                    p1 = node_id_to_pos_map[a]
                    p2 = node_id_to_pos_map[b]
                    viz.ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        color="#B2B2B2",
                        linewidth=2.5,
                        zorder=5,
                    )
        else:
            print("[Topology] 'connection' not found in hand_config. Skipping topology drawing.")
    
    # 10. Display and save the result
    title = f"{hand_type_enum.name} Hand Visualization"
    # Legend: axis semantics (FLEX/ABD/ROT) + node semantics (PROX/MID/DIST/TIP)
    legend_handles = [
        Line2D([0], [0], color=JOINT_AXIS_COLORS[FLEX], lw=3, label="axis: FLEX"),
        Line2D([0], [0], color=JOINT_AXIS_COLORS[ABD], lw=3, label="axis: ABD"),
        Line2D([0], [0], color=JOINT_AXIS_COLORS[ROT], lw=3, label="axis: ROT"),
        Line2D([0], [0], color="#9E9E9E", lw=3, label="axis: inactive"),
        Patch(facecolor=(0.65, 0.65, 0.65, float(args.mesh_alpha)), edgecolor='none', label='mesh'),
        Line2D([0], [0], marker='o', linestyle='None', markersize=7, markerfacecolor=JOINT_AXIS_COLORS[FLEX], markeredgecolor='none', label="node band(top): FLEX on"),
        Line2D([0], [0], marker='o', linestyle='None', markersize=7, markerfacecolor=JOINT_AXIS_COLORS[ROT], markeredgecolor='none', label="node band(mid): ROT on"),
        Line2D([0], [0], marker='o', linestyle='None', markersize=7, markerfacecolor=JOINT_AXIS_COLORS[ABD], markeredgecolor='none', label="node band(bot): ABD on"),
        Line2D([0], [0], marker='o', linestyle='None', markersize=7, markerfacecolor="#9E9E9E", markeredgecolor='none', label="node band: inactive"),
        Line2D([0], [0], marker='o', linestyle='None', markersize=7, markerfacecolor=SEMANTIC_COLORS[NodeTypeSemantic.PROXIMAL], markeredgecolor='none', label="node: PROX"),
        Line2D([0], [0], marker='o', linestyle='None', markersize=7, markerfacecolor=SEMANTIC_COLORS[NodeTypeSemantic.MIDDLE], markeredgecolor='none', label="node: MID"),
        Line2D([0], [0], marker='o', linestyle='None', markersize=7, markerfacecolor=SEMANTIC_COLORS[NodeTypeSemantic.DISTAL], markeredgecolor='none', label="node: DIST"),
        Line2D([0], [0], marker='o', linestyle='None', markersize=7, markerfacecolor=SEMANTIC_COLORS[NodeTypeSemantic.FINGERTIP], markeredgecolor='none', label="node: TIP"),
    ]

    other_key = getattr(NodeTypeSemantic, "METACARPAL", None)
    if other_key is not None:
        legend_handles.append(
            Line2D([0], [0], marker='o', linestyle='None', markersize=7, markerfacecolor=SEMANTIC_COLORS[other_key], markeredgecolor='none', label="node: METACARPAL")
        )

    viz.ax.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.9)
    viz.show(title=title, legend=False)
    if args.save:
        viz.save(f"{hand_type_enum.name.lower()}_hand_visualization.png")
