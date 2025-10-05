import json
import os
import glob
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from vdbfusion import VDBVolume
import tqdm
import argparse  # For command-line interface

# Enable EXR reading in OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def pose_to_matrix(pose):
    """
    Converts a 7-element pose [qx, qy, qz, qw, tx, ty, tz]
    into a 4x4 transformation matrix (camera-to-world).
    """
    q = np.array(pose[:4])
    t = np.array(pose[4:])

    # Convert quaternion to rotation matrix
    r = R.from_quat([q[0], q[1], q[2], q[3]])
    R_mat = r.as_matrix()

    # Flip along Z-axis (to align with Open3D coordinate conventions)
    F_z = np.diag([1, 1, -1, 1])

    # Combine into a single 4x4 transform
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t

    return T @ F_z


def update_K(K, orig_size=(1280, 720), new_size=(640, 360)):
    """Adjusts the intrinsic matrix for resized images."""
    sx = new_size[0] / orig_size[0]
    sy = new_size[1] / orig_size[1]
    S = np.diag([sx, sy, 1.0])
    new_K = S @ K
    return new_K


def depth_to_world_xyz(depth, K, T=np.eye(4), max_depth=2.0, min_depth=0.25):
    """
    Converts a depth map to a per-pixel XYZ image in world coordinates.
    """
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Convert depth to camera coordinates
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    y = -y  # Flip Y-axis to align with standard coordinates
    z = depth

    xyz_cam = np.stack((x, y, z), axis=-1)

    # Mask invalid depths
    xyz_cam = np.where((xyz_cam[..., 2:3] < max_depth) &
                       (xyz_cam[..., 2:3] > min_depth),
                       xyz_cam, [0, 0, 0])

    # Homogeneous coordinates
    xyz_hom = np.concatenate([xyz_cam, np.ones_like(z[..., None])], axis=-1)

    # Apply extrinsic transform (camera → world)
    xyz_world = np.einsum('ij,hwj->hwi', T, xyz_hom)[..., :3]

    return xyz_world, xyz_cam


def build_normal_xyz(xyz, norm_factor=0.25, ksize=3):
    """
    Computes per-pixel surface normals using Scharr gradients.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    # Compute Scharr derivatives
    Sxx = cv2.Scharr(x.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0 / norm_factor)
    Sxy = cv2.Scharr(x.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0 / norm_factor)
    Syx = cv2.Scharr(y.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0 / norm_factor)
    Syy = cv2.Scharr(y.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0 / norm_factor)
    Szx = cv2.Scharr(z.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0 / norm_factor)
    Szy = cv2.Scharr(z.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0 / norm_factor)

    # Cross product to get normals
    normal = -np.dstack((Syx * Szy - Szx * Syy,
                         Szx * Sxy - Szy * Sxx,
                         Sxx * Syy - Syx * Sxy))

    # Normalize
    n = np.linalg.norm(normal, axis=2) + 1e-10
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    return normal


class DatasetIterator:
    """
    Iterates through Record3D dataset folders:
    loads color images, EXR depth maps, intrinsics, and poses.
    """

    def __init__(self, data_path, width=192, height=256, max_depth=1.5, min_depth=0.25):
        self.glob_path = os.path.join(data_path, "rgb/*.jpg")
        self.data = sorted(glob.glob(self.glob_path),
                           key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
        self.metadata_path = os.path.join(data_path, "metadata.json")

        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.min_depth = min_depth
        self._index = 0
        self._reset()

    def _reset(self):
        """Reset the iterator index."""
        self._index = 0

    def __iter__(self):
        self._reset()
        return self

    def __next__(self):
        """Get next frame."""
        if self._index >= len(self.data):
            raise StopIteration
        ret_dict = self.get_data(self._index)
        self._index += 1
        return ret_dict

    def __getitem__(self, idx):
        """Get frame by index."""
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range")
        return self.get_data(idx)

    def __len__(self):
        return len(self.data)

    def get_data(self, idx):
        """Load one frame: RGB, depth, intrinsics, pose."""
        rgb_path = self.data[self._index]
        exr_path = rgb_path.replace("rgb", "depth").replace(".jpg", ".exr")

        # Intrinsics
        pfic_color = self.metadata['perFrameIntrinsicCoeffs'][idx]
        K = np.eye(3)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = pfic_color[0:4]

        # Pose (Quaternion + translation)
        pose = self.metadata['poses'][idx]
        T = pose_to_matrix(pose)

        # Load color + depth
        color = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)[..., 2]

        # Resize and update intrinsics
        K_new = update_K(K, color.shape[0:2], (self.height, self.width))
        color = cv2.resize(color, (self.width, self.height), cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (self.width, self.height), cv2.INTER_NEAREST)

        # Compute 3D coordinates and normals
        xyz_world, xyz_ego = depth_to_world_xyz(depth, K_new, T,
                                                max_depth=self.max_depth,
                                                min_depth=self.min_depth)
        normals_ego = build_normal_xyz(xyz_ego)
        normals_world = build_normal_xyz(xyz_world)

        return {
            "color": color,
            "xyz_ego": xyz_ego,
            "xyz_world": xyz_world,
            "normals_ego": normals_ego,
            "normals_world": normals_world,
            "K": K_new,
            "T": T
        }


def main():
    """
    Main reconstruction entry point.
    Loads Record3D frames, integrates via VDB, and visualizes the result.
    """
    parser = argparse.ArgumentParser(description="Record3D → Open3D reconstruction pipeline.")
    parser.add_argument(
        "--data_path", type=str,
        default="/home/hannes/Downloads/Robot_Record3D-20251004T153553Z-1-001/Robot_Record3D/",
        help="Path to Record3D dataset folder (default: example path)."
    )
    parser.add_argument("--width", type=int, default=192, help="Output image width.")
    parser.add_argument("--height", type=int, default=256, help="Output image height.")
    parser.add_argument("--max_depth", type=float, default=1.5, help="Max depth threshold (m).")
    parser.add_argument("--min_depth", type=float, default=0.25, help="Min depth threshold (m).")
    parser.add_argument("--voxel_size", type=float, default=0.005, help="VDB fusion voxel size.")
    parser.add_argument("--sdf_trunc", type=float, default=0.8, help="Truncation distance for VDB fusion.")
    parser.add_argument("--space_carving", action="store_true",
                        help="Enable space carving in VDB fusion (disabled by default).")
    args = parser.parse_args()

    # Initialize global map and VDB volume
    map_cloud = o3d.geometry.PointCloud()
    vdb_volume = VDBVolume(
        voxel_size=args.voxel_size,
        sdf_trunc=args.sdf_trunc,
        space_carving=args.space_carving
    )

    # Load dataset
    dataset = DatasetIterator(args.data_path, width=args.width, height=args.height,
                              max_depth=args.max_depth, min_depth=args.min_depth)

    # Process frames
    for idx, dataItem in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dataItem["xyz_world"].reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(dataItem["color"].reshape(-1, 3) / 255.0)

        vdb_volume.integrate(dataItem["xyz_world"].reshape(-1, 3), dataItem["T"])
        map_cloud += pcd

        cv2.imshow("color", dataItem["color"])
        cv2.imshow("normals_ego", np.uint8(255 * (dataItem["normals_ego"] + 1) / 2))
        cv2.imshow("normals_world", np.uint8(255 * (dataItem["normals_world"] + 1) / 2))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            o3d.visualization.draw_geometries([pcd])
        print(f"Processed frame {idx}")

    # Extract mesh from VDB volume
    vertices, triangles = vdb_volume.extract_triangle_mesh(fill_holes=True, min_weight=5.0)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    # Colorize mesh by nearest color from map_cloud
    pcd_tree = o3d.geometry.KDTreeFlann(map_cloud)
    query_points = np.asarray(mesh.vertices)
    colors = []
    for i in tqdm.tqdm(range(query_points.shape[0]), desc="Colorizing mesh"):
        [k, idx, dist] = pcd_tree.search_knn_vector_3d(query_points[i:i+1, :].T, 1)
        colors.append(np.asarray(map_cloud.colors)[idx])
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(colors))

    o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    main()
