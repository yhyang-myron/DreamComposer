import numpy as np
import torch

def T_to_R_t(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t

def R_t_to_T(R, t):
    if torch.is_tensor(R):
        T = torch.eye(4, device=R.device, dtype=R.dtype)
    else:
        T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def T_to_pose(T):
    """
    Convert T to pose.
    """
    rot = np.transpose(T[:3, :3])  # (3, 3)
    trans = -np.dot(rot, T[:3, 3:])  # (3, 1)
    pose = np.concatenate((rot, trans), axis=-1)  # (3, 4)
    pose = np.concatenate((pose, [[0, 0, 0, 1]]))
    return pose

def pose_to_T(pose):
    """
    Convert pose to T.
    """
    rot = np.transpose(pose[:3, :3])  # (3, 3)
    trans = -np.dot(rot, pose[:3, 3:])  # (3, 1)
    T = np.concatenate((rot, trans), axis=-1)  # (3, 4)
    T = np.concatenate((T, [[0, 0, 0, 1]]))
    return T

def T_blender_to_pinhole(T_blender):
    """
    Converts blender T to pinhole. Compared to the pinhole model, Blender has
    the Y and Z axes flipped, while the X axis is the same.

    - Pinhole/OpenCV/COLMAP/Ours
        - +X: Right
        - +Y: Down
        - +Z: The view direction, pointing forward and away from the camera
    - Blender/OpenGL/Nerfstudio
        - +X: Right
        - +Y: Up
        - +Z: The negative view direction, pointing back and away from the camera
        - -Z: The view direction
    """

    R_b2p = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    R_blender, t_blender = T_to_R_t(T_blender)
    R = R_b2p @ R_blender
    t = t_blender @ R_b2p
    T = R_t_to_T(R, t)

    return T

def pose_opengl_to_opencv(pose_opengl):
    """
    Convert pose from OpenGL convention to OpenCV convention.

    - OpenCV
        - +X: Right
        - +Y: Down
        - +Z: The view direction, pointing forward and away from the camera
        - Used in: OpenCV, COLMAP, camtools default
    - OpenGL
        - +X: Right
        - +Y: Up
        - +Z: The negative view direction, pointing back and away from the camera
        - -Z: The view direction
        - Used in: OpenGL, Blender, Nerfstudio
    """
    pose = np.copy(pose_opengl)
    pose[0:3, 1:3] *= -1
    return pose


def pose_opencv_to_opengl(pose):
    """
    Convert pose from OpenCV convention to OpenGL convention.

    - OpenCV
        - +X: Right
        - +Y: Down
        - +Z: The view direction, pointing forward and away from the camera
        - Used in: OpenCV, COLMAP, camtools default
    - OpenGL
        - +X: Right
        - +Y: Up
        - +Z: The negative view direction, pointing back and away from the camera
        - -Z: The view direction
        - Used in: OpenGL, Blender, Nerfstudio
    """
    pose_opengl = np.copy(pose)
    pose_opengl[0:3, 1:3] *= -1
    return pose_opengl


def euler_to_R(yaw, pitch, roll):
    """
    Convert Euler angles to rotation matrix. Given a unit vector x, R @ x is x
    rotated by applying yaw, pitch, and roll consecutively. Ref:
    https://en.wikipedia.org/wiki/Euler_angles
    Args:
        yaw (float): Rotation around the z-axis (from x-axis to y-axis).
        pitch (float): Rotation around the y-axis (from z-axis to x-axis).
        roll (float): Rotation around the x-axis (from y-axis to z-axis).
    Returns:
        Rotation matrix R of shape (3, 3).
    """
    sin_y = np.sin(yaw)
    cos_y = np.cos(yaw)
    sin_p = np.sin(pitch)
    cos_p = np.cos(pitch)
    sin_r = np.sin(roll)
    cos_r = np.cos(roll)
    R = np.array(
        [
            [
                cos_y * cos_p,
                cos_y * sin_p * sin_r - sin_y * cos_r,
                cos_y * sin_p * cos_r + sin_y * sin_r,
            ],
            [
                sin_y * cos_p,
                sin_y * sin_p * sin_r + cos_y * cos_r,
                sin_y * sin_p * cos_r - cos_y * sin_r,
            ],
            [
                -sin_p,
                cos_p * sin_r,
                cos_p * cos_r,
            ],
        ]
    )
    return R

def spherical_to_T_towards_origin(radius, theta, phi):
    """
    Convert spherical coordinates (ISO convention) to T, where the cameras looks
    at the origin from a distance (radius), and the camera up direction alines
    with the z-axis (the angle between the up direction and the z-axis is
    smaller than pi/2).
    Args:
        radius (float): Distance from the origin.
        theta (float): Inclination, angle w.r.t. positive polar (+z) axis.
            Range: [0, pi].
        phi (float): Azimuth, rotation angle from the initial meridian (x-y)
            plane. Range: [0, 2*pi].
    Returns:
        T of shape (4, 4).
    Ref:
        https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """
    if not 0 <= theta <= np.pi:
        raise ValueError(f"Expected theta in [0, pi], but got {theta}.")
    if not 0 <= phi <= 2 * np.pi:
        raise ValueError(f"Expected phi in [0, 2*pi], but got {phi}.")

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Default   : look at +Z, up is -Y.
    # After init: look at +X, up is +Z.
    init_R = euler_to_R(-np.pi / 2, 0, -np.pi / 2)
    # Rotate along z axis.
    phi_R = euler_to_R(phi + np.pi, 0, 0)
    # Rotate along y axis.
    theta_R = euler_to_R(0, np.pi / 2 - theta, 0)

    # Combine rotations, the order matters.
    R = phi_R @ theta_R @ init_R
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = [x, y, z]
    T = pose_to_T(pose)

    return T

def spherical_to_Pose_towards_origin(radius, theta, phi):
    """
    Convert spherical coordinates (ISO convention) to T, where the cameras looks
    at the origin from a distance (radius), and the camera up direction alines
    with the z-axis (the angle between the up direction and the z-axis is
    smaller than pi/2).
    Args:
        radius (float): Distance from the origin.
        theta (float): Inclination, angle w.r.t. positive polar (+z) axis.
            Range: [0, pi].
        phi (float): Azimuth, rotation angle from the initial meridian (x-y)
            plane. Range: [0, 2*pi].
    Returns:
        T of shape (4, 4).
    Ref:
        https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """
    if not 0 <= theta <= np.pi:
        raise ValueError(f"Expected theta in [0, pi], but got {theta}.")
    if not 0 <= phi <= 2 * np.pi:
        raise ValueError(f"Expected phi in [0, 2*pi], but got {phi}.")

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Default   : look at +Z, up is -Y.
    # After init: look at +X, up is +Z.
    init_R = euler_to_R(-np.pi / 2, 0, -np.pi / 2)
    # Rotate along z axis.
    phi_R = euler_to_R(phi + np.pi, 0, 0)
    # Rotate along y axis.
    theta_R = euler_to_R(0, np.pi / 2 - theta, 0)

    # Combine rotations, the order matters.
    R = phi_R @ theta_R @ init_R
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = [x, y, z]

    return pose

def spherical_to_Pose_towards_origin_torch(radius, theta, phi):
    """
    Convert spherical coordinates (ISO convention) to T, where the cameras looks
    at the origin from a distance (radius), and the camera up direction alines
    with the z-axis (the angle between the up direction and the z-axis is
    smaller than pi/2).
    Args:
        radius (float): Distance from the origin.
        theta (float): Inclination, angle w.r.t. positive polar (+z) axis.
            Range: [0, pi].
        phi (float): Azimuth, rotation angle from the initial meridian (x-y)
            plane. Range: [0, 2*pi].
    Returns:
        T of shape (4, 4).
    Ref:
        https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """
    if not 0 <= theta <= torch.pi:
        raise ValueError(f"Expected theta in [0, pi], but got {theta}.")
    if not 0 <= phi <= 2 * torch.pi:
        raise ValueError(f"Expected phi in [0, 2*pi], but got {phi}.")

    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)

    # Default   : look at +Z, up is -Y.
    # After init: look at +X, up is +Z.
    init_R = euler_to_R(-torch.pi / 2, 0, -torch.pi / 2)
    # Rotate along z axis.
    phi_R = euler_to_R(phi + torch.pi, 0, 0)
    # Rotate along y axis.
    theta_R = euler_to_R(0, torch.pi / 2 - theta, 0)

    # Combine rotations, the order matters.
    R = phi_R @ theta_R @ init_R
    R = torch.tensor(R)
    pose = torch.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = torch.tensor([x, y, z])

    return pose