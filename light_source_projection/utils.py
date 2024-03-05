import numpy as np


def create_object_plane(args):
    # define a grid of x, y, and z values
    x = np.linspace(-args.object_plane_size, args.object_plane_size, args.n_plane_points)
    y = np.linspace(-args.object_plane_size, args.object_plane_size, args.n_plane_points)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x)
    
    return  x, y, z


def create_projection_plane(args):
    # define a grid of x and y values
    x = np.linspace(-args.projection_plane_size, args.projection_plane_size, args.n_plane_points)
    y = np.linspace(-args.projection_plane_size, args.projection_plane_size, args.n_plane_points)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x) - args.camera_distance
    
    return  x, y, z


def parameters(args, theta_1, theta_2, theta_3):
    # Angle of the camera
    camera_angle_radian = np.radians([theta_1, theta_2, theta_3])

    object_plane = create_object_plane(args)
    projection_plane = create_projection_plane(args)

    # Parameters for the camera
    origin = np.array([0, 0, 0])
    camera_source = np.array([0, 0, 1]) * args.camera_distance

    return camera_angle_radian, object_plane, projection_plane, origin, camera_source


def calculate_angle_between_vectors(v_1, v_2):
    if np.all(v_1 == 0) or np.all(v_2 == 0):
        return np.pi
    if v_1[0] == v_2[0] and v_1[1] == v_2[1]:
        return 0.00

    dot_product = np.dot(v_1, v_2)  # v_1[0] * v_2[0] + v_1[1] * v_2[1] + v_1[2] * v_2[2] 
    magnitude1 = np.linalg.norm(v_1)  # np.sqrt(v_1[0] ** 2 + v_1[1] ** 2 + v_1[2] ** 2)
    magnitude2 = np.linalg.norm(v_2)  # np.sqrt(v_2[0] ** 2 + v_2[1] ** 2 + v_2[2] ** 2)

    angle = np.arccos(dot_product / (magnitude1 * magnitude2))

    return np.degrees(angle)