import numpy as np


def create_object_plane(args):
    # define a grid of x, y, and z values
    x = np.linspace(-args.object_plane_size, args.object_plane_size, args.num_points)
    y = np.linspace(-args.object_plane_size, args.object_plane_size, args.num_points)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x)
    
    return  x, y, z


def create_projection_plane(args):
    # define a grid of x and y values
    x = np.linspace(-args.projection_plane_size, args.projection_plane_size, args.num_points)
    y = np.linspace(-args.projection_plane_size, args.projection_plane_size, args.num_points)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x) - args.camera_distance
    
    return  x, y, z


def euler_angles_to_rotation_matrix(camera_angle):
    theta_1, theta_2, theta_3 = camera_angle
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_1), -np.sin(theta_1)],
        [0, np.sin(theta_1), np.cos(theta_1)]
    ])
    rotation_matrix_y = np.array([
        [np.cos(theta_2), 0, np.sin(theta_2)],
        [0, 1, 0],
        [-np.sin(theta_2), 0, np.cos(theta_2)]
    ])
    rotation_matrix_z = np.array([
        [np.cos(theta_3), -np.sin(theta_3), 0],
        [np.sin(theta_3), np.cos(theta_3), 0],
        [0, 0, 1]
    ])
    rotation_matrix = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x

    return rotation_matrix


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