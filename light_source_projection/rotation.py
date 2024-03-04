import numpy as np


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


def rotation_of_camera(rotation_matrix, camera_source):
    # Rotate the camera
    camera_rotate = rotation_matrix @ camera_source.T

    return camera_rotate


def rotation_of_object_plane(args, object_plane, rotation_matrix, projection_plane):
    # Rotate the object plane
    object_plane_coordinate = np.stack([object_plane[0].flatten(), object_plane[1].flatten(), object_plane[2].flatten()])
    object_plane_rotated = rotation_matrix @ object_plane_coordinate
    object_plane_rotated = object_plane_rotated.reshape(3, args.num_points, args.num_points)

    # Center of the projection plane
    projection_plane_center = np.mean(np.mean(projection_plane, axis=1), axis=1)

    return projection_plane_center, object_plane_rotated


def rotation_of_projection_plane(args, rotation_matrix, projection_plane):
    # Rotate the projection plane
    projection_plane_coordinate = np.stack([projection_plane[0].flatten(), projection_plane[1].flatten(), projection_plane[2].flatten()])
    projection_plane_rotated = rotation_matrix @ projection_plane_coordinate
    projection_plane_rotated = projection_plane_rotated.reshape(3, args.num_points, args.num_points)

    # Center of the projection plane rotated
    projection_plane_rotated_center = np.mean(np.mean(projection_plane_rotated, axis=1), axis=1)

    return projection_plane_rotated_center, projection_plane_rotated