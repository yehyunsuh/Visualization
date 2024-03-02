import os
import argparse
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from io import BytesIO
from tqdm import tqdm

from utils import *


def parameters(theta_1, theta_2, theta_3):
    # Angle of the camera
    camera_angle_radian = np.radians([theta_1, theta_2, theta_3])

    object_plane = create_object_plane(args)
    projection_plane = create_projection_plane(args)

    # Parameters for the camera
    origin = np.array([0, 0, 0])
    camera_source = np.array([0, 0, 1]) * args.camera_distance

    return camera_angle_radian, object_plane, projection_plane, origin, camera_source


def rotation_of_camera_and_projection_plane(args, theta_1, theta_2, theta_3):
    camera_angle_radian, object_plane, projection_plane, origin, camera_source = parameters(theta_1, theta_2, theta_3)

    # Rotation matrix
    rotation_matrix = euler_angles_to_rotation_matrix(camera_angle_radian)

    # Rotate the camera
    camera_rotate = rotation_matrix @ camera_source.T

    # Rotate the projection plane
    projection_plane_coordinate = np.stack([projection_plane[0].flatten(), projection_plane[1].flatten(), projection_plane[2].flatten()])
    projection_plane_rotated = rotation_matrix @ projection_plane_coordinate
    projection_plane_rotated = projection_plane_rotated.reshape(3, args.num_points, args.num_points)

    # Center of the projection plane rotated
    projection_plane_rotated_center = np.mean(np.mean(projection_plane_rotated, axis=1), axis=1)

    return camera_rotate, origin, projection_plane_rotated_center, object_plane, projection_plane_rotated


def rotation_of_object_plane(args, theta_1, theta_2, theta_3):
    camera_angle_radian, object_plane, projection_plane, origin, camera_source = parameters(theta_1, theta_2, theta_3)

    # Rotation matrix
    rotation_matrix = euler_angles_to_rotation_matrix(camera_angle_radian)

    # Rotate the object plane
    object_plane_coordinate = np.stack([object_plane[0].flatten(), object_plane[1].flatten(), object_plane[2].flatten()])
    object_plane_rotated = rotation_matrix @ object_plane_coordinate
    object_plane_rotated = object_plane_rotated.reshape(3, args.num_points, args.num_points)

    # Center of the projection plane
    projection_plane_center = np.mean(np.mean(projection_plane, axis=1), axis=1)

    return camera_source, origin, projection_plane_center, object_plane_rotated, projection_plane


def baseline_visualization(args, theta_1, theta_2, theta_3, camera_position, origin, projection_plane_center, object_plane, projection_plane):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(object_plane[0], object_plane[1], object_plane[2], alpha=0.5, color='r')
    ax.plot_surface(projection_plane[0], projection_plane[1], projection_plane[2], alpha=0.5, color='b')

    ax.scatter(origin[0], origin[1], origin[2], color='black', s=30, label='Origin')
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], color='orange', s=30, label='Camera')
    ax.scatter(projection_plane_center[0], projection_plane_center[1], projection_plane_center[2], color='b', s=30, label='Target Center')

    ax.plot([camera_position[0], projection_plane_center[0]], [camera_position[1], projection_plane_center[1]], [camera_position[2], projection_plane_center[2]], color='black')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-args.camera_distance * 1.5, args.camera_distance * 1.5)
    ax.set_ylim(-args.camera_distance * 1.5, args.camera_distance * 1.5)
    ax.set_zlim(-args.camera_distance * 1.5, args.camera_distance * 1.5)
    ax.set_title(f'Camera Rotation: {theta_1}, {theta_2}, {theta_3} degrees')
    ax.legend(loc='upper left')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_buffer = imageio.imread(buf)

    # plt.savefig(f'visualization/light_source_projection/{theta_1}_{theta_2}_{theta_3}.png')
    plt.close()

    return image_buffer


def baseline_projection(args, theta_1, theta_2, theta_3):
    images_camera_rotation, images_object_plane_rotation = [], []
    camera_position, origin, projection_plane_center, object_plane, projection_plane = rotation_of_camera_and_projection_plane(
        args, theta_1, theta_2, theta_3
    )
    image_buffer_camera_rotation = baseline_visualization(
        args, theta_1, theta_2, theta_3, camera_position, origin, projection_plane_center, object_plane, projection_plane
    )

    camera_position, origin, projection_plane_center, object_plane, projection_plane = rotation_of_object_plane(
        args, theta_1, theta_2, theta_3
    )
    image_buffer_object_plane_rotation = baseline_visualization(
        args, theta_1, theta_2, theta_3, camera_position, origin, projection_plane_center, object_plane, projection_plane
    )

    images_camera_rotation.append(image_buffer_camera_rotation)
    images_object_plane_rotation.append(image_buffer_object_plane_rotation)

    return images_camera_rotation, images_object_plane_rotation
    

def circle_projection():
    pass


def translated_circle_projection():
    pass


def main(args):
    images_random_points = []

    theta_1_array = np.linspace(0, 0, 1, dtype=int)
    theta_2_array = np.linspace(0, 30, 31, dtype=int)
    theta_3_array = np.linspace(0, 0, 1, dtype=int)
    for theta_1 in tqdm(theta_1_array):
        for theta_2 in theta_2_array:
            for theta_3 in theta_3_array:
                camera_rotation, object_plane_rotation = baseline_projection(args, theta_1, theta_2, theta_3)
                concatenated_image = np.concatenate([camera_rotation, object_plane_rotation], axis=1)
                images_random_points.append(concatenated_image)

    theta_1_array = np.linspace(1, 30, 30, dtype=int)
    theta_2_array = np.linspace(30, 30, 1, dtype=int)
    theta_3_array = np.linspace(0, 0, 1, dtype=int)
    for theta_1 in tqdm(theta_1_array):
        for theta_2 in theta_2_array:
            for theta_3 in theta_3_array:
                camera_rotation, object_plane_rotation = baseline_projection(args, theta_1, theta_2, theta_3)
                concatenated_image = np.concatenate([camera_rotation, object_plane_rotation], axis=1)
                images_random_points.append(concatenated_image)

    imageio.mimsave('visualization/light_source_projection/light_source_camera_rotation.gif', images_random_points, 'GIF', fps=10, loop=3)


if __name__ == '__main__':
    os.makedirs('visualization/light_source_projection', exist_ok=True)
    parser = argparse.ArgumentParser(description='Light Source Projection')

    parser.add_argument('--num_points', type=int, default=1000, help='Number of points (default: 1000)')
    parser.add_argument('--object_plane_size', type=int, default=200, help='Size of the object plane (default: 200)')
    parser.add_argument('--projection_plane_size', type=int, default=500, help='Size of the projection plane (default: 500)')
    parser.add_argument('--camera_distance', type=int, default=500, help='Distance of camera (default: 500)')

    args = parser.parse_args()

    main(args)