import os
import argparse
import numpy as np
import imageio.v2 as imageio

from tqdm import tqdm

from utils import *
from rotation import *
from visualization import *


def baseline_camera_and_projection_plane(args, object_plane, projection_plane, origin, camera_source, rotation_matrix):
    camera_rotate = rotation_of_camera(
        rotation_matrix, camera_source
    )
    projection_plane_rotated_center, projection_plane_rotated = rotation_of_projection_plane(
        args, rotation_matrix, projection_plane
    )

    return camera_rotate, origin, projection_plane_rotated_center, object_plane, projection_plane_rotated


def baseline_object_plane(args, object_plane, projection_plane, origin, camera_source, rotation_matrix):
    projection_plane_center, object_plane_rotated = rotation_of_object_plane(
        args, object_plane, rotation_matrix, projection_plane
    )

    return camera_source, origin, projection_plane_center, object_plane_rotated, projection_plane


def baseline_projection(args, theta_1, theta_2, theta_3):
    camera_angle_radian, object_plane, projection_plane, origin, camera_source = parameters(args, theta_1, theta_2, theta_3)
    rotation_matrix = euler_angles_to_rotation_matrix(camera_angle_radian)

    # Rotation of camera and projection plane
    camera_position, origin, projection_plane_rotated_center, object_plane, projection_plane_rotated = baseline_camera_and_projection_plane(
        args, object_plane, projection_plane, origin, camera_source, rotation_matrix
    )
    image_buffer_camera_rotation = baseline_visualization(
        args, theta_1, theta_2, theta_3, camera_position, origin, projection_plane_rotated_center, object_plane, projection_plane_rotated
    )

    # Rotation of object plane
    camera_position, origin, projection_plane_center, object_plane_rotated, projection_plane = baseline_object_plane(
        args, object_plane, projection_plane, origin, camera_source, rotation_matrix
    )
    image_buffer_object_plane_rotation = baseline_visualization(
        args, theta_1, theta_2, theta_3, camera_position, origin, projection_plane_center, object_plane_rotated, projection_plane
    )

    return np.concatenate([image_buffer_camera_rotation, image_buffer_object_plane_rotation], axis=1)


def random_point_object_plane(args, object_plane, projection_plane, origin, camera_source, rotation_matrix, points):
    projection_plane_center, object_plane_rotated = rotation_of_object_plane(
        args, object_plane, rotation_matrix, projection_plane
    )
    rotated_points = rotation_of_points(
        rotation_matrix, points
    )

    return camera_source, origin, projection_plane_center, object_plane_rotated, projection_plane, rotated_points


def random_point_projection(args, theta_1, theta_2, theta_3, points):
    camera_angle_radian, object_plane, projection_plane, origin, camera_source = parameters(args, theta_1, theta_2, theta_3)
    rotation_matrix = euler_angles_to_rotation_matrix(camera_angle_radian)

    # Projected points on the projection plane
    H, h = args.camera_distance * 2, args.camera_distance
    ratio = H / (H - h - points[2])

    # Rotation of object plane and points
    camera_position, origin, projection_plane_center, object_plane_rotated, projection_plane, rotated_points = random_point_object_plane(
        args, object_plane, projection_plane, origin, camera_source, rotation_matrix, points
    )

    x_points_projection = rotated_points[0] * ratio
    y_points_projection = rotated_points[1] * ratio
    z_points_projection = np.zeros_like(x_points_projection) - h
    points_projection = np.stack([x_points_projection, y_points_projection, z_points_projection])

    image_buffer_object_plane_rotation = point_projection_visualization(
        args, theta_1, theta_2, theta_3, camera_position, origin, projection_plane_center, object_plane_rotated, projection_plane, rotated_points, points_projection
    )

    return image_buffer_object_plane_rotation


def main(args):
    # Set random seed for reproducibility
    np.random.seed(5)

    # Random points on the object plane
    x_points = np.random.uniform(-args.object_plane_size//2, args.object_plane_size//2, args.n_random_points)
    y_points = np.random.uniform(-args.object_plane_size//2, args.object_plane_size//2, args.n_random_points)
    z_points = np.zeros_like(x_points)
    points = np.stack([x_points, y_points, z_points])

    # Circle points on the object plane with radius of args.radius
    theta = np.linspace(0, 2 * np.pi, args.n_random_points)
    x_circle_points = args.radius * np.cos(theta)
    y_circle_points = args.radius * np.sin(theta)
    z_circle_points = np.zeros_like(x_circle_points)
    circle_points = np.stack([x_circle_points, y_circle_points, z_circle_points])
    
    # Translate the circle_points by args.k on circle_points[0]
    translated_circle_points = np.stack([x_circle_points, y_circle_points, z_circle_points])
    translated_circle_points[0] += args.k

    # Translate the circle_points by args.k on circle_points[0] and args.l on circle_points[2]
    translated_circle_points2 = np.stack([x_circle_points, y_circle_points, z_circle_points])
    translated_circle_points2[0] += args.k
    translated_circle_points2[2] += args.l

    images_baseline, images_random_point = [], []
    images_circle, images_translated_circle = [], []
    images_translated_circle2 = []

    theta_1_array = np.linspace(0, 0, 1, dtype=int)
    theta_2_array = np.linspace(0, 30, 31, dtype=int)
    theta_3_array = np.linspace(0, 0, 1, dtype=int)
    for theta_1 in tqdm(theta_1_array):
        for theta_2 in theta_2_array:
            for theta_3 in theta_3_array:
                images_baseline.append(baseline_projection(args, theta_1, theta_2, theta_3))
                images_random_point.append(random_point_projection(args, theta_1, theta_2, theta_3, points))
                images_circle.append(random_point_projection(args, theta_1, theta_2, theta_3, circle_points))
                images_translated_circle.append(random_point_projection(args, theta_1, theta_2, theta_3, translated_circle_points))
                images_translated_circle2.append(random_point_projection(args, theta_1, theta_2, theta_3, translated_circle_points2))

    theta_1_array = np.linspace(1, 30, 30, dtype=int)
    theta_2_array = np.linspace(30, 30, 1, dtype=int)
    theta_3_array = np.linspace(0, 0, 1, dtype=int)
    for theta_1 in tqdm(theta_1_array):
        for theta_2 in theta_2_array:
            for theta_3 in theta_3_array:
                images_baseline.append(baseline_projection(args, theta_1, theta_2, theta_3))
                images_random_point.append(random_point_projection(args, theta_1, theta_2, theta_3, points))
                images_circle.append(random_point_projection(args, theta_1, theta_2, theta_3, circle_points))
                images_translated_circle.append(random_point_projection(args, theta_1, theta_2, theta_3, translated_circle_points))
                images_translated_circle2.append(random_point_projection(args, theta_1, theta_2, theta_3, translated_circle_points2))

    imageio.mimsave('visualization/light_source_projection/light_source_camera_rotation.gif', images_baseline, 'GIF', fps=10, loop=3)
    imageio.mimsave('visualization/light_source_projection/light_source_random_point.gif', images_random_point, 'GIF', fps=10, loop=3)
    imageio.mimsave('visualization/light_source_projection/light_source_circle.gif', images_circle, 'GIF', fps=10, loop=3)
    imageio.mimsave('visualization/light_source_projection/light_source_translated_circle.gif', images_translated_circle, 'GIF', fps=10, loop=3)
    imageio.mimsave('visualization/light_source_projection/light_source_translated_circle2.gif', images_translated_circle2, 'GIF', fps=10, loop=3)


if __name__ == '__main__':
    os.makedirs('visualization/light_source_projection', exist_ok=True)
    parser = argparse.ArgumentParser(description='Light Source Projection')

    parser.add_argument('--n_plane_points', type=int, default=1000, help='Number of points on planes (default: 1000)')
    parser.add_argument('--object_plane_size', type=int, default=300, help='Size of the object plane (default: 200)')
    parser.add_argument('--projection_plane_size', type=int, default=500, help='Size of the projection plane (default: 500)')
    parser.add_argument('--camera_distance', type=int, default=500, help='Distance of camera (default: 500)')
    parser.add_argument('--n_random_points', type=int, default=20, help='Number of points (default: 30)')

    # Circle points
    parser.add_argument('--radius', type=int, default=100, help='Radius of the circle points (default: 100)')
    parser.add_argument('--k', type=int, default=50, help='Translation of circle points (default: 50)')
    parser.add_argument('--l', type=int, default=20, help='Translation of circle points (default: 20)')

    args = parser.parse_args()

    main(args)