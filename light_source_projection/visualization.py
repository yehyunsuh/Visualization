import imageio.v2 as imageio
import matplotlib.pyplot as plt

from io import BytesIO


def visualization(camera_position, origin, projection_plane_center, object_plane, projection_plane):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot object and projection plane
    ax.plot_surface(object_plane[0], object_plane[1], object_plane[2], alpha=0.5, color='r')
    ax.plot_surface(projection_plane[0], projection_plane[1], projection_plane[2], alpha=0.5, color='b')

    # Plot origin, camera, and target center
    ax.scatter(origin[0], origin[1], origin[2], color='black', s=30, label='Origin')
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], color='orange', s=30, label='Camera')
    ax.scatter(projection_plane_center[0], projection_plane_center[1], projection_plane_center[2], color='b', s=30, label='Target Center')

    # Plot the line between the camera and the target center
    ax.plot([camera_position[0], projection_plane_center[0]], [camera_position[1], projection_plane_center[1]], [camera_position[2], projection_plane_center[2]], color='black')

    return fig, ax


def visualization_points(ax, camera_position, points, points_projection):
    # Plot the points and their projection
    ax.scatter(points[0], points[1], points[2], color='green', s=30, label='Points')
    ax.scatter(points_projection[0], points_projection[1], points_projection[2], color='purple', s=30, label='Points Projection')

    # Plot the line between the camera, points and their projection
    for i in range(len(points[0])):
        ax.plot([camera_position[0], points[0][i]], [camera_position[1], points[1][i]], [camera_position[2], points[2][i]], color='black', linewidth=0.5)
        ax.plot([points[0][i], points_projection[0][i]], [points[1][i], points_projection[1][i]], [points[2][i], points_projection[2][i]], color='black', linewidth=0.5)

    return ax


def set_axes(args, ax, theta_1, theta_2, theta_3):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-args.camera_distance * 1.5, args.camera_distance * 1.5)
    ax.set_ylim(-args.camera_distance * 1.5, args.camera_distance * 1.5)
    ax.set_zlim(-args.camera_distance * 1.5, args.camera_distance * 1.5)
    ax.set_title(f'Camera Rotation: {theta_1}, {theta_2}, {theta_3} degrees')
    ax.legend(loc='upper left')

    return ax


def png_to_buffer():
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_buffer = imageio.imread(buf)
    plt.close()

    return image_buffer


def baseline_visualization(args, theta_1, theta_2, theta_3, camera_position, origin, projection_plane_center, object_plane, projection_plane):
    fig, ax = visualization(camera_position, origin, projection_plane_center, object_plane, projection_plane)
    ax = set_axes(args, ax, theta_1, theta_2, theta_3)

    return png_to_buffer()


def point_projection_visualization(args, theta_1, theta_2, theta_3, camera_position, origin, projection_plane_center, object_plane, projection_plane, points, points_projection):
    fig, ax = visualization(camera_position, origin, projection_plane_center, object_plane, projection_plane)
    ax = visualization_points(ax, camera_position, points, points_projection)
    ax = set_axes(args, ax, theta_1, theta_2, theta_3)

    return png_to_buffer()