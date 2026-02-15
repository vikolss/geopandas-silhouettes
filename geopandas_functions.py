import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely import Polygon, LineString, Point, MultiLineString, MultiPolygon
import os

# Convert spherical coordinates to Cartesian coordinates
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

# Convert Cartesian coordinates to spherical coordinates
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return np.array([r, theta, phi])

# Rotate a point around an axis
def rotate(point, axis, angle):
    # Normalize the axis vector
    axis = axis / np.sqrt(np.dot(axis, axis))
    
    # Compute the cosine and sine of the angle
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Use Rodrigues' rotation formula to rotate the point
    rotated_point = cos_angle * point + sin_angle * np.cross(axis, point) + (1 - cos_angle) * np.dot(axis, point) * axis
    
    return rotated_point

# Main function
def rotate_point_spherical(spherical_point1, spherical_point2, angle):
    # Convert the points to Cartesian coordinates
    cartesian_point1 = spherical_to_cartesian(*spherical_point1)
    cartesian_point2 = spherical_to_cartesian(*spherical_point2)
    
    # Rotate the first point around the second point
    rotated_point = rotate(cartesian_point1, cartesian_point2, angle)
    
    # Convert the rotated point back to spherical coordinates
    spherical_rotated_point = cartesian_to_spherical(*rotated_point)
    
    return spherical_rotated_point

def rotate_points(spherical_point1, points_to_rotate, angle):
    new_points = []
    for point in points_to_rotate:
        rotated_point = rotate_point_spherical(point, spherical_point1, angle)
        new_points.append(rotated_point)
    return new_points

def get_vertical_circle(point, num_points=120):
    thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=True) + point[1]
    thetas = thetas % (2 * np.pi)
    other_side_idx = thetas > np.pi
    thetas[other_side_idx] = 2 * np.pi - thetas[other_side_idx]
    phis = np.ones(num_points) * point[2]
    phis[other_side_idx] = (point[2] - np.pi) % (2 * np.pi)
    circle = [(1, theta, phi) for theta, phi in zip(thetas, phis)]
    return circle

def get_rotated_circle(point, angle):
    pole_circle = get_vertical_circle(point)
    great_circle = rotate_points(point, pole_circle, angle)
    return great_circle

def get_great_circle(latitude, longitude, angle):
    latitude = np.radians(90 - latitude)
    longitude = np.radians(longitude)
    angle = np.radians(angle)
    spherical_point = [1, latitude, longitude]

    circle = np.array(get_rotated_circle(spherical_point, angle))

    latitudes = 90 - np.degrees(circle[:, 1])
    longitudes = np.degrees(circle[:, 2])
    return latitudes, longitudes

def get_polygon(latitudes, longitudes):
    return Polygon([(y, x) for x, y in zip(latitudes, longitudes)])

def get_line(latitudes, longitudes):
    return LineString([(y, x) for x, y in zip(latitudes, longitudes)])

def get_coords(line):
    if isinstance(line, LineString):
        return line.coords.xy
    elif isinstance(line, MultiLineString):
        x_coords = []
        y_coords = []
        for sub_line in line.geoms:
            x, y = get_coords(sub_line)
            x_coords += x
            y_coords += y
        return np.array(x_coords), np.array(y_coords)

def find_closest_intersections(latitudes, longitudes, map, direction=None):
    inter_1, inter_2 = None, None
    if direction is None or direction == 0:
        for i in range(2, len(latitudes)):
            line = get_line(latitudes[i-2:i], longitudes[i-2:i])
            if line.length > 10: continue
            if line.intersects(map):
                x, y = get_coords(line.intersection(map))
                inter_1 = (x[0], y[0])
                break
    if direction is None or direction == 1:
        for i in range(2, len(latitudes)):
            # line = get_line(latitudes[-i:-(i-2)], longitudes[-i:-(i-2)])
            line = get_line(latitudes[::-1][i-2:i], longitudes[::-1][i-2:i])
            if line.length > 10: continue
            if line.intersects(map):
                x, y = get_coords(line.intersection(map))
                inter_2 = (x[-1], y[-1])
                inter_2 = (x[0], y[0])
                break
    return inter_1, inter_2

def generate_all_great_circles(latitude, longitude, num_circles=100, angles=None):
    if angles is None:
        angles = np.linspace(0, 180, num_circles, endpoint=False)
    all_circles = []
    for angle in angles:
        latitudes, longitudes = get_great_circle(latitude, longitude, angle)
        all_circles.append([latitudes, longitudes])
    return all_circles

def get_all_intersections(latitude, longitude, map, num_circles=100):
    all_circles = generate_all_great_circles(latitude, longitude, num_circles)
    intersections_1 = []
    intersections_2 = []
    for circle in all_circles:
        intersection = find_closest_intersections(circle[0], circle[1], map)
        intersections_1.append(intersection[0])
        intersections_2.append(intersection[1])
    return np.concatenate([intersections_1, intersections_2])

def distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

def sequential_distances(coordinates):
    distances = []
    p1 = coordinates[0]
    for p2 in coordinates[1:]:
        distances.append(distance(p1, p2))
        p1 = p2
    return distances

def get_all_intersections_dynamic(latitude, longitude, map, max_distance=10, max_iterations=5):
    initial_circles = 30
    angles = list(np.linspace(0, 180, initial_circles, endpoint=True))
    all_circles = generate_all_great_circles(latitude, longitude, angles=angles)
    intersections_1 = []
    intersections_2 = []
    for circle in all_circles:
        intersection = find_closest_intersections(circle[0], circle[1], map)
        intersections_1.append(intersection[0])
        intersections_2.append(intersection[1])
    for _ in range(max_iterations):
        distances_1 = sequential_distances(intersections_1)
        distances_2 = sequential_distances(intersections_2)
        distances = np.array([distances_1, distances_2])
        max_distances = np.max(distances, axis=0)
        new_indices = []
        new_angles = []
        new_intersections_1 = []
        new_intersections_2 = []
        for i, dist in enumerate(max_distances):
            if dist > max_distance:
                new_angle = np.mean(angles[i:i+2])
                latitudes, longitudes = get_great_circle(latitude, longitude, new_angle)
                intersection = find_closest_intersections(latitudes, longitudes, map)
                new_indices.append(i)
                new_angles.append(new_angle)
                new_intersections_1.append(intersection[0])
                new_intersections_2.append(intersection[1])
        for i, ang, int_1, int_2 in list(zip(new_indices, new_angles, new_intersections_1, new_intersections_2))[::-1]:
            intersections_1.insert(i + 1, int_1)
            intersections_2.insert(i + 1, int_2)
            angles.insert(i + 1, ang)

        # min_direction = np.argmin(distances[min_idx])
        # direction = None
    return np.concatenate([intersections_1, intersections_2])

def unwrap_intersection(intersections, longitude):
    new_intersections = intersections.copy()
    cutoff = len(new_intersections)//2
    right_half_x = new_intersections[:cutoff, 0]
    right_half_x[right_half_x < (longitude - 1)] += 360
    new_intersections[:cutoff, 0] = right_half_x
    left_half_x = new_intersections[cutoff:, 0]
    left_half_x[left_half_x > (longitude + 1)] -= 360
    new_intersections[cutoff:, 0] = left_half_x
    return new_intersections

def point_in_ocean(latitude, longitude, map_obj):
    point = Point(longitude, latitude)
    return not map_obj.contains(point)

def set_new_pole(points, pole_latitude, pole_longitude):
    pole_theta = np.radians(pole_longitude)
    pole_phi = np.radians(90 - pole_latitude)

    thetas = np.radians(points[:, 0])
    phis = np.radians(90 - points[:, 1])
    spherical_points = [[1, phi, theta] for phi, theta in zip(phis, thetas)]
    rotated_points = []
    for point in spherical_points:
        cartesian_point = spherical_to_cartesian(*point)
        rotated_point = rotate(cartesian_point, [0, 0, 1], -pole_theta)
        rotated_point = rotate(rotated_point, [0, 1, 0], pole_phi)

        rotated_points.append(cartesian_to_spherical(*rotated_point))

    rotated_spherical_points = np.array(rotated_points)
    latitudes = 90 - np.degrees(rotated_spherical_points[:, 1])
    longitudes = (np.degrees(rotated_spherical_points[:, 2]) + 180) % 360 - 180
    rotated_spherical_points = np.array([longitudes, latitudes]).T
    return rotated_spherical_points

def interpolate_points(points, max_angle_gap=10):
    long = points[:, 0]
    lat = points[:, 1]
    diffs = np.diff(long)
    insert_indices = []
    insert_long = []
    insert_lat = []
    for i, diff in enumerate(diffs):
        diff_val = np.abs(diff)
        if diff_val > max_angle_gap and diff_val < (360 - max_angle_gap):
            insert_indices.append(i+1)
            sign = np.sign(diff)
            if diff_val > 180:
                diff_val = 360 - diff_val
                sign *= -1
            insert_num = int(diff_val // max_angle_gap)

            lat_vals = np.linspace(lat[i], lat[i+1], insert_num+2)[1:-1]
            long_vals = np.linspace(0, diff_val * sign, insert_num+2)[1:-1]
            long_vals = long[i] + long_vals
            long_vals = (long_vals + 180) % 360 -180
            insert_long.append(long_vals)
            insert_lat.append(lat_vals)
    for idx, long_vals, lat_vals in list(zip(insert_indices, insert_long, insert_lat))[::-1]:
        long = np.concatenate([long[:idx], long_vals, long[idx:]])
        lat = np.concatenate([lat[:idx], lat_vals, lat[idx:]])
    return np.array([long, lat]).T



# Plotting functions

def plot_intersections(intersections, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if 'color' not in kwargs:
        kwargs['color'] = 'r'
    for inter in intersections:
        ax.plot(inter[0], inter[1], '.', **kwargs)
    return ax

def create_silhouette_plot(intersections):
    _, ax = plt.subplots(figsize=(40,20))
    plot_intersections(intersections, ax=ax, markersize=2)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xlim_diff = np.diff(xlim)
    ylim_diff = np.diff(ylim)
    if xlim_diff < (ylim_diff * 2):
        margin = ((ylim_diff * 2) - xlim_diff) / 2
        xlim = (xlim[0] - margin, xlim[1] + margin)
        ax.set_xlim(xlim)
    else:
        margin = ((xlim_diff / 2) - ylim_diff) / 2
        ylim = (ylim[0] - margin, ylim[1] + margin)
        ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    return xlim, ylim

def wrap_map_obj(map_obj, direction):
    assert direction in [-1, 1]
    wrapped_map_list = []
    for poly in map_obj.geoms:
        X, Y = poly.exterior.xy
        new_X = np.array(X) + direction * 360
        new_poly = Polygon([(x, y) for x, y in zip(new_X, Y)])
        wrapped_map_list.append(new_poly)

    wrapped_map = MultiPolygon(wrapped_map_list)
    wrapped_map = wrapped_map.union(map_obj)
    return wrapped_map

def plot_with_map(map_obj, base_latitude, base_longitude, intersections=None, xlim=None, ylim=None):
    if xlim is not None and xlim[0] < -180:
        map_obj = wrap_map_obj(map_obj, -1)
    if xlim is not None and xlim[1] > 180:
        map_obj = wrap_map_obj(map_obj, 1)
    map_df = gpd.GeoDataFrame([0], geometry=[map_obj])
    ax = map_df.plot(figsize=(40,20), alpha=0.3, color='b')
    ax.plot(base_longitude, base_latitude, 'ok', markeredgecolor='b', markersize=10)
    if intersections is not None:
        plot_intersections(intersections, ax=ax, markersize=2)
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim((-180, 180))
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim((-90, 90))

def save_images(intersections, map_obj, base_latitude, base_longitude, dir):
    name = f"lat:{base_latitude}_long:{base_longitude}"
    xlim, ylim = create_silhouette_plot(intersections)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, f"{name}_silhouette.png"))
    plt.close()
    plot_with_map(intersections, map_obj, base_latitude, base_longitude, xlim, ylim)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, f"{name}_map.png"))
    plt.close()

def no_gaps_in_angles(angles_list, tolerance=5):
    """
    Check if a list of angles contains no gaps.

    Parameters:
        angles_list (list): List of angles in degrees.
        tolerance (float): The maximum tolerance for angle gaps.
                           Default is 5 degrees.

    Returns:
        bool: True if the list of angles contains no gaps, False otherwise.
    """
    angles_list = sorted(angles_list)  # Sort the angles in ascending order

    for i in range(1, len(angles_list)):
        angle_diff = (angles_list[i] - angles_list[i - 1]) % 360
        if angle_diff > tolerance:
            return False

    # Check if the difference between the last and first angles is within tolerance
    if (angles_list[0] - angles_list[-1] + 360) % 360 > tolerance:
        return False

    return True

def plot_polar(ax, x, y, *args, **kwargs):
    x = x + 90
    ax.plot(np.radians(x), 90 - y, *args, **kwargs)

def fill_polar(ax, x, y, *args, **kwargs):
    x = x + 90
    if no_gaps_in_angles(x, 30) and np.min(y) < -70:
        new_x = (np.linspace(0, 360) + x[0])[::-1]
        new_y = np.ones(len(new_x)) * -90
        x = np.concatenate([x, new_x])
        y = np.concatenate([y, new_y])

    ax.fill(np.radians(x), 90 - y, *args, **kwargs)

def create_polar_silhouette_plot(intersections, **kwargs):
    fig, ax = plt.subplots(figsize=(30,30), subplot_kw={'projection': 'polar'})
    if 'color' not in kwargs:
        kwargs['color'] = 'r'
    plot_polar(ax, intersections[:, 0], intersections[:, 1], '.', markersize=2, **kwargs)
    max_radius = np.max(90 - intersections[:, 1])
    ax.set_ylim((0, np.min([180, max_radius + 3])))
    plot_polar(ax, 0, 90, 'bo')
    ax.set_xticks([])
    ax.set_yticks([])

def plot_polar_map(coords, intersections=None):
    if intersections is not None:
        create_polar_silhouette_plot(intersections)
        ax = plt.gca()
    else:
        _, ax = plt.subplots(figsize=(30,30), subplot_kw={'projection': 'polar'})
    for coord in coords:
        fill_polar(ax, coord[:, 0], coord[:, 1], 'b', alpha=0.3)

def get_polar_points(points, pole_lat, pole_long):
    pole_long = pole_long + 180
    return set_new_pole(points, pole_lat, pole_long)

def get_polar_map_coords(map_obj, pole_lat, pole_long):
    pole_long = pole_long + 180
    polar_map_coords = []
    for poly in map_obj.geoms:
        x, y = poly.exterior.xy
        points = set_new_pole(np.array([x, y]).T, pole_lat, pole_long)
        points = interpolate_points(points, 10)
        polar_map_coords.append(points)
    return polar_map_coords

def save_all_plots(intersections, map_obj, base_latitude, base_longitude, dir):
    name = f"lat:{base_latitude}_long:{base_longitude}"
    polar_inters = get_polar_points(intersections, base_latitude, base_longitude)
    polar_map_coords = get_polar_map_coords(map_obj, base_latitude, base_longitude)

    create_polar_silhouette_plot(polar_inters)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, f"{name}_polar_silhouette.png"))
    plt.close()
    plot_polar_map(polar_map_coords, polar_inters)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, f"{name}_polar_map.png"))
    plt.close()
    intersections = unwrap_intersection(intersections, base_longitude)
    xlim, ylim = create_silhouette_plot(intersections)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, f"{name}_silhouette.png"))
    plt.close()
    plot_with_map(map_obj, base_latitude, base_longitude, intersections, xlim=xlim, ylim=ylim)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, f"{name}_map.png"))
    plt.close()