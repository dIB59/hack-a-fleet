import pydeck as pdk
from pydeck.data_utils import compute_view

WIDTH = 300
HEIGHT = 300

CLUSTER_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (230, 25, 75),
    (60, 180, 75),
    (245, 130, 49),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (188, 246, 12),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (154, 99, 36),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 216, 177),
    (0, 0, 128),
    (169, 169, 169),
    (255, 255, 255),
    (0, 0, 0),
]


def get_cluster_colors(labels):
    return [
        CLUSTER_COLORS[label] if label != -1 else [255.0, 255.0, 255.0]
        for label in labels
    ]


def flip_coordinates_order(path):
    """Flip the order of the coordinates in a path"""
    return [(p[1], p[0]) for p in path]


def make_paths_layer(paths, colors=None, opacity=0.95):
    if colors is None:
        colors = [[333, 33, 33]] * len(paths)
    paths_pdk = [
        {"path": flip_coordinates_order(path), "color": color}
        for path, color in zip(paths, colors)
    ]
    return pdk.Layer(
        "PathLayer",
        paths_pdk,
        get_color="color",
        opacity=opacity,
        width_min_pixels=5,
        rounded=True,
    )


def plot_paths(paths, colors=None):
    layer = make_paths_layer(paths, colors=colors)
    points = [point for path in paths for point in flip_coordinates_order(path)]
    view_state = compute_view(points)
    r = pdk.Deck(layers=[layer], initial_view_state=view_state)
    return r


def plot_path_clusters(path_clusters, representative_paths):
    cluster_colors = get_cluster_colors(range(len(path_clusters) + 1))
    layers = []
    points = []
    for paths, c_color in zip(path_clusters, cluster_colors):
        colors = [c_color] * len(paths)
        points += [point for path in paths for point in flip_coordinates_order(path)]
        layers.append(make_paths_layer(paths, colors, opacity=0.1))

    layers.append(make_paths_layer(representative_paths, cluster_colors))
    view_state = compute_view(points)
    r = pdk.Deck(layers=layers, initial_view_state=view_state)
    return r
