"""
Displays two videos side by side, allowing the user to calibrate the camera.
"""

import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QDockWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut
import pygfx as gfx
from wgpu.gui.qt import WgpuCanvas
import numpy as np
import sleap_io as sio
import attrs
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import least_squares


@attrs.define(eq=False)
class CorrespondingPointSet:
    """Container for a set of corresponding points at a given frame index across any number of videos."""

    frame_idx: int
    video_points: dict[sio.Video, list[tuple[float, float]]] = attrs.field(factory=dict)
    name: str = ""

    def __getitem__(self, video: sio.Video):
        return self.video_points[video]

    def __setitem__(self, video: sio.Video, point: list[tuple[float, float]]):
        self.video_points[video] = point

    def __contains__(self, video: sio.Video):
        return video in self.video_points


def robust_cost_function(
    params, n_cameras, n_points, camera_indices, point_indices, points_2d, camera_matrix
):
    camera_params = params[: n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9 :].reshape((n_points, 3))

    errors = []
    for i in range(len(points_2d)):
        camera_idx = camera_indices[i]
        point_idx = point_indices[i]
        point_3d = points_3d[point_idx]
        camera_param = camera_params[camera_idx]

        projected_point = project_point(
            point_3d, camera_matrix, camera_param[:3], camera_param[3:6]
        )
        error = np.linalg.norm(points_2d[i] - projected_point)
        errors.append(error)

    return huber_loss(np.array(errors), delta=1.0)


def huber_loss(errors, delta):
    abs_errors = np.abs(errors)
    quadratic = np.minimum(abs_errors, delta)
    linear = abs_errors - quadratic
    return 0.5 * quadratic**2 + delta * linear


def initialize_cameras(n_cameras):
    # Initialize camera parameters: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    return np.hstack([np.ones((n_cameras, 4)), np.zeros((n_cameras, 5))])


def initialize_3d_points(n_points):
    # Random initial guess for 3D points
    return np.random.rand(n_points, 3)


def bundle_adjust_self_calibration(
    points_2d, camera_indices, point_indices, camera_matrix
):
    # Initialize parameters
    n_cameras = len(np.unique(camera_indices))
    n_points = len(np.unique(point_indices))
    camera_params = initialize_cameras(n_cameras)
    points_3d = initialize_3d_points(n_points)

    # Define optimization parameters
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

    # Optimize
    res = least_squares(
        robust_cost_function,
        x0,
        args=(
            n_cameras,
            n_points,
            camera_indices,
            point_indices,
            points_2d,
            camera_matrix,
        ),
        method="trf",
        ftol=1e-4,
        xtol=1e-4,
        max_nfev=200,
    )

    # Extract results
    camera_params = res.x[: n_cameras * 9].reshape((n_cameras, 9))
    points_3d = res.x[n_cameras * 9 :].reshape((n_points, 3))

    return camera_params, points_3d


def project_point(point_3d, camera_matrix, rvec, tvec):
    """Projects a single 3D point to 2D using the camera matrix, rotation vector, and translation vector."""
    point_2d, _ = cv2.projectPoints(
        np.array([point_3d]), rvec, tvec, camera_matrix, distCoeffs=None
    )
    return point_2d.reshape(2)


def estimate_initial_intrinsics(all_points1, all_points2, image_size):
    """
    Estimates the initial camera intrinsics using self-calibration via bundle adjustment.

    Parameters:
    all_points1 (np.ndarray): First set of points.
    all_points2 (np.ndarray): Second set of points.
    image_size (tuple): Size of the image (width, height).

    Returns:
    camera_matrix (np.ndarray): The estimated camera matrix.
    """
    points_2d = np.vstack((all_points1, all_points2))
    camera_indices = np.hstack(
        (np.zeros(len(all_points1)), np.ones(len(all_points2)))
    ).astype(int)
    point_indices = np.arange(len(points_2d)).astype(int)

    # Initial camera matrix guess
    camera_matrix = np.array(
        [[1, 0, image_size[0] / 2], [0, 1, image_size[1] / 2], [0, 0, 1]],
        dtype=np.float32,
    )

    camera_params, _ = bundle_adjust_self_calibration(
        points_2d, camera_indices, point_indices, camera_matrix
    )

    # Extract the camera matrix from the first camera's parameters
    fx, fy, cx, cy = camera_params[0, :4]
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    return camera_matrix


def project(points_3d, camera_matrix, rvec, tvec):
    """Projects 3D points to 2D using the camera matrix, rotation vector, and translation vector."""
    points_2d, _ = cv2.projectPoints(
        points_3d, rvec, tvec, camera_matrix, distCoeffs=None
    )
    return points_2d.reshape(-1, 2)


def reprojection_error(
    params, n_cameras, n_points, camera_indices, point_indices, points_2d, camera_matrix
):
    """Computes the reprojection error."""
    camera_params = params[: n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9 :].reshape((n_points, 3))

    errors = []
    for i in range(points_2d.shape[0]):
        camera_idx = int(camera_indices[i])
        point_idx = int(point_indices[i])
        camera_param = camera_params[camera_idx]
        rvec = camera_param[:3]
        tvec = camera_param[3:6]
        projected_point = project_point(points_3d[point_idx], camera_matrix, rvec, tvec)
        errors.append(np.linalg.norm(points_2d[i] - projected_point))
    return np.array(errors)


def bundle_adjustment(corresponding_points: list[CorrespondingPointSet]):
    """Performs bundle adjustment using CorrespondingPointSet."""
    image_size = None
    points_2d = []
    camera_indices = []
    point_indices = []
    point_set_indices = []
    point_set_map = {}
    point_counter = 0

    videos = []
    for point_set in corresponding_points:
        for video in point_set.video_points.keys():
            if video not in videos:
                videos.append(video)
            if image_size is None:
                image_size = (video.shape[2], video.shape[1])

    for point_set_idx, point_set in enumerate(corresponding_points):
        print("point_set_idx:", point_set_idx)
        print("point_set.frame_idx:", point_set.frame_idx)
        for video, point in point_set.video_points.items():
            video_idx = videos.index(video)
            print("video_idx:", video_idx)
            if (point_set_idx, video_idx) not in point_set_map:
                point_set_map[(point_set_idx, video_idx)] = point_counter
                point_counter += 1
            points_2d.append(point)
            camera_indices.append(video_idx)
            point_indices.append(point_set_map[(point_set.frame_idx, video_idx)])
            point_set_indices.append(point_set_idx)

    print("point_set_map:", point_set_map)
    print("points_2d:", points_2d)
    print("point_counter:", point_counter)

    points_2d = np.array(points_2d, dtype=np.float32)
    camera_indices = np.array(camera_indices, dtype=np.int32)
    point_indices = np.array(point_indices, dtype=np.int32)
    point_set_indices = np.array(point_set_indices, dtype=np.int32)
    n_cameras = len(corresponding_points[0].video_points)
    n_points = len(point_set_map)

    # Estimate initial camera matrix
    camera_matrix = estimate_initial_intrinsics(
        points_2d[camera_indices == 0], points_2d[camera_indices == 1], image_size
    )

    # Initial estimates
    camera_params = np.zeros((n_cameras, 9))  # [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    camera_params[:, :4] = camera_matrix[:2, :2].ravel()  # Set fx, fy, cx, cy
    points_3d = np.random.rand(n_points, 3)  # Random initial guess for 3D points

    # Correctly initialize x0 with the appropriate size
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

    res = least_squares(
        reprojection_error,
        x0,
        verbose=2,
        x_scale="jac",
        ftol=1e-6,
        method="trf",
        args=(
            n_cameras,
            n_points,
            camera_indices,
            point_indices,
            points_2d,
            camera_matrix,
        ),
    )

    optimized_params = res.x
    camera_params = optimized_params[: n_cameras * 9].reshape((n_cameras, 9))
    points_3d = optimized_params[n_cameras * 9 :].reshape((n_points, 3))
    reproj_err = res.cost

    # Extract rvecs and tvecs from camera_params
    rvecs = camera_params[:, :3]
    tvecs = camera_params[:, 3:6]

    # Compute the reprojections
    reprojected_point_sets = []
    for i in range(points_2d.shape[0]):
        camera_idx = int(camera_indices[i])
        point_idx = int(point_indices[i])
        point_set_idx = int(point_set_indices[i])
        camera_param = camera_params[camera_idx]
        rvec = camera_param[:3]
        tvec = camera_param[3:6]
        point_2d_reproj = project_point(points_3d[point_idx], camera_matrix, rvec, tvec)

        # Create a new CorrespondingPointSet with the reprojected points
        reprojected_point_set = CorrespondingPointSet(
            frame_idx=corresponding_points[point_set_idx].frame_idx,
            video_points={video: point_2d_reproj for video in videos},
            name="Reprojected Points",
        )
        reprojected_point_sets.append(reprojected_point_set)

    return camera_matrix, rvecs, tvecs, points_3d, reproj_err, reprojected_point_sets


class VideoPlayer(QWidget):
    def __init__(self, video, initial_frame=0, app: QMainWindow = None):
        super().__init__()
        self.app = app
        self.video = video
        self.current_frame = initial_frame
        self.frame = self.video[self.current_frame]
        self.image_height, self.image_width, self.image_channels = self.frame.shape
        self.points = []

        layout = QVBoxLayout()
        self.canvas = WgpuCanvas(size=(self.image_width, self.image_height))
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()

        self.image_texture = gfx.Texture(self.frame, dim=2)
        self.image = gfx.Image(
            gfx.Geometry(grid=self.image_texture),
            gfx.ImageBasicMaterial(clim=(0, 255), pick_write=True),
        )
        self.scene.add(self.image)

        self.camera = gfx.OrthographicCamera(self.image_width, self.image_height)
        self.camera.local.position = (self.image_width / 2, self.image_height / 2, 0)
        self.camera.local.scale_y = -1

        self.pan_zoom_controller = gfx.PanZoomController(
            self.camera, register_events=self.renderer
        )

        self.renderer.add_event_handler(self.on_double_click, "double_click")

        # Set the initial frame
        self.seek_frame(self.current_frame)

    def on_double_click(self, event: gfx.PointerEvent):
        if event.button == 1:
            if "index" in event.pick_info:
                x = event.pick_info["index"][0] + event.pick_info["pixel_coord"][0]
                y = event.pick_info["index"][1] + event.pick_info["pixel_coord"][1]
                self.app.add_correspondence(self, (x, y))

    def seek_frame(self, frame_idx):
        self.current_frame = frame_idx % len(self.video)
        frame = self.video[self.current_frame]
        self.image_texture.data[:, :, :] = frame
        self.image.geometry.grid.update_range((0, 0, 0), self.image_texture.size)

        for point in self.points:
            self.scene.remove(point)
        self.points = []

        cmap = plt.get_cmap("tab20")
        color_ind = -1
        for point_set in self.app.correspondences:
            if point_set.frame_idx == self.current_frame:
                color_ind += 1
                for video, point in point_set.video_points.items():
                    if video == self.video:
                        pt = gfx.Points(
                            geometry=gfx.Geometry(
                                positions=np.array(
                                    [[point[0], point[1], 1]], dtype=np.float32
                                )
                            ),
                            material=gfx.PointsMarkerMaterial(
                                marker="x",
                                size=10,
                                color=cmap(color_ind % len(cmap.colors)),
                                edge_width=0.5,
                                edge_color="white",
                            ),
                        )
                        self.points.append(pt)
                        self.scene.add(pt)

        self.render()

    def render(self):
        self.canvas.request_draw(lambda: self.renderer.render(self.scene, self.camera))

    def next_frame(self):
        self.seek_frame(self.current_frame + 1)

    def prev_frame(self):
        self.seek_frame(self.current_frame - 1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Janky Calibrate")

        self.video_paths = [
            "tests/data/minimal_session/mid/minimal_mid.mp4",
            "tests/data/minimal_session/top/minimal_top.mp4",
        ]
        self.videos = [sio.load_video(path) for path in self.video_paths]
        self.correspondences: list[CorrespondingPointSet] = []

        central_widget = QWidget()
        layout = QVBoxLayout()

        # Add side dock for point correspondences list
        self.dock_widget = QDockWidget("Correspondences", self)
        dock_contents = QWidget()
        self.dock_layout = QVBoxLayout()
        dock_contents.setLayout(self.dock_layout)
        self.dock_widget.setWidget(dock_contents)

        self.correspondence_tree = QTreeWidget()
        self.correspondence_tree.setHeaderLabels(["Video", "Point"])
        header = self.correspondence_tree.header()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.correspondence_tree.itemSelectionChanged.connect(
            self.on_tree_selection_changed
        )
        self.dock_layout.addWidget(self.correspondence_tree)

        # Add a button to run bundle adjustment
        self.bundle_adjust_button = QPushButton("Bundle Adjust")
        self.bundle_adjust_button.clicked.connect(self.bundle_adjust)
        self.dock_layout.addWidget(self.bundle_adjust_button)

        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)

        video_layout = QHBoxLayout()
        self.video1 = VideoPlayer(self.videos[0], app=self)
        self.video2 = VideoPlayer(self.videos[1], app=self)
        video_layout.addWidget(self.video1)
        video_layout.addWidget(self.video2)

        self._frame_idx = 0
        self.n_frames = len(self.videos[0])

        layout.addLayout(video_layout)

        # Horizontal layout for buttons
        button_layout = QHBoxLayout()

        # Add previous frame button
        self.prev_button = QPushButton("Previous Frame")
        self.prev_button.clicked.connect(self.prev_frame)
        button_layout.addWidget(self.prev_button)

        # Add a seekbar
        self.seekbar = QSlider(Qt.Horizontal)
        self.seekbar.setMinimum(0)
        self.seekbar.setMaximum(self.n_frames)
        self.seekbar.setValue(0)
        self.seekbar.valueChanged.connect(self.seek_frame)
        button_layout.addWidget(self.seekbar)

        # Add a frame index label
        self.frame_index_label = QLabel("0")
        button_layout.addWidget(self.frame_index_label)

        # Add next frame button
        self.next_button = QPushButton("Next Frame")
        self.next_button.clicked.connect(self.next_frame)
        button_layout.addWidget(self.next_button)

        layout.addLayout(button_layout)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Adjust layout to fit a given width, keeping the aspect ratio of the videos
        self.adjust_layout()

        # Add keyboard shortcuts for navigation
        QShortcut(QKeySequence("Left"), self, self.prev_frame)
        QShortcut(QKeySequence("Right"), self, self.next_frame)
        # Shift + right/left to skip 10 frames
        QShortcut(
            QKeySequence("Shift+Right"),
            self,
            lambda: self.seek_frame(self.frame_idx + 10),
        )
        QShortcut(
            QKeySequence("Shift+Left"),
            self,
            lambda: self.seek_frame(self.frame_idx - 10),
        )
        QShortcut(QKeySequence("Q"), self, self.close)

        # Add keyboard shortcuts for zooming
        QShortcut(QKeySequence("W"), self, self.zoom_in)
        QShortcut(QKeySequence("S"), self, self.zoom_out)

        # press space to print all correspondences
        QShortcut(QKeySequence("Space"), self, self.print_correspondences)
        QShortcut(QKeySequence("B"), self, self.bundle_adjust)

        # Initialize correspondences
        self.add_correspondence(self.video1, (236.521540885339, 425.7391152951992))
        self.add_correspondence(self.video2, (134.35598918342333, 650.816317395286))

        self.add_correspondence(self.video1, (341.147725498134, 323.338944396721))
        self.add_correspondence(self.video2, (141.04605425502163, 447.8890229198987))

        self.add_correspondence(self.video1, (619.4085950633514, 980.0346576058048))
        self.add_correspondence(self.video2, (876.934596403741, 900.5727245981037))

        self.add_correspondence(self.video1, (922.1564699784445, 441.3216385416123))
        self.add_correspondence(self.video2, (888.0841962299814, 249.42193458603253))

        self.add_correspondence(self.video1, (321.89448604452275, 500.9786351868236))
        self.add_correspondence(self.video2, (260.90631232412153, 610.3258292259762))

        self.add_correspondence(self.video1, (377.93392811630446, 441.5770096962475))
        self.add_correspondence(self.video2, (270.8303259206595, 497.85192545698294))

        self.add_correspondence(self.video1, (341.50844335757336, 573.8294826339442))
        self.add_correspondence(self.video2, (297.8463235727128, 638.4442441330538))

        self.add_correspondence(self.video1, (432.8524592763089, 474.0797029923685))
        self.add_correspondence(self.video2, (299.5003767014448, 467.5281877584905))

        self.add_correspondence(self.video1, (635.5191335723719, 904.1674520176534))
        self.add_correspondence(self.video2, (780.6867096869253, 759.0914896944737))

        self.add_correspondence(self.video1, (710.1291367123453, 840.2162648955023))
        self.add_correspondence(self.video2, (814.3527943737017, 673.0924457532038))

        self.add_correspondence(self.video1, (635.5191335723719, 904.1674520176534))
        self.add_correspondence(self.video2, (780.6867096869253, 759.0914896944737))

        self.add_correspondence(self.video1, (710.1291367123453, 840.2162648955023))
        self.add_correspondence(self.video2, (814.3527943737017, 673.0924457532038))

        self.add_correspondence(self.video1, (546.0281454391826, 613.0882810564711))
        self.add_correspondence(self.video2, (543.3560256614746, 552.7603858138051))

        self.add_correspondence(self.video1, (836.5793621729284, 571.0914448717701))
        self.add_correspondence(self.video2, (786.8234909352043, 334.2429997308254))

    def print_correspondences(self):
        for point_set in self.correspondences:
            print(point_set.video_points)

    def zoom_in(self):
        self.video1.pan_zoom_controller.zoom(0.5, rect=self.video1.renderer.rect)
        self.video2.pan_zoom_controller.zoom(0.5, rect=self.video2.renderer.rect)
        self.video1.render()
        self.video2.render()

    def zoom_out(self):
        self.video1.pan_zoom_controller.zoom(-0.5, rect=self.video1.renderer.rect)
        self.video2.pan_zoom_controller.zoom(-0.5, rect=self.video2.renderer.rect)
        self.video1.render()
        self.video2.render()

    def adjust_layout(self):
        # Get the width of the window
        window_width = 1500

        # Calculate the height based on the aspect ratio of both videos
        video1_aspect_ratio = self.video1.image_width / self.video1.image_height
        video2_aspect_ratio = self.video2.image_width / self.video2.image_height

        # Use the larger aspect ratio to ensure both videos fit
        max_aspect_ratio = max(video1_aspect_ratio, video2_aspect_ratio)

        # Calculate the height based on the maximum aspect ratio
        video_height = window_width / (
            2 * max_aspect_ratio
        )  # Divide by 2 since we have two videos side by side

        # Account for the dock widget
        window_width += self.dock_widget.width()

        # Set the size of the window
        self.resize(int(window_width), int(video_height))

        # Adjust the layout to fit the new size
        self.layout().activate()

    @property
    def frame_idx(self):
        return self._frame_idx

    @frame_idx.setter
    def frame_idx(self, value):
        if value == self.frame_idx:
            return
        self._frame_idx = value % self.n_frames

        self.frame_index_label.setText(str(self.frame_idx))
        if self.seekbar.value() != self.frame_idx:
            self.seekbar.setValue(self.frame_idx)
        self.video1.seek_frame(self.frame_idx)
        self.video2.seek_frame(self.frame_idx)

    def next_frame(self):
        self.frame_idx += 1

    def prev_frame(self):
        self.frame_idx -= 1

    def seek_frame(self, value):
        self.frame_idx = value

    def add_correspondence(self, video_player, point):
        frame_idx = self.frame_idx
        video = video_player.video

        selected_items = self.correspondence_tree.selectedItems()
        if not selected_items:
            # No point set selected, create a new one
            point_set = CorrespondingPointSet(
                frame_idx, name=f"Point set: {len(self.correspondences)}"
            )
            point_set[video] = point
            self.correspondences.append(point_set)
            item = self.add_point_set_to_tree(point_set)

        else:
            # A point set is selected
            selected_item = selected_items[0]
            if selected_item.text(0).startswith("Video"):
                selected_item = selected_item.parent()

            # Get the point set
            point_set_ind = int(selected_item.text(0).replace("Point set: ", ""))
            point_set = self.correspondences[point_set_ind]

            if video in point_set:
                # Create a new point set if the video already has a point in the current set
                point_set = CorrespondingPointSet(
                    frame_idx, name=f"Point set: {len(self.correspondences)}"
                )
                point_set[video] = point
                self.correspondences.append(point_set)
                self.add_point_set_to_tree(point_set)

            else:
                # Add the point to the point set
                point_set[video] = point
                self.update_tree_item(point_set)

        video_player.seek_frame(frame_idx)

    def add_point_set_to_tree(self, point_set):
        item = QTreeWidgetItem([point_set.name])
        self.correspondence_tree.addTopLevelItem(item)
        for video, point in point_set.video_points.items():
            video_ind = self.videos.index(video)
            child = QTreeWidgetItem([f"Video: {video_ind}", f"{point}"])
            item.addChild(child)

        item.setExpanded(True)
        self.correspondence_tree.setCurrentItem(child)

        return item

    def update_tree_item(self, point_set):
        for i in range(self.correspondence_tree.topLevelItemCount()):
            item = self.correspondence_tree.topLevelItem(i)
            if item.text(0) == point_set.name:
                for video, point in point_set.video_points.items():
                    video_ind = self.videos.index(video)

                    found_video = False
                    for j in range(item.childCount()):
                        child = item.child(j)
                        if child.text(0) == f"Video: {video_ind}":
                            found_video = True
                            break

                    if not found_video:
                        child = QTreeWidgetItem([f"Video: {video_ind}", f"{point}"])
                        item.addChild(child)

    def on_tree_selection_changed(self):
        selected_items = self.correspondence_tree.selectedItems()
        if selected_items:
            selected_item = selected_items[0]
            if selected_item.text(0).startswith("Video: "):
                selected_item = selected_item.parent()

            point_set_ind = int(selected_item.text(0).replace("Point set: ", ""))
            point_set = self.correspondences[point_set_ind]
            self.seek_frame(point_set.frame_idx)
            self.highlight_points(point_set)

    def highlight_points(self, point_set):
        for video, point in point_set.video_points.items():
            # if video == self.video1.video:
            #     self.video1.highlight_point(point)
            # elif video == self.video2.video:
            #     self.video2.highlight_point(point)
            pass

    def bundle_adjust(self):
        print("len(self.correspondences):", len(self.correspondences))
        (
            camera_matrix,
            rvecs,
            tvecs,
            points_3d,
            reprojection_error,
            reprojected_point_sets,
        ) = bundle_adjustment(self.correspondences)

        euclidean_errors = []
        for point_set, reprojected_point_set in zip(
            self.correspondences, reprojected_point_sets
        ):
            for pt, reprojected_pt in zip(
                point_set.video_points.values(),
                reprojected_point_set.video_points.values(),
            ):
                euclidean_errors.append(
                    np.linalg.norm(np.array(pt) - np.array(reprojected_pt))
                )
        euclidean_errors = np.array(euclidean_errors)

        print("Camera matrix:")
        print(camera_matrix)
        print()
        print("Rotation vectors:")
        print(rvecs)
        print()
        print("Translation vectors:")
        print(tvecs)
        print()
        print("3D points:")
        print(points_3d)
        print()
        print("Reprojection error:", reprojection_error)
        print()
        print("Mean euclidean error:", np.mean(euclidean_errors))

        # plot the reprojected points
        print(len(reprojected_point_sets))
        n_plotted = 0
        for reprojected_point_set in reprojected_point_sets:
            for video, point in reprojected_point_set.video_points.items():
                if video == self.video1.video:
                    self.video1.scene.add(
                        gfx.Points(
                            geometry=gfx.Geometry(
                                positions=np.array(
                                    [[point[0], point[1], 1]], dtype=np.float32
                                )
                            ),
                            material=gfx.PointsMarkerMaterial(
                                marker="+",
                                size=10,
                                color="red",
                            ),
                        )
                    )
                    self.video1.render()
                    n_plotted += 1
                elif video == self.video2.video:
                    self.video2.scene.add(
                        gfx.Points(
                            geometry=gfx.Geometry(
                                positions=np.array(
                                    [[point[0], point[1], 1]], dtype=np.float32
                                )
                            ),
                            material=gfx.PointsMarkerMaterial(
                                marker="+",
                                size=10,
                                color="red",
                            ),
                        )
                    )
                    n_plotted += 1
                    self.video2.render()

        print("n_plotted:", n_plotted)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
