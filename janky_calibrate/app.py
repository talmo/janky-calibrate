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
    QListWidget,
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
    rvecs = params[: n_cameras * 3].reshape((n_cameras, 3))
    tvecs = params[n_cameras * 3 : n_cameras * 6].reshape((n_cameras, 3))
    points_3d = params[n_cameras * 6 :].reshape((n_points, 3))

    error = []
    for i in range(points_2d.shape[0]):
        camera_idx = camera_indices[i]
        point_idx = point_indices[i]
        projected_point = project(
            points_3d[point_idx], camera_matrix, rvecs[camera_idx], tvecs[camera_idx]
        )
        error.append(points_2d[i] - projected_point)
    return np.concatenate(error)


def bundle_adjustment(corresponding_points, camera_matrix):
    """Performs bundle adjustment using CorrespondingPointSet."""
    points_2d = []
    camera_indices = []
    point_indices = []
    point_set_map = {}
    point_counter = 0

    for point_set in corresponding_points:
        for video, point in point_set.video_points.items():
            video_idx = corresponding_points[0].video_points.keys().index(video)
            if (point_set.frame_idx, video_idx) not in point_set_map:
                point_set_map[(point_set.frame_idx, video_idx)] = point_counter
                point_counter += 1
            points_2d.append(point)
            camera_indices.append(video_idx)
            point_indices.append(point_set_map[(point_set.frame_idx, video_idx)])

    points_2d = np.array(points_2d, dtype=np.float32)
    camera_indices = np.array(camera_indices, dtype=np.int32)
    point_indices = np.array(point_indices, dtype=np.int32)
    n_cameras = len(corresponding_points[0].video_points)
    n_points = len(point_set_map)

    # Initial estimates
    rvecs = np.zeros((n_cameras, 3))
    tvecs = np.zeros((n_cameras, 3))
    points_3d = np.random.rand(n_points, 3)  # Random initial guess for 3D points

    x0 = np.hstack((rvecs.ravel(), tvecs.ravel(), points_3d.ravel()))

    res = least_squares(
        reprojection_error,
        x0,
        verbose=2,
        x_scale="jac",
        ftol=1e-4,
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
    rvecs = optimized_params[: n_cameras * 3].reshape((n_cameras, 3))
    tvecs = optimized_params[n_cameras * 3 : n_cameras * 6].reshape((n_cameras, 3))
    points_3d = optimized_params[n_cameras * 6 :].reshape((n_points, 3))

    return rvecs, tvecs, points_3d


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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
