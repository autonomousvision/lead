import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


from PyQt6.QtCore import Qt, QPointF, QRectF, QEvent, QTimer
from PyQt6.QtGui import QBrush, QColor, QPen, QPainter, QKeySequence, QShortcut, QIntValidator, QPixmap, QCursor
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QAbstractItemView,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsSimpleTextItem,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QInputDialog,
    QToolButton,
)
QT_BACKEND = "PyQt6"

# PyQt6 enums
CTRL_MOD = Qt.KeyboardModifier.ControlModifier
USER_ROLE = Qt.ItemDataRole.UserRole
STRONG_FOCUS = Qt.FocusPolicy.StrongFocus

ITEM_SELECTABLE = QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
ITEM_MOVABLE = QGraphicsItem.GraphicsItemFlag.ItemIsMovable
ITEM_SENDS_GEOMETRY = QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges

DRAG_MODE = QGraphicsView.DragMode.RubberBandDrag
ANCHOR_MODE = QGraphicsView.ViewportAnchor.AnchorUnderMouse

EXTENDED_SELECTION = QAbstractItemView.SelectionMode.ExtendedSelection

POS_CHANGED = QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
ROT_CHANGED = QGraphicsItem.GraphicsItemChange.ItemRotationHasChanged

ANTIALIAS = QPainter.RenderHint.Antialiasing
DELETE_KEY = Qt.Key.Key_Delete

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_EXTENTS_PATH = MODULE_DIR / "util" / "static_extents.py"
DEFAULT_LANE_WIDTH = 3.5
DEFAULT_VEHICLE_EXTENTS = (2.5, 1.1)
BACKGROUND_SET_OPTIONS = {
    "[Add Set] Construction": "Construction",
    "[Add Set] Accident": "Accident",
    "[Add Set] ParkedVehicle": "ParkedVehicle",
}


@dataclass
class Obstacle:
    label: str
    object_id: str
    x: float
    y: float
    yaw: float
    extent_x: float
    extent_y: float


def index_to_label(index: int) -> str:
    # a, b, ... z, aa, ab, ...
    label = ""
    n = index
    while True:
        label = chr(ord("a") + (n % 26)) + label
        n = n // 26 - 1
        if n < 0:
            break
    return label


def load_static_extents(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    scope: dict = {}
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, path, "exec"), scope)  # trusted local file
    return scope.get("STATIC_EXTENTS", {})

def to_scene_xy(x: float, y: float, scale: float) -> QPointF:
    # CARLA: x forward, y right -> Scene: up is negative y, right is positive x
    return QPointF(y * scale, -x * scale)

def from_scene_xy(px: float, py: float, scale: float) -> tuple[float, float]:
    # Scene -> CARLA
    return (-py / scale, px / scale)

def to_scene_yaw(yaw_deg: float) -> float:
    # CARLA yaw (0 = +x forward, y right) -> Scene angle (0 = +x right)
    rad = math.radians(yaw_deg)
    x = math.cos(rad)
    y = math.sin(rad)
    dx = y
    dy = -x
    return math.degrees(math.atan2(dy, dx))

def from_scene_yaw(rot_deg: float) -> float:
    # Scene angle -> CARLA yaw
    rad = math.radians(rot_deg)
    dx = math.cos(rad)
    dy = math.sin(rad)
    x = -dy
    y = dx
    return normalize_yaw(math.degrees(math.atan2(y, x)))

def normalize_yaw(yaw_deg: float) -> float:
    yaw = yaw_deg % 360.0
    if abs(yaw - 360.0) < 1e-6 or abs(yaw) < 1e-6:
        return 0.0
    return yaw


class ObstacleItem(QGraphicsItem):
    def __init__(self, obstacle: Obstacle, scale: float = 1.0, on_change=None):
        super().__init__()
        self.obstacle = obstacle
        self.scale = scale
        self.on_change = on_change
        self.setFlags(
            ITEM_SELECTABLE | ITEM_MOVABLE | ITEM_SENDS_GEOMETRY
        )
        self.visual_mode = "edit"
        self.setZValue(1)
        self.setPos(to_scene_xy(obstacle.x, obstacle.y, scale))
        self.setRotation(to_scene_yaw(obstacle.yaw))

    def boundingRect(self) -> QRectF:
        w = self.obstacle.extent_x * 2 * self.scale
        h = self.obstacle.extent_y * 2 * self.scale
        return QRectF(-w / 2, -h / 2, w, h)

    def paint(self, painter, option, widget=None):
        rect = self.boundingRect()
        if self.visual_mode == "base":
            # In base view, foreground/custom objects take the muted overlay look.
            fill = QColor(0, 0, 0, 0)
            border = QColor(140, 140, 140, 180)
        else:
            fill = QColor(80, 160, 240, 90)
            border = QColor(40, 90, 160, 210)
        if self.isSelected():
            border = QColor(240, 120, 40, 220)
        painter.setBrush(QBrush(fill))
        pen_style = Qt.PenStyle.DashLine if self.visual_mode == "base" else Qt.PenStyle.SolidLine
        painter.setPen(QPen(border, 2, pen_style))
        painter.drawRect(rect)
        # heading line
        painter.setPen(QPen(QColor(30, 30, 30, 180), 2))
        painter.drawLine(QPointF(0, 0), QPointF(rect.width() / 2, 0))

    def itemChange(self, change, value):
        if change == POS_CHANGED:
            pos = self.pos()
            self.obstacle.x, self.obstacle.y = from_scene_xy(pos.x(), pos.y(), self.scale)
            if self.on_change:
                self.on_change(self)
        if change == ROT_CHANGED:
            self.obstacle.yaw = normalize_yaw(from_scene_yaw(self.rotation()))
            if self.on_change:
                self.on_change(self)
        return super().itemChange(change, value)

    def set_editable(self, editable: bool):
        self.setFlag(ITEM_SELECTABLE, bool(editable))
        self.setFlag(ITEM_MOVABLE, bool(editable))

    def set_visual_mode(self, mode: str):
        self.visual_mode = "base" if str(mode) == "base" else "edit"
        self.update()


class OverlayItem(QGraphicsItem):
    def __init__(self, x: float, y: float, yaw: float, extent_x: float, extent_y: float, color: QColor, scale: float):
        super().__init__()
        self.extent_x = extent_x
        self.extent_y = extent_y
        self.color = color
        self.visual_mode = "edit"
        self.scale = scale
        self.setPos(to_scene_xy(x, y, scale))
        self.setRotation(to_scene_yaw(yaw))
        self.setZValue(0)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

    def boundingRect(self) -> QRectF:
        w = self.extent_x * 2 * self.scale
        h = self.extent_y * 2 * self.scale
        return QRectF(-w / 2, -h / 2, w, h)

    def paint(self, painter, option, widget=None):
        rect = self.boundingRect()
        if self.visual_mode == "base":
            # In base view, background/overlay objects take the custom-obstacle look.
            fill = QColor(80, 160, 240, 90)
            border = QColor(40, 90, 160, 210)
            pen = QPen(border, 2, Qt.PenStyle.SolidLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QBrush(fill))
            painter.drawRect(rect)
        else:
            color = QColor(140, 140, 140, 180)
            width = 2
            pen = QPen(color, width, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.drawRect(rect)
        # heading line
        painter.drawLine(QPointF(0, 0), QPointF(rect.width() / 2, 0))

    def set_visual_mode(self, mode: str):
        self.visual_mode = "base" if str(mode) == "base" else "edit"
        self.update()


class CanvasView(QGraphicsView):
    def __init__(
        self,
        scene,
        parent=None,
        on_delete=None,
        on_readonly_interaction=None,
        on_obstacle_double_click=None,
    ):
        super().__init__(scene, parent)
        self.setRenderHints(self.renderHints() | ANTIALIAS)
        self.setDragMode(DRAG_MODE)
        self.setTransformationAnchor(ANCHOR_MODE)
        self.setFocusPolicy(STRONG_FOCUS)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.on_delete = on_delete
        self.on_readonly_interaction = on_readonly_interaction
        self.on_obstacle_double_click = on_obstacle_double_click
        self.edit_mode_enabled = True
        self.panning = False
        self.last_pan_pos = None

    def wheelEvent(self, event):
        if not self.edit_mode_enabled and event.modifiers() & CTRL_MOD and self.on_readonly_interaction:
            self.on_readonly_interaction()
        if self.edit_mode_enabled and event.modifiers() & CTRL_MOD:
            selected = [
                i
                for i in self.scene().selectedItems()
                if isinstance(i, ObstacleItem) and i.flags() & ITEM_MOVABLE
            ]
            if selected:
                delta = 5.0 if event.angleDelta().y() > 0 else -5.0
                for item in selected:
                    new_yaw = normalize_yaw(from_scene_yaw(item.rotation()) + delta)
                    item.obstacle.yaw = new_yaw
                    item.setRotation(to_scene_yaw(new_yaw))
            event.accept()
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        event.accept()
        return

    def scrollContentsBy(self, dx: int, dy: int):
        # Disable scrolling/panning via scrollbars or trackpad inertia.
        return

    def keyPressEvent(self, event):
        if not self.edit_mode_enabled and event.key() == DELETE_KEY and self.on_readonly_interaction:
            self.on_readonly_interaction()
        if self.edit_mode_enabled and event.key() == DELETE_KEY and self.on_delete:
            self.on_delete()
            event.accept()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if (
            not self.edit_mode_enabled
            and event.button() == Qt.MouseButton.LeftButton
            and self.on_readonly_interaction
        ):
            self.on_readonly_interaction()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.panning and self.last_pan_pos is not None:
            last_scene = self.mapToScene(self.last_pan_pos)
            current_scene = self.mapToScene(event.pos())
            delta = current_scene - last_scene
            self.translate(delta.x(), delta.y())
            self.last_pan_pos = event.pos()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton and self.panning:
            self.panning = False
            self.last_pan_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.on_obstacle_double_click:
            item = self.itemAt(event.pos())
            if isinstance(item, ObstacleItem):
                if not self.edit_mode_enabled and self.on_readonly_interaction:
                    self.on_readonly_interaction()
                self.on_obstacle_double_click(item)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)


class Designer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Custom Obstacle Designer")
        self.scale = 60.0  # pixels per meter
        self.default_view_zoom = 0.6
        self.static_extents = load_static_extents(DEFAULT_EXTENTS_PATH)
        self.obstacles: list[Obstacle] = []
        self.items: dict[str, ObstacleItem] = {}
        self.syncing_selection = False
        self.copied: list[Obstacle] = []
        self.overlay_items: list[OverlayItem] = []
        self.road_items = []
        self.axis_items = []
        self.allowed_object_ids = None
        self.allowed_overlay_types = None
        self.edit_mode_enabled = True
        self.construction_preview_params = None
        self.altered_construction_editor_mode = False
        self._suppress_auto_switch = False
        self.prop_preview_dir = MODULE_DIR / "images" / "carla_props_0.9.15"
        self.prop_preview_cache = {}
        self.preview_combo_viewports = {}
        self.preview_combo_widgets = {}
        self.preview_combo_lineedits = {}

        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(-5000, -5000, 10000, 10000)
        self.view = CanvasView(
            self.scene,
            self,
            on_delete=self.delete_selected,
            on_readonly_interaction=self._switch_to_edit_mode_for_interaction,
            on_obstacle_double_click=self._on_obstacle_item_double_clicked,
        )
        self.view.setBackgroundBrush(QBrush(QColor(245, 245, 245)))
        self.view.setObjectName("previewCanvas")

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(EXTENDED_SELECTION)
        self.list_widget.itemSelectionChanged.connect(self.on_list_selection_changed)
        self.list_widget.itemDoubleClicked.connect(self._on_list_item_double_clicked)
        self.list_widget.setMouseTracking(True)
        self.list_widget.viewport().setMouseTracking(True)

        self.object_combo = QComboBox()
        self.object_combo.setEditable(True)
        self.object_combo.setMinimumContentsLength(28)
        self.object_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.object_combo.view().setMouseTracking(True)
        self.object_combo.view().setTextElideMode(Qt.TextElideMode.ElideNone)
        self.object_combo.view().viewport().setMouseTracking(True)
        self._refresh_object_combo_items(sorted(self.static_extents.keys()))
        self.register_prop_preview_combo(self.object_combo)

        self.prop_preview_popup = QLabel(None, Qt.WindowType.ToolTip)
        self.prop_preview_popup.setStyleSheet(
            "QLabel { background: #ffffff; border: 1px solid #9a9a9a; padding: 2px; }"
        )
        self.prop_preview_popup.hide()
        self.prop_preview_size = (260, 180)

        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self.add_obstacle)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.remove_selected)

        self.extent_label = QLabel("Extent: -")
        self.help_label = QLabel("Drag to move. Ctrl+Wheel rotate 5 deg. Wheel zoom. Ctrl+C/V copy/paste.")

        self.overlay_group = QGroupBox("Baseline scenario")
        overlay_layout = QFormLayout(self.overlay_group)
        self.overlay_enabled = QCheckBox("Show baseline")
        self.overlay_enabled.setChecked(True)
        self.overlay_type = QComboBox()
        self.overlay_type.addItems(["Construction", "Accident", "ParkedVehicle"])
        self.overlay_direction = QComboBox()
        self.overlay_direction.addItems(["right", "left"])
        self.overlay_lane_width = None
        self.road_enabled = QCheckBox("Show road")
        self.road_enabled.setChecked(True)
        self.overlay_distance = None
        overlay_layout.addRow(self.overlay_enabled)
        overlay_layout.addRow("Scenario", self.overlay_type)
        overlay_layout.addRow("Side", self.overlay_direction)
        overlay_layout.addRow(self.road_enabled)

        self.overlay_enabled.stateChanged.connect(self.update_overlay)
        self.overlay_type.currentIndexChanged.connect(self.update_overlay)
        self.overlay_direction.currentIndexChanged.connect(self.update_overlay)
        self.road_enabled.stateChanged.connect(self.update_road)

        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(-10000, 10000)
        self.x_spin.setDecimals(2)
        self.x_spin.valueChanged.connect(self.on_spin_changed)
        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(-10000, 10000)
        self.y_spin.setDecimals(2)
        self.y_spin.valueChanged.connect(self.on_spin_changed)
        self.yaw_spin = QDoubleSpinBox()
        self.yaw_spin.setRange(0, 360)
        self.yaw_spin.setDecimals(1)
        self.yaw_spin.valueChanged.connect(self.on_spin_changed)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.export_btn = QPushButton("Export <objects/>")
        self.export_btn.clicked.connect(self.export_objects)
        self.copy_btn = QPushButton("Copy to Clipboard")
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        self.load_btn = QPushButton("Load from XML")
        self.load_btn.clicked.connect(self.load_from_xml_dialog)

        self.delete_shortcut = QShortcut(QKeySequence("Delete"), self)
        self.delete_shortcut.activated.connect(self.delete_selected)
        self.copy_shortcut = QShortcut(QKeySequence("Ctrl+C"), self)
        self.copy_shortcut.activated.connect(self.copy_selected)
        self.paste_shortcut = QShortcut(QKeySequence("Ctrl+V"), self)
        self.paste_shortcut.activated.connect(self.paste_copied)

        self.distance_row = QWidget()
        self.distance_row_layout = QHBoxLayout(self.distance_row)
        self.distance_row_layout.setContentsMargins(0, 0, 0, 0)
        self.distance_label = QLabel("Distance")
        self.distance_input = QLineEdit()
        self.distance_input.setValidator(QIntValidator())
        self.distance_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.distance_input.setText("100")
        self.distance_row_layout.addWidget(self.distance_label)
        self.distance_row_layout.addWidget(self.distance_input)
        self.distance_row.setVisible(False)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Objects"))
        left_layout.addWidget(self.list_widget)

        obj_row = QHBoxLayout()
        obj_row.addWidget(self.object_combo)
        obj_row.addWidget(self.add_btn)
        obj_row.addWidget(self.remove_btn)
        obj_row.addWidget(self.load_btn)
        left_layout.addLayout(obj_row)
        left_layout.addWidget(self.extent_label)
        left_layout.addWidget(self.help_label)
        left_layout.addWidget(self.overlay_group)

        prop_toggle_row = QHBoxLayout()
        self.prop_toggle_btn = QToolButton()
        self.prop_toggle_btn.setText("Show Transform (X/Y/Yaw)")
        self.prop_toggle_btn.setCheckable(True)
        self.prop_toggle_btn.setChecked(False)
        prop_toggle_row.addWidget(self.prop_toggle_btn)
        prop_toggle_row.addStretch(1)
        left_layout.addLayout(prop_toggle_row)

        prop_box = QGroupBox("Selected")
        prop_layout = QFormLayout(prop_box)
        prop_layout.addRow("X", self.x_spin)
        prop_layout.addRow("Y", self.y_spin)
        prop_layout.addRow("Yaw", self.yaw_spin)
        prop_box.setVisible(False)

        def toggle_transform(checked):
            prop_box.setVisible(bool(checked))
            self.prop_toggle_btn.setText("Hide Transform (X/Y/Yaw)" if checked else "Show Transform (X/Y/Yaw)")

        self.prop_toggle_btn.toggled.connect(toggle_transform)
        left_layout.addWidget(prop_box)

        left_layout.addWidget(self.distance_row)

        self.output_label = QLabel("Output")
        left_layout.addWidget(self.output_label)
        left_layout.addWidget(self.output)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.export_btn)
        btn_row.addWidget(self.copy_btn)
        left_layout.addLayout(btn_row)

        self.mode_button_group = QButtonGroup(self)
        self.mode_button_group.setExclusive(True)
        self.base_view_button = QPushButton("View Base Scenario")
        self.base_view_button.setCheckable(True)
        self.custom_edit_button = QPushButton("Edit Custom Scenario")
        self.custom_edit_button.setCheckable(True)
        self.custom_edit_button.setChecked(True)
        self.mode_button_group.addButton(self.base_view_button)
        self.mode_button_group.addButton(self.custom_edit_button)
        self.mode_switch_box = QWidget(self.view.viewport())
        self.mode_switch_box.setObjectName("editorModeSwitch")
        mode_switch_box_layout = QHBoxLayout(self.mode_switch_box)
        mode_switch_box_layout.setContentsMargins(8, 6, 8, 6)
        mode_switch_box_layout.setSpacing(8)
        mode_switch_box_layout.addWidget(self.base_view_button)
        mode_switch_box_layout.addWidget(self.custom_edit_button)
        self.mode_switch_box.show()

        self.legend_box = QWidget(self.view.viewport())
        self.legend_box.setObjectName("editorLegend")
        legend_layout = QVBoxLayout(self.legend_box)
        legend_layout.setContentsMargins(10, 8, 10, 8)
        legend_layout.setSpacing(8)

        title = QLabel("Legend")
        title.setObjectName("editorLegendTitle")
        legend_layout.addWidget(title)

        custom_row = QHBoxLayout()
        custom_row.setContentsMargins(0, 4, 0, 4)
        custom_row.setSpacing(6)
        self.legend_custom_swatch = QLabel()
        self.legend_custom_swatch.setFixedSize(14, 14)
        self.legend_custom_text = QLabel("Custom objects")
        custom_row.addWidget(self.legend_custom_swatch)
        custom_row.addWidget(self.legend_custom_text)
        custom_row.addStretch()
        legend_layout.addLayout(custom_row)

        overlay_row = QHBoxLayout()
        overlay_row.setContentsMargins(0, 4, 0, 4)
        overlay_row.setSpacing(6)
        self.legend_overlay_swatch = QLabel()
        self.legend_overlay_swatch.setFixedSize(14, 14)
        self.legend_overlay_text = QLabel("Base overlay")
        overlay_row.addWidget(self.legend_overlay_swatch)
        overlay_row.addWidget(self.legend_overlay_text)
        overlay_row.addStretch()
        legend_layout.addLayout(overlay_row)

        self.legend_box.show()
        self.view.viewport().installEventFilter(self)
        self.list_widget.viewport().installEventFilter(self)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        right_layout.addWidget(self.view, stretch=1)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(splitter)

        self.scene.selectionChanged.connect(self.on_scene_selection)
        self.base_view_button.clicked.connect(lambda: self.set_edit_mode(False))
        self.custom_edit_button.clicked.connect(lambda: self.set_edit_mode(True))
        self.draw_grid()
        self.update_road()
        self.update_overlay()
        self.set_edit_mode(True)
        self.setStyleSheet(
            """
            QWidget#editorModeSwitch {
                border: 1px solid #b6b6b6;
                border-radius: 8px;
                background: #f4f4f4;
                padding: 4px;
            }
            QWidget#editorModeSwitch QPushButton {
                padding: 4px 10px;
                border-radius: 6px;
                border: 1px solid #b0b0b0;
                background: #ffffff;
                font-weight: 500;
                font-size: 10px;
            }
            QWidget#editorModeSwitch QPushButton:checked {
                background: #1f7a45;
                color: #ffffff;
                border: 1px solid #1f7a45;
            }
            QWidget#editorLegend {
                border: 1px solid #b6b6b6;
                border-radius: 8px;
                background: rgba(244, 244, 244, 235);
            }
            QLabel#editorLegendTitle {
                font-weight: 600;
                font-size: 11px;
                margin-bottom: 2px;
            }
            QWidget#editorLegend QLabel {
                font-size: 11px;
                font-weight: 400;
            }
            """
        )
        self._position_mode_switch()
        self._position_legend()
        QTimer.singleShot(0, self._apply_default_zoom)

    def _apply_default_zoom(self):
        self.view.resetTransform()
        self.view.scale(self.default_view_zoom, self.default_view_zoom)
        self.view.centerOn(0.0, 0.0)

    def configure_editor_profile(self, editor_profile: dict | None):
        """
        Apply profile-based restrictions for unified scenario editing.
        """
        if not isinstance(editor_profile, dict):
            return

        object_mode = str(editor_profile.get("object_mode", "any"))
        overlay_types = editor_profile.get("overlay_types", None)
        self._apply_object_mode(object_mode)
        self._apply_overlay_types(overlay_types)

    def _apply_object_mode(self, object_mode: str):
        all_ids = sorted(self.static_extents.keys())
        if object_mode == "vehicle_only":
            filtered = [x for x in all_ids if x.startswith("vehicle.")]
        elif object_mode == "props_only":
            filtered = [x for x in all_ids if x.startswith("static.prop.")]
        elif object_mode == "construction":
            tokens = ("constructioncone", "trafficwarning", "dirtdebris", "barrier", "warning")
            filtered = [x for x in all_ids if any(token in x for token in tokens)]
            if not filtered:
                filtered = [x for x in all_ids if x.startswith("static.prop.")]
        else:
            filtered = all_ids

        self.allowed_object_ids = set(filtered)
        self._refresh_object_combo_items(filtered)

    def _refresh_object_combo_items(self, object_ids):
        self.object_combo.clear()
        self.object_combo.addItems(list(BACKGROUND_SET_OPTIONS.keys()))
        self.object_combo.insertSeparator(self.object_combo.count())
        self.object_combo.addItems(object_ids)

    def register_prop_preview_combo(self, combo_box: QComboBox):
        view = combo_box.view()
        viewport = view.viewport()
        view.setMouseTracking(True)
        viewport.setMouseTracking(True)
        viewport.installEventFilter(self)
        self.preview_combo_viewports[viewport] = combo_box
        combo_box.setMouseTracking(True)
        combo_box.installEventFilter(self)
        self.preview_combo_widgets[combo_box] = combo_box
        line_edit = combo_box.lineEdit()
        if line_edit is not None:
            line_edit.setMouseTracking(True)
            line_edit.installEventFilter(self)
            self.preview_combo_lineedits[line_edit] = combo_box

    def make_static_prop_selector(self, current_value: str = "") -> QComboBox:
        combo_box = QComboBox(self)
        combo_box.setEditable(False)
        combo_box.setMinimumContentsLength(28)
        combo_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo_box.addItems(sorted(self.static_extents.keys()))
        combo_box.view().setTextElideMode(Qt.TextElideMode.ElideNone)
        if current_value and combo_box.findText(str(current_value)) >= 0:
            combo_box.setCurrentText(str(current_value))
        self.register_prop_preview_combo(combo_box)
        return combo_box

    def _pick_existing_object_id(self, candidates, prefix_fallback=""):
        for object_id in candidates:
            if object_id in self.static_extents:
                return object_id
        if prefix_fallback:
            matches = [x for x in self.static_extents.keys() if x.startswith(prefix_fallback)]
            if matches:
                return sorted(matches)[0]
        return None

    def _vehicle_ids_for_background(self, count: int):
        preferred = [
            "vehicle.audi.tt",
            "vehicle.tesla.model3",
            "vehicle.dodge.charger_police",
            "vehicle.mercedes.coupe",
            "vehicle.nissan.patrol",
        ]
        available = [x for x in preferred if x in self.static_extents]
        if len(available) < count:
            all_vehicle_ids = sorted([x for x in self.static_extents.keys() if x.startswith("vehicle.")])
            for vehicle_id in all_vehicle_ids:
                if vehicle_id not in available:
                    available.append(vehicle_id)
                if len(available) >= count:
                    break
        if not available:
            return []
        out = []
        for i in range(count):
            out.append(available[i % len(available)])
        return out

    def _add_background_set(self, scenario_name: str):
        direction = self.overlay_direction.currentText()
        objects_to_add = []
        if scenario_name == "Construction":
            overlay_objs = self._overlay_construction(0.0, DEFAULT_LANE_WIDTH, direction)
            warning_id = self._pick_existing_object_id(
                ["static.prop.trafficwarning", "static.prop.warningconstruction"],
                prefix_fallback="static.prop.",
            )
            debris_id = self._pick_existing_object_id(
                ["static.prop.dirtdebris02", "static.prop.dirtdebris01"],
                prefix_fallback="static.prop.",
            )
            cone_id = self._pick_existing_object_id(
                ["static.prop.constructioncone", "static.prop.trafficcone01", "static.prop.trafficcone02"],
                prefix_fallback="static.prop.",
            )
            for idx, obj in enumerate(overlay_objs):
                if idx == 0:
                    object_id = warning_id
                elif idx == 1:
                    object_id = debris_id
                else:
                    object_id = cone_id
                if object_id:
                    objects_to_add.append((object_id, obj["x"], obj["y"], obj["yaw"]))
        elif scenario_name == "Accident":
            overlay_objs = self._overlay_accident(0.0, DEFAULT_LANE_WIDTH, direction)
            vehicle_ids = self._vehicle_ids_for_background(len(overlay_objs))
            for obj, object_id in zip(overlay_objs, vehicle_ids):
                objects_to_add.append((object_id, obj["x"], obj["y"], obj["yaw"]))
        elif scenario_name == "ParkedVehicle":
            overlay_objs = self._overlay_parked(0.0, DEFAULT_LANE_WIDTH, direction)
            vehicle_ids = self._vehicle_ids_for_background(len(overlay_objs))
            for obj, object_id in zip(overlay_objs, vehicle_ids):
                objects_to_add.append((object_id, obj["x"], obj["y"], obj["yaw"]))

        if not objects_to_add:
            QMessageBox.warning(self, "Add Set Failed", "Could not resolve matching object IDs for this set.")
            return

        for object_id, x, y, yaw in objects_to_add:
            self.add_obstacle_with_params(object_id, x, y, yaw, allow_disallowed=True)
        self.sync_selection_from_list()

    def _apply_overlay_types(self, overlay_types):
        if not overlay_types:
            return
        allowed = [str(x) for x in overlay_types if str(x) in ("Construction", "Accident", "ParkedVehicle")]
        if not allowed:
            return
        self.allowed_overlay_types = tuple(allowed)
        current = self.overlay_type.currentText()
        self.overlay_type.blockSignals(True)
        self.overlay_type.clear()
        self.overlay_type.addItems(list(self.allowed_overlay_types))
        if current in self.allowed_overlay_types:
            self.overlay_type.setCurrentText(current)
        self.overlay_type.blockSignals(False)
        self.update_overlay()

    def get_objects_attr_map(self) -> dict:
        """
        Build the XML-style <objects> attribute map from current obstacles.
        """
        objects = {}
        for obs in self.obstacles:
            objects[obs.label] = f"id={obs.object_id} x={obs.x:.2f} y={obs.y:.2f} yaw={obs.yaw:.2f}"
        return objects

    def load_from_objects_attr_map(
        self,
        objects_attr_map: dict,
        allow_disallowed: bool = True,
        select_last: bool = True,
    ):
        """
        Load obstacles from an XML-style <objects> attribute map.
        """
        if not isinstance(objects_attr_map, dict):
            return

        def parse_kv(s: str) -> dict:
            parts = str(s).split()
            out = {}
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    out[k] = v
            return out

        entries = []
        for key in sorted(objects_attr_map.keys()):
            raw = objects_attr_map[key]
            kv = parse_kv(raw)
            object_id = kv.get("id")
            if not object_id:
                continue
            try:
                x = float(kv.get("x", 0.0))
                y = float(kv.get("y", 0.0))
                yaw = float(kv.get("yaw", 0.0))
            except ValueError:
                continue
            entries.append((object_id, x, y, yaw))

        self.clear_all_objects()
        for object_id, x, y, yaw in entries:
            self.add_obstacle_with_params(object_id, x, y, yaw, allow_disallowed=allow_disallowed)
        if select_last and self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(self.list_widget.count() - 1)
            self.sync_selection_from_list()
        else:
            self.list_widget.clearSelection()
            self.scene.clearSelection()

    def set_overlay_selection(self, overlay_type: str, direction: str):
        """
        Configure the overlay controls from stored values.
        """
        overlay_type_map = {
            "none": "Construction",
            "construction": "Construction",
            "accident": "Accident",
            "parkedvehicle": "ParkedVehicle",
            "parked": "ParkedVehicle",
        }
        overlay_text = overlay_type_map.get(str(overlay_type).lower(), "Construction")
        direction_text = "left" if str(direction).lower() == "left" else "right"
        show_overlay = str(overlay_type).lower() != "none"

        self.overlay_enabled.blockSignals(True)
        self.overlay_type.blockSignals(True)
        self.overlay_direction.blockSignals(True)
        self.overlay_enabled.setChecked(show_overlay)
        self.overlay_type.setCurrentText(overlay_text)
        self.overlay_direction.setCurrentText(direction_text)
        self.overlay_enabled.blockSignals(False)
        self.overlay_type.blockSignals(False)
        self.overlay_direction.blockSignals(False)
        self.update_overlay()

    def get_overlay_selection(self) -> tuple[str, str]:
        """
        Return normalized overlay type and direction values.
        """
        overlay_map = {
            "Construction": "construction",
            "Accident": "accident",
            "ParkedVehicle": "parkedvehicle",
        }
        overlay_type = overlay_map.get(self.overlay_type.currentText(), "construction")
        direction = "left" if self.overlay_direction.currentText() == "left" else "right"
        return overlay_type, direction

    def configure_for_route_builder(self):
        """
        Hide standalone export/output controls when embedded in the route builder.
        """
        self.output_label.setVisible(False)
        self.output.setVisible(False)
        self.export_btn.setVisible(False)
        self.copy_btn.setVisible(False)
        self.distance_row.setVisible(True)

    def configure_for_altered_construction_editor(self):
        """
        Configure a constrained construction preview editor:
        - fixed baseline scenario type (Construction)
        - no custom objects list / add-remove controls
        """
        self.configure_for_route_builder()
        self.altered_construction_editor_mode = True
        self.distance_row.setVisible(False)
        self.list_widget.setVisible(False)
        self.object_combo.setVisible(False)
        self.add_btn.setVisible(False)
        self.remove_btn.setVisible(False)
        self.load_btn.setVisible(False)
        self.extent_label.setVisible(False)
        self.help_label.setVisible(False)
        self.prop_toggle_btn.setVisible(False)
        for label in self.findChildren(QLabel):
            if label.text().strip() == "Objects":
                label.setVisible(False)
        self.overlay_type.setEnabled(False)
        self._apply_overlay_types(["Construction"])
        self.overlay_type.setCurrentText("Construction")
        self.update_overlay()

    def configure_for_bad_parking_editor(self):
        """
        Configure a constrained bad-parking preview editor:
        - fixed baseline scenario type (ParkedVehicle)
        - single custom vehicle expected (type/x/y/yaw)
        """
        self.configure_for_route_builder()
        self.distance_row.setVisible(True)
        self.list_widget.setVisible(False)
        self.object_combo.setVisible(False)
        self.add_btn.setVisible(False)
        self.remove_btn.setVisible(False)
        self.load_btn.setVisible(False)
        self.extent_label.setVisible(False)
        self.help_label.setVisible(False)
        self.prop_toggle_btn.setChecked(True)  # auto-expand X/Y/Yaw panel
        self.mode_switch_box.setVisible(False)  # bad parking is always in edit mode
        for label in self.findChildren(QLabel):
            if label.text().strip() == "Objects":
                label.setVisible(False)
        self.overlay_enabled.setChecked(True)
        self.overlay_enabled.setEnabled(False)
        self.overlay_type.setEnabled(False)
        self._apply_overlay_types(["ParkedVehicle"])
        self.overlay_type.setCurrentText("ParkedVehicle")
        self.delete_shortcut.setEnabled(False)
        self.copy_shortcut.setEnabled(False)
        self.paste_shortcut.setEnabled(False)
        self.update_overlay()

    def set_parameter_widgets(self, entries):
        """
        Replace the parameter row content (where distance is shown in builder mode).
        entries: list[tuple[str, QWidget]]
        """
        while self.distance_row_layout.count():
            item = self.distance_row_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        if not entries:
            self.distance_row_layout.addWidget(self.distance_label)
            self.distance_row_layout.addWidget(self.distance_input)
            return

        for label_text, widget in entries:
            if label_text:
                label = QLabel(label_text)
                self.distance_row_layout.addWidget(label)
            self.distance_row_layout.addWidget(widget)

    def set_distance_value(self, value):
        try:
            self.distance_input.setText(str(int(float(value))))
        except (TypeError, ValueError):
            self.distance_input.setText("")

    def set_construction_preview_params(self, warning_sign=None, debris=None, cones=None):
        self.construction_preview_params = {
            "warning_sign": warning_sign,
            "debris": debris,
            "cones": cones,
        }
        if self.altered_construction_editor_mode:
            self._rebuild_altered_construction_custom_objects()
        self.update_overlay()

    def _rebuild_altered_construction_custom_objects(self):
        direction = self.overlay_direction.currentText()
        preview = self.construction_preview_params or {}
        objects = self._overlay_construction(0.0, DEFAULT_LANE_WIDTH, direction, preview_params=preview)
        objects_attr_map = {}
        for i, obj in enumerate(objects):
            object_id = str(obj.get("object_id", "")).strip()
            if not object_id:
                continue
            label = index_to_label(i)
            objects_attr_map[label] = f"id={object_id} x={obj['x']:.2f} y={obj['y']:.2f} yaw={obj['yaw']:.2f}"
        self.load_from_objects_attr_map(objects_attr_map, allow_disallowed=True, select_last=False)

    def get_distance_value(self):
        text = self.distance_input.text().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def draw_grid(self):
        grid_pen = QPen(QColor(200, 200, 200), 1)
        bold_pen = QPen(QColor(170, 170, 170), 1)
        step = self.scale
        grid_extent = 5000
        for x in range(-grid_extent, grid_extent + 1, int(step)):
            pen = bold_pen if x == 0 else grid_pen
            self.scene.addLine(x, -grid_extent, x, grid_extent, pen)
        for y in range(-grid_extent, grid_extent + 1, int(step)):
            pen = bold_pen if y == 0 else grid_pen
            self.scene.addLine(-grid_extent, y, grid_extent, y, pen)

        # Coordinate frame at origin (CARLA: +x forward/up, +y right)
        axis_len = self.scale * 0.6
        x_pen = QPen(QColor(200, 60, 60), 2)
        y_pen = QPen(QColor(60, 160, 60), 2)
        x_pen.setCosmetic(True)
        y_pen.setCosmetic(True)
        # +x (forward) is up in scene
        self.axis_items.append(self.scene.addLine(0, 0, 0, -axis_len, x_pen))
        # +y (right) is right in scene
        self.axis_items.append(self.scene.addLine(0, 0, axis_len, 0, y_pen))

    def clear_overlay(self):
        for item in self.overlay_items:
            self.scene.removeItem(item)
        self.overlay_items = []

    def clear_road(self):
        for item in self.road_items:
            self.scene.removeItem(item)
        self.road_items = []

    def update_road(self):
        self.clear_road()
        if not self.road_enabled.isChecked():
            return
        lane_width = DEFAULT_LANE_WIDTH
        extent = 5000
        # Lane boundaries for three lanes: ego lane centered at y=0 with one lane to each side
        right_edge = lane_width * 1.5
        right_middle = lane_width * 0.5
        left_middle = -lane_width * 0.5
        left_edge = -lane_width * 1.5
        # Convert CARLA y to scene x (right), so y -> x
        def scene_x(y):
            return y * self.scale
        pen_edge = QPen(QColor(120, 120, 120), 2)
        pen_edge.setCosmetic(True)
        # edges + lane divider (two lanes)
        self.road_items.append(self.scene.addLine(scene_x(left_edge), -extent, scene_x(left_edge), extent, pen_edge))
        self.road_items.append(self.scene.addLine(scene_x(left_middle), -extent, scene_x(left_middle), extent, pen_edge))
        self.road_items.append(self.scene.addLine(scene_x(right_middle), -extent, scene_x(right_middle), extent, pen_edge))
        self.road_items.append(self.scene.addLine(scene_x(right_edge), -extent, scene_x(right_edge), extent, pen_edge))

    def update_overlay(self):
        self.clear_overlay()
        if not self.overlay_enabled.isChecked():
            return
        overlay_type = self.overlay_type.currentText()
        direction = self.overlay_direction.currentText()
        lane_width = DEFAULT_LANE_WIDTH
        distance = 0.0

        objects = []
        if overlay_type == "Construction":
            objects = self._overlay_construction(distance, lane_width, direction)
        elif overlay_type == "Accident":
            objects = self._overlay_accident(distance, lane_width, direction)
        elif overlay_type == "ParkedVehicle":
            objects = self._overlay_parked(distance, lane_width, direction)

        gray = QColor(140, 140, 140, 220)
        for obj in objects:
            item = OverlayItem(
                obj["x"],
                obj["y"],
                obj["yaw"],
                obj["extent_x"],
                obj["extent_y"],
                gray,
                self.scale,
            )
            item.set_visual_mode("edit" if self.edit_mode_enabled else "base")
            self.scene.addItem(item)
            self.overlay_items.append(item)

    def _extent_for(self, object_id: str, default=(0.2, 0.2)):
        ext = self.static_extents.get(object_id)
        if not ext:
            return default
        return float(ext[0]), float(ext[1])

    def _forward(self, yaw_deg: float):
        rad = math.radians(yaw_deg)
        return math.cos(rad), math.sin(rad)

    def _right(self, yaw_deg: float):
        rad = math.radians(yaw_deg)
        return -math.sin(rad), math.cos(rad)

    def _overlay_construction(self, distance: float, lane_width: float, direction: str, preview_params: dict | None = None):
        objs = []
        # base transform at x=0, y=0, yaw=0 (trigger point)
        start_x, start_y, start_yaw = 0.0, 0.0, 0.0
        preview = preview_params or {}
        warning_id = str(preview.get("warning_sign") or "static.prop.trafficwarning")
        debris_id = str(preview.get("debris") or "static.prop.dirtdebris02")
        cones_mask = str(preview.get("cones") or "1111111")
        cone_id = self._pick_existing_object_id(
            ["static.prop.constructioncone", "static.prop.trafficcone01", "static.prop.trafficcone02"],
            prefix_fallback="static.prop.",
        ) or "static.prop.constructioncone"
        # traffic warning
        warn_extent = self._extent_for(warning_id, default=(1.2, 1.4))
        warn_x = start_x - 5.0
        warn_y = start_y
        if warning_id.lower() != "none":
            objs.append(
                {
                    "x": warn_x,
                    "y": warn_y,
                    "yaw": 270.0,
                    "extent_x": warn_extent[0],
                    "extent_y": warn_extent[1],
                    "color": QColor(200, 80, 80, 180),
                    "object_id": warning_id,
                    "kind": "warning",
                }
            )
        # debris
        debris_extent = self._extent_for(debris_id, default=(0.9, 0.75))
        debris_x = start_x + 2.0
        debris_y = start_y
        if debris_id.lower() != "none":
            objs.append(
                {
                    "x": debris_x,
                    "y": debris_y,
                    "yaw": 90.0,
                    "extent_x": debris_extent[0],
                    "extent_y": debris_extent[1],
                    "color": QColor(120, 80, 40, 180),
                    "object_id": debris_id,
                    "kind": "debris",
                }
            )
        # cones setup
        cone_extent = self._extent_for("static.prop.constructioncone", default=(0.17, 0.17))
        k = 0.85 * lane_width / 2.0
        side_y = -k if direction == "right" else k
        yaw = 0.0
        lengths = [4, 3]
        offsets = [2, 1]
        cur_x, cur_y = start_x, side_y
        cur_yaw = yaw
        cone_start_idx = len(objs)
        for i in range(len(lengths)):
            total = lengths[i] * offsets[i]
            dist = 0.0
            fx, fy = self._forward(cur_yaw)
            while dist < total:
                dist += offsets[i]
                objs.append(
                    {
                        "x": cur_x + fx * dist,
                        "y": cur_y + fy * dist,
                        "yaw": cur_yaw,
                        "extent_x": cone_extent[0],
                        "extent_y": cone_extent[1],
                        "color": QColor(255, 140, 0, 200),
                        "object_id": cone_id,
                        "cone_index": len(objs) - cone_start_idx,
                        "kind": "cone",
                    }
                )
            # advance and turn
            cur_x += fx * total
            cur_y += fy * total
            if i == 0 and direction == "left":
                cur_yaw -= 90.0
            else:
                cur_yaw += 90.0
        if cones_mask:
            cones = objs[cone_start_idx:]
            non_cones = objs[:cone_start_idx]
            filtered_cones = [cone for cone, bit in zip(cones, cones_mask) if bit == "1"]
            objs = non_cones + filtered_cones
        return objs

    def _overlay_accident(self, distance: float, lane_width: float, direction: str):
        objs = []
        offset = 0.6 * lane_width / 2.0
        y = offset if direction == "right" else -offset
        extent_x, extent_y = DEFAULT_VEHICLE_EXTENTS
        xs = [0.0, 10.0, 16.0]
        for x in xs:
            objs.append(
                {
                    "x": x,
                    "y": y,
                    "yaw": 0.0,
                    "extent_x": extent_x,
                    "extent_y": extent_y,
                    "color": QColor(60, 120, 200, 180),
                }
            )
        return objs

    def _overlay_parked(self, distance: float, lane_width: float, direction: str):
        objs = []
        offset = 0.7 * lane_width / 2.0
        y = offset if direction == "right" else -offset
        extent_x, extent_y = DEFAULT_VEHICLE_EXTENTS
        objs.append(
            {
                "x": 0.0,
                "y": y,
                "yaw": 0.0,
                "extent_x": extent_x,
                "extent_y": extent_y,
                "color": QColor(60, 160, 120, 180),
            }
        )
        return objs

    def current_obstacle(self) -> Obstacle | None:
        items = self.list_widget.selectedItems()
        if len(items) != 1:
            return None
        label = items[0].data(USER_ROLE)
        for obs in self.obstacles:
            if obs.label == label:
                return obs
        return None

    def add_obstacle(self):
        if not self.edit_mode_enabled:
            self._switch_to_edit_mode_for_interaction()
        object_id = self.object_combo.currentText().strip()
        if not object_id:
            return
        background_set = BACKGROUND_SET_OPTIONS.get(object_id)
        if background_set is not None:
            self._add_background_set(background_set)
            return
        if self.allowed_object_ids is not None and object_id not in self.allowed_object_ids:
            QMessageBox.warning(self, "Object Not Allowed", "This object is not allowed for this scenario type.")
            return
        extent_x, extent_y = self._get_extent_for_object_id(object_id)
        label = index_to_label(len(self.obstacles))
        offset_step = 0.5
        cols = 5
        idx = len(self.obstacles)
        offset_x = (idx % cols) * offset_step
        offset_y = (idx // cols) * offset_step
        obs = Obstacle(
            label=label,
            object_id=object_id,
            x=offset_x,
            y=offset_y,
            yaw=0.0,
            extent_x=extent_x,
            extent_y=extent_y,
        )
        self.obstacles.append(obs)
        item = QListWidgetItem(f"{label}: {object_id}")
        item.setData(USER_ROLE, label)
        self.list_widget.addItem(item)
        self.add_item_to_scene(obs)
        self.list_widget.clearSelection()
        self.list_widget.setCurrentItem(item)
        self.sync_selection_from_list()

    def add_obstacle_with_params(self, object_id: str, x: float, y: float, yaw: float, allow_disallowed: bool = False):
        if not allow_disallowed and self.allowed_object_ids is not None and object_id not in self.allowed_object_ids:
            return None
        extent_x, extent_y = self._get_extent_for_object_id(object_id)
        label = index_to_label(len(self.obstacles))
        obs = Obstacle(
            label=label,
            object_id=object_id,
            x=x,
            y=y,
            yaw=normalize_yaw(yaw),
            extent_x=extent_x,
            extent_y=extent_y,
        )
        self.obstacles.append(obs)
        item = QListWidgetItem(f"{label}: {object_id}")
        item.setData(USER_ROLE, label)
        self.list_widget.addItem(item)
        self.add_item_to_scene(obs)
        return obs

    def add_item_to_scene(self, obs: Obstacle):
        item = ObstacleItem(obs, scale=self.scale, on_change=self.on_item_changed)
        if self.altered_construction_editor_mode:
            item.setFlag(ITEM_SELECTABLE, False)
            item.setFlag(ITEM_MOVABLE, False)
        else:
            item.set_editable(self.edit_mode_enabled)
        self.scene.addItem(item)
        self.items[obs.label] = item

    def remove_selected(self):
        self.delete_selected()

    def delete_selected(self):
        if not self.edit_mode_enabled:
            self._switch_to_edit_mode_for_interaction()
        labels = [i.data(USER_ROLE) for i in self.list_widget.selectedItems()]
        if not labels:
            return
        self.obstacles = [o for o in self.obstacles if o.label not in labels]
        for label in labels:
            item = self.items.pop(label, None)
            if item:
                self.scene.removeItem(item)
        for i in reversed(range(self.list_widget.count())):
            if self.list_widget.item(i).data(USER_ROLE) in labels:
                self.list_widget.takeItem(i)
        self.refresh_labels()

    def copy_selected(self):
        if not self.edit_mode_enabled:
            self._switch_to_edit_mode_for_interaction()
        selected = [i for i in self.scene.selectedItems() if isinstance(i, ObstacleItem)]
        if not selected:
            return
        self.copied = [
            Obstacle(
                label="",
                object_id=item.obstacle.object_id,
                x=item.obstacle.x,
                y=item.obstacle.y,
                yaw=item.obstacle.yaw,
                extent_x=item.obstacle.extent_x,
                extent_y=item.obstacle.extent_y,
            )
            for item in selected
        ]

    def paste_copied(self):
        if not self.edit_mode_enabled:
            self._switch_to_edit_mode_for_interaction()
        if not self.copied:
            return
        offset = 0.5
        self.list_widget.clearSelection()
        for obs in self.copied:
            new_obs = self.add_obstacle_with_params(
                obs.object_id,
                obs.x + offset,
                obs.y + offset,
                obs.yaw,
            )
            if new_obs:
                item = self.items.get(new_obs.label)
                if item:
                    item.setSelected(True)
        self.sync_selection_from_list()

    def refresh_labels(self):
        for idx, obs in enumerate(self.obstacles):
            obs.label = index_to_label(idx)
        self.list_widget.clear()
        for obs in self.obstacles:
            item = QListWidgetItem(f"{obs.label}: {obs.object_id}")
            item.setData(USER_ROLE, obs.label)
            self.list_widget.addItem(item)
        # rebuild items map with updated labels
        new_items = {}
        for item in list(self.items.values()):
            if isinstance(item, ObstacleItem):
                new_items[item.obstacle.label] = item
        self.items = new_items
        self.sync_selection_from_list()

    def on_list_selection_changed(self):
        if (
            not self.edit_mode_enabled
            and not self._suppress_auto_switch
            and bool(self.list_widget.selectedItems())
        ):
            self._switch_to_edit_mode_for_interaction()
        if self.syncing_selection:
            return
        self.sync_selection_from_list()
        obs = self.current_obstacle()
        if obs:
            self.update_property_fields(obs)
            self.extent_label.setText(f"Extent: {obs.extent_x:.2f} x {obs.extent_y:.2f} (m, half)")
            item = self.items.get(obs.label)
            if item:
                self.view.centerOn(item)
        else:
            self.clear_property_fields()

    def on_scene_selection(self):
        if not self.edit_mode_enabled and not self._suppress_auto_switch:
            selected_obstacles = [i for i in self.scene.selectedItems() if isinstance(i, ObstacleItem)]
            if selected_obstacles:
                self._switch_to_edit_mode_for_interaction()
        if self.syncing_selection:
            return
        selected = self.scene.selectedItems()
        self.syncing_selection = True
        self.list_widget.clearSelection()
        for item in selected:
            if isinstance(item, ObstacleItem):
                label = item.obstacle.label
                for i in range(self.list_widget.count()):
                    li = self.list_widget.item(i)
                    if li.data(USER_ROLE) == label:
                        li.setSelected(True)
                        break
        self.syncing_selection = False
        obs = self.current_obstacle()
        if obs:
            self.update_property_fields(obs)
            self.extent_label.setText(f"Extent: {obs.extent_x:.2f} x {obs.extent_y:.2f} (m, half)")
        else:
            self.clear_property_fields()

    def update_property_fields(self, obs: Obstacle):
        self.x_spin.blockSignals(True)
        self.y_spin.blockSignals(True)
        self.yaw_spin.blockSignals(True)
        self.x_spin.setValue(obs.x)
        self.y_spin.setValue(obs.y)
        self.yaw_spin.setValue(normalize_yaw(obs.yaw))
        self.x_spin.blockSignals(False)
        self.y_spin.blockSignals(False)
        self.yaw_spin.blockSignals(False)
        self.x_spin.setEnabled(True)
        self.y_spin.setEnabled(True)
        self.yaw_spin.setEnabled(True)

    def clear_property_fields(self):
        self.x_spin.blockSignals(True)
        self.y_spin.blockSignals(True)
        self.yaw_spin.blockSignals(True)
        self.x_spin.setValue(0.0)
        self.y_spin.setValue(0.0)
        self.yaw_spin.setValue(0.0)
        self.x_spin.blockSignals(False)
        self.y_spin.blockSignals(False)
        self.yaw_spin.blockSignals(False)
        self.x_spin.setEnabled(False)
        self.y_spin.setEnabled(False)
        self.yaw_spin.setEnabled(False)
        self.extent_label.setText("Extent: -")

    def on_spin_changed(self):
        if not self.edit_mode_enabled:
            self._switch_to_edit_mode_for_interaction()
        obs = self.current_obstacle()
        if not obs:
            return
        obs.x = float(self.x_spin.value())
        obs.y = float(self.y_spin.value())
        obs.yaw = normalize_yaw(float(self.yaw_spin.value()))
        item = self.items.get(obs.label)
        if item:
            item.setPos(to_scene_xy(obs.x, obs.y, self.scale))
            item.setRotation(to_scene_yaw(obs.yaw))
        self.sync_selection_from_list()

    def export_objects(self):
        parts = []
        for label, value in self.get_objects_attr_map().items():
            parts.append(f'{label}="{value}"')
        content = "<objects " + " ".join(parts) + "/>"
        self.output.setPlainText(content)

    def copy_to_clipboard(self):
        self.export_objects()
        QApplication.clipboard().setText(self.output.toPlainText())

    def clear_all_objects(self):
        self.obstacles = []
        for item in list(self.items.values()):
            self.scene.removeItem(item)
        self.items = {}
        self.list_widget.clear()
        self.sync_selection_from_list()

    def load_from_xml_dialog(self):
        if not self.edit_mode_enabled:
            self._switch_to_edit_mode_for_interaction()
        path, _ = QFileDialog.getOpenFileName(
            self, "Select scenario XML", "", "XML (*.xml)"
        )
        if not path:
            return
        self.load_from_xml(path)

    def load_from_xml(self, path: str):
        if not self.edit_mode_enabled:
            self._switch_to_edit_mode_for_interaction()
        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except Exception as exc:
            QMessageBox.warning(self, "Load Failed", f"Could not parse XML: {exc}")
            return

        objects_node = root.find(".//scenario/objects")
        if objects_node is None or not objects_node.attrib:
            QMessageBox.warning(self, "Load Failed", "No <objects> found in XML.")
            return

        # Parse attributes like a="id=... x=... y=... yaw=..."
        def parse_kv(s: str) -> dict:
            parts = s.split()
            out = {}
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    out[k] = v
            return out

        entries = []
        for key in sorted(objects_node.attrib.keys()):
            raw = objects_node.attrib[key]
            kv = parse_kv(raw)
            object_id = kv.get("id")
            if not object_id:
                continue
            x = float(kv.get("x", 0.0))
            y = float(kv.get("y", 0.0))
            yaw = float(kv.get("yaw", 0.0))
            entries.append((object_id, x, y, yaw))

        if not entries:
            QMessageBox.warning(self, "Load Failed", "No valid objects in XML.")
            return

        self.clear_all_objects()
        for object_id, x, y, yaw in entries:
            self.add_obstacle_with_params(object_id, x, y, yaw, allow_disallowed=True)
        # select last loaded object
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(self.list_widget.count() - 1)
            self.sync_selection_from_list()

    def on_item_changed(self, item: ObstacleItem):
        if not self.edit_mode_enabled:
            return
        obs = item.obstacle
        current = self.current_obstacle()
        if current and current.label == obs.label:
            self.update_property_fields(obs)

    def sync_selection_from_list(self):
        self.syncing_selection = True
        self.scene.clearSelection()
        if not self.edit_mode_enabled:
            self.syncing_selection = False
            return
        for li in self.list_widget.selectedItems():
            label = li.data(USER_ROLE)
            item = self.items.get(label)
            if item:
                item.setSelected(True)
        self.syncing_selection = False

    def set_edit_mode(self, enabled: bool):
        self.edit_mode_enabled = bool(enabled)
        mode_name = "edit" if self.edit_mode_enabled else "base"
        self.base_view_button.setChecked(not self.edit_mode_enabled)
        self.custom_edit_button.setChecked(self.edit_mode_enabled)
        self.view.edit_mode_enabled = self.edit_mode_enabled
        self.view.setDragMode(DRAG_MODE if self.edit_mode_enabled else QGraphicsView.DragMode.NoDrag)
        if not self.edit_mode_enabled:
            self._suppress_auto_switch = True
            try:
                self.list_widget.clearSelection()
                self.scene.clearSelection()
                self.clear_property_fields()
            finally:
                self._suppress_auto_switch = False

        for item in self.items.values():
            if isinstance(item, ObstacleItem):
                if self.altered_construction_editor_mode:
                    item.setFlag(ITEM_SELECTABLE, False)
                    item.setFlag(ITEM_MOVABLE, False)
                else:
                    item.set_editable(self.edit_mode_enabled)
                item.set_visual_mode(mode_name)
        for overlay_item in self.overlay_items:
            overlay_item.set_visual_mode(mode_name)
        self._update_legend()
        self.scene.update()
        self._position_mode_switch()
        self._position_legend()

    def _switch_to_edit_mode_for_interaction(self):
        if self.edit_mode_enabled:
            return
        self.set_edit_mode(True)

    def _extract_object_id_from_list_item(self, item: QListWidgetItem) -> str:
        if item is None:
            return ""
        text = item.text()
        if ":" in text:
            return text.split(":", 1)[1].strip()
        return ""

    def _available_object_ids(self):
        if self.allowed_object_ids is not None:
            return sorted(self.allowed_object_ids)
        return sorted(self.static_extents.keys())

    def _get_extent_for_object_id(self, object_id: str):
        ext = self.static_extents.get(object_id)
        if ext:
            return float(ext[0]), float(ext[1])
        return float(DEFAULT_VEHICLE_EXTENTS[0]), float(DEFAULT_VEHICLE_EXTENTS[1])

    def available_object_ids(self):
        return self._available_object_ids()

    def _prompt_new_object_id(self, current_id: str):
        options = self._available_object_ids()
        if not options:
            return None
        if current_id in options:
            current_index = options.index(current_id)
        else:
            options = [current_id] + options
            current_index = 0
        selected, ok = QInputDialog.getItem(
            self,
            "Change Prop",
            "Select new prop id:",
            options,
            current_index,
            False,
        )
        if not ok:
            return None
        chosen = str(selected).strip()
        return chosen if chosen else None

    def _apply_object_id_change(self, obs: Obstacle, new_object_id: str):
        extent_x, extent_y = self._get_extent_for_object_id(new_object_id)
        obs.object_id = new_object_id
        item = self.items.get(obs.label)
        if item:
            item.prepareGeometryChange()
        obs.extent_x = extent_x
        obs.extent_y = extent_y
        if item:
            item.update()
        for i in range(self.list_widget.count()):
            list_item = self.list_widget.item(i)
            if list_item.data(USER_ROLE) == obs.label:
                list_item.setText(f"{obs.label}: {obs.object_id}")
                break
        if self.current_obstacle() and self.current_obstacle().label == obs.label:
            self.update_property_fields(obs)
            self.extent_label.setText(f"Extent: {obs.extent_x:.2f} x {obs.extent_y:.2f} (m, half)")

    def _change_obstacle_prop(self, obs: Obstacle):
        if obs is None:
            return
        if not self.edit_mode_enabled:
            self._switch_to_edit_mode_for_interaction()
        new_object_id = self._prompt_new_object_id(obs.object_id)
        if not new_object_id or new_object_id == obs.object_id:
            return
        self._apply_object_id_change(obs, new_object_id)

    def set_obstacle_object_id(self, obs: Obstacle, new_object_id: str):
        if obs is None or not new_object_id:
            return
        if not self.edit_mode_enabled:
            self._switch_to_edit_mode_for_interaction()
        if new_object_id == obs.object_id:
            return
        self._apply_object_id_change(obs, new_object_id)

    def _on_list_item_double_clicked(self, item: QListWidgetItem):
        label = item.data(USER_ROLE)
        target = None
        for obs in self.obstacles:
            if obs.label == label:
                target = obs
                break
        self._change_obstacle_prop(target)

    def _on_obstacle_item_double_clicked(self, obstacle_item: ObstacleItem):
        self._change_obstacle_prop(obstacle_item.obstacle)

    def _find_preview_image_path(self, object_id: str):
        if not object_id:
            return None
        normalized = object_id.strip().lower().replace(" ", "_").replace(".", "_")
        if not normalized:
            return None
        candidates = [
            self.prop_preview_dir / f"{normalized}.webp",
            self.prop_preview_dir / f"{normalized}.png",
            self.prop_preview_dir / f"{normalized}.jpg",
            self.prop_preview_dir / f"{normalized}.jpeg",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        matches = list(self.prop_preview_dir.glob(f"{normalized}.*"))
        if matches:
            return matches[0]
        return None

    def _show_prop_preview(self, object_id: str, global_pos):
        image_path = self._find_preview_image_path(object_id)
        if image_path is None:
            self._hide_prop_preview()
            return

        cache_key = str(image_path)
        pixmap = self.prop_preview_cache.get(cache_key)
        if pixmap is None:
            loaded = QPixmap(str(image_path))
            if loaded.isNull():
                self._hide_prop_preview()
                return
            target_w, target_h = self.prop_preview_size
            # Fill the preview card and crop center for a tighter zoomed-in look.
            fitted = loaded.scaled(
                target_w,
                target_h,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = max(0, (fitted.width() - target_w) // 2)
            y = max(0, (fitted.height() - target_h) // 2)
            pixmap = fitted.copy(x, y, target_w, target_h)
            self.prop_preview_cache[cache_key] = pixmap

        self.prop_preview_popup.setPixmap(pixmap)
        self.prop_preview_popup.adjustSize()
        self.prop_preview_popup.move(global_pos.x() + 18, global_pos.y() + 20)
        self.prop_preview_popup.show()

    def _hide_prop_preview(self):
        self.prop_preview_popup.hide()

    def _position_mode_switch(self):
        if not hasattr(self, "mode_switch_box") or self.mode_switch_box is None:
            return
        viewport = self.view.viewport()
        box_width = self.mode_switch_box.sizeHint().width()
        x = max(8, (viewport.width() - box_width) // 2)
        self.mode_switch_box.move(x, 10)
        self.mode_switch_box.raise_()

    def _position_legend(self):
        if not hasattr(self, "legend_box") or self.legend_box is None:
            return
        viewport = self.view.viewport()
        legend_size = self.legend_box.sizeHint()
        x = max(8, viewport.width() - legend_size.width() - 10)
        y = max(8, viewport.height() - legend_size.height() - 10)
        self.legend_box.move(x, y)
        self.legend_box.raise_()

    def _update_legend(self):
        if self.edit_mode_enabled:
            self.legend_custom_text.setText("Custom objects")
            self.legend_custom_swatch.setStyleSheet(
                "background: rgba(80,160,240,180); border: 2px solid rgba(40,90,160,220);"
            )
            self.legend_overlay_swatch.setStyleSheet(
                "background: transparent; border: 2px dashed rgba(140,140,140,200);"
            )
        else:
            self.legend_custom_text.setText("Custom objects")
            self.legend_custom_swatch.setStyleSheet(
                "background: transparent; border: 2px dashed rgba(140,140,140,200);"
            )
            self.legend_overlay_swatch.setStyleSheet(
                "background: rgba(80,160,240,180); border: 2px solid rgba(40,90,160,220);"
            )

    def eventFilter(self, watched, event):
        if watched is self.view.viewport() and event.type() == QEvent.Type.Resize:
            self._position_mode_switch()
            self._position_legend()
            return super().eventFilter(watched, event)

        if watched in self.preview_combo_viewports:
            combo_box = self.preview_combo_viewports[watched]
            if event.type() == QEvent.Type.MouseMove:
                index = combo_box.view().indexAt(event.pos())
                if index.isValid():
                    object_id = str(index.data())
                    self._show_prop_preview(object_id, watched.mapToGlobal(event.pos()))
                else:
                    self._hide_prop_preview()
            elif event.type() in (QEvent.Type.Leave, QEvent.Type.Hide):
                self._hide_prop_preview()
            return super().eventFilter(watched, event)

        if watched in self.preview_combo_widgets:
            combo_box = self.preview_combo_widgets[watched]
            if event.type() in (QEvent.Type.Enter, QEvent.Type.MouseMove):
                object_id = combo_box.currentText().strip()
                if object_id:
                    global_pos = QCursor.pos()
                    self._show_prop_preview(object_id, global_pos)
                else:
                    self._hide_prop_preview()
            elif event.type() in (QEvent.Type.Leave, QEvent.Type.Hide):
                self._hide_prop_preview()
            return super().eventFilter(watched, event)

        if watched in self.preview_combo_lineedits:
            combo_box = self.preview_combo_lineedits[watched]
            if event.type() in (QEvent.Type.Enter, QEvent.Type.MouseMove):
                object_id = combo_box.currentText().strip()
                if object_id:
                    global_pos = QCursor.pos()
                    self._show_prop_preview(object_id, global_pos)
                else:
                    self._hide_prop_preview()
            elif event.type() in (QEvent.Type.Leave, QEvent.Type.Hide):
                self._hide_prop_preview()
            return super().eventFilter(watched, event)

        if watched is self.list_widget.viewport():
            if event.type() == QEvent.Type.MouseMove:
                item = self.list_widget.itemAt(event.pos())
                object_id = self._extract_object_id_from_list_item(item)
                if object_id:
                    self._show_prop_preview(object_id, watched.mapToGlobal(event.pos()))
                else:
                    self._hide_prop_preview()
            elif event.type() in (QEvent.Type.Leave, QEvent.Type.Hide):
                self._hide_prop_preview()
        return super().eventFilter(watched, event)


def main():
    app = QApplication(sys.argv)
    window = Designer()
    window.resize(1300, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
