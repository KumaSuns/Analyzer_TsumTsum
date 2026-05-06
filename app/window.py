import platform

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QButtonGroup,
    QLabel,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    def __init__(self, os_name: str = "", compute_status: str = "") -> None:
        super().__init__()
        self.setWindowTitle("ツムツム解析アプリ")
        self.resize(960, 640)
        self.os_name = os_name or platform.system()
        self.compute_status = compute_status or "CPU:使用 / GPU:未使用"
        self._setup_menu()
        self._setup_status_bar()
        self._setup_central_widget()

    def _setup_menu(self) -> None:
        file_menu = self.menuBar().addMenu("ファイル")

        exit_action = QAction("終了", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = self.menuBar().addMenu("ヘルプ")
        about_action = QAction("このアプリについて", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_status_bar(self) -> None:
        self.statusBar().hide()

    def _setup_central_widget(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(0)

        header = QFrame(central)
        header.setObjectName("headerSection")
        header.setStyleSheet("QFrame#headerSection { border: none; background: #FFF; }")
        header.setFixedHeight(50)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        button_height = 36
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        self.feature_buttons = []
        button_labels = [
            "テスト・処理",
            "動画ツール",
            "button3",
            "button4",
            "button5",
            "button6",
            "button7",
            "button8",
        ]
        for index in range(8):
            button = QPushButton(button_labels[index], header)
            button.setFixedSize(100, button_height)
            button.setCheckable(True)
            if index == 0:
                button.setChecked(True)
            header_layout.addWidget(button)
            self.button_group.addButton(button, index + 1)
            self.feature_buttons.append(button)
        self.button_group.idClicked.connect(self._on_feature_changed)
        close_button = QPushButton("✖", header)
        close_button.setFixedSize(button_height, button_height)
        close_button.clicked.connect(self.close)
        header_layout.addWidget(close_button)

        main = QFrame(central)
        main.setObjectName("mainSection")
        main.setStyleSheet("QFrame#mainSection { border: none; background: #FFF; }")
        main_layout = QHBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        left = QFrame(main)
        left.setObjectName("leftSection")
        left.setStyleSheet("QFrame#leftSection { border: 1px solid black; background: #FFF; }")
        self.left_layout = QVBoxLayout(left)
        self.left_layout.setContentsMargins(0, 0, 0, 0)

        center = QFrame(main)
        center.setObjectName("centerSection")
        center.setStyleSheet("QFrame#centerSection { border: 1px solid black; background: #FFF; }")
        center.setFixedWidth(480)
        self.center_layout = QVBoxLayout(center)
        self.center_layout.setContentsMargins(0, 0, 0, 0)

        right = QFrame(main)
        right.setObjectName("rightSection")
        right.setStyleSheet("QFrame#rightSection { border: 1px solid black; background: #FFF; }")
        self.right_layout = QVBoxLayout(right)
        self.right_layout.setContentsMargins(0, 0, 0, 0)

        main_layout.addWidget(left, 1)
        main_layout.addWidget(center, 0)
        main_layout.addWidget(right, 1)

        footer = QFrame(central)
        footer.setObjectName("footerSection")
        footer.setStyleSheet("QFrame#footerSection { border: none; background: #FFF; }")
        footer.setFixedHeight(50)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        version_label = QLabel(
            f"Version_00_00_01  {self.os_name}  {self.compute_status}",
            footer,
        )
        version_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        footer_layout.addWidget(version_label, 1)

        layout.addWidget(header)
        layout.addWidget(main, 1)
        layout.addWidget(footer)
        self.setCentralWidget(central)
        self._render_feature_ui(1)

    def _on_feature_changed(self, feature_id: int) -> None:
        self._render_feature_ui(feature_id)

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _render_feature_ui(self, feature_id: int) -> None:
        self._clear_layout(self.left_layout)
        self._clear_layout(self.center_layout)
        self._clear_layout(self.right_layout)

        self.left_layout.addWidget(QLabel(f"left - button{feature_id}"))
        self.center_layout.addWidget(QLabel(f"center - button{feature_id}"))
        self.right_layout.addWidget(QLabel(f"right - button{feature_id}"))

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "Analyzer_TsumTsum について",
            "Analyzer_TsumTsum\nPySide6製デスクトップアプリの土台です。",
        )
