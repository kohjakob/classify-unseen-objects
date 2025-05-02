from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from pipeline_gui.views.point_cloud_view import PointCloudView
from pipeline_gui.views.umap_view import UMAPView
from pipeline_gui.views.info_panel_view import InfoPanelView

class MainView(QMainWindow):
    def __init__(self):
        super(MainView, self).__init__()
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle('Point Cloud Viewer with UMAP')
        self.setGeometry(100, 100, 1200, 600)
        
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Left and right layouts
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()
        
        # Create point cloud views
        self.instance_view = PointCloudView()
        self.scene_view = PointCloudView()
        
        # Create UMAP view
        self.umap_view = UMAPView(self.central_widget)
        
        # Create info panel
        self.info_panel = InfoPanelView()
        
        # Create labels
        self.instance_label = QLabel("Isolated Instance")
        self.instance_label.setFont(QFont('Arial', 12, QFont.Bold))
        self.instance_label.setAlignment(Qt.AlignCenter)
        
        self.scene_label = QLabel("Scan")
        self.scene_label.setFont(QFont('Arial', 12, QFont.Bold))
        self.scene_label.setAlignment(Qt.AlignCenter)
        
        self.umap_label = QLabel("UMAP Projection of Feature Space")
        self.umap_label.setFont(QFont('Arial', 12, QFont.Bold))
        self.umap_label.setAlignment(Qt.AlignCenter)
        
        # Create navigation buttons
        self.next_button = QPushButton('Next (Furthest in Cluster)')
        self.next_cluster_button = QPushButton('Next Cluster')
        self.prev_scan_button = QPushButton('Prev Scan')
        self.next_scan_button = QPushButton('Next Scan')
        
        # Button layouts
        self.instance_buttons_layout = QHBoxLayout()
        self.instance_buttons_layout.addWidget(self.next_button)
        self.instance_buttons_layout.addWidget(self.next_cluster_button)
        
        self.scan_buttons_layout = QHBoxLayout()
        self.scan_buttons_layout.addWidget(self.prev_scan_button)
        self.scan_buttons_layout.addWidget(self.next_scan_button)
        
        # Arrange left layout
        self.left_layout.addWidget(self.instance_label)
        self.left_layout.addWidget(self.instance_view.widget)
        self.left_layout.addWidget(self.scene_label)
        self.left_layout.addWidget(self.scene_view.widget)
        self.left_layout.addWidget(self.info_panel.frame)
        self.left_layout.addLayout(self.scan_buttons_layout)
        
        # Arrange right layout
        self.right_layout.addWidget(self.umap_label)
        self.right_layout.addWidget(self.umap_view)
        self.right_layout.addLayout(self.instance_buttons_layout)
        
        # Add layouts to main layout
        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.right_layout)
    
    def connect_button_handlers(self, next_handler, next_cluster_handler, prev_scan_handler, next_scan_handler):
        self.next_button.clicked.connect(next_handler)
        self.next_cluster_button.clicked.connect(next_cluster_handler)
        self.prev_scan_button.clicked.connect(prev_scan_handler)
        self.next_scan_button.clicked.connect(next_scan_handler)