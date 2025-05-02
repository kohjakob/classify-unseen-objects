from PyQt5.QtWidgets import QVBoxLayout, QLabel, QFrame
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

class InfoPanelView:
    def __init__(self):
        self.frame = QFrame()
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setStyleSheet("background-color: #f0f0f0; border: 1px solid lightgray; border-radius: 5px;")
        
        self.layout = QVBoxLayout(self.frame)
        
        self.gt_label_widget = QLabel("Ground Truth Label: None")
        self.gt_label_widget.setFont(QFont('Arial', 12, QFont.Bold))
        self.gt_label_widget.setAlignment(Qt.AlignCenter)
        
        self.cluster_label_widget = QLabel("Cluster: None")
        self.cluster_label_widget.setFont(QFont('Arial', 12, QFont.Bold))
        self.cluster_label_widget.setAlignment(Qt.AlignCenter)
        
        self.scan_label_widget = QLabel("Scan: None")
        self.scan_label_widget.setFont(QFont('Arial', 12, QFont.Bold))
        self.scan_label_widget.setAlignment(Qt.AlignCenter)
        
        self.layout.addWidget(self.gt_label_widget)
        self.layout.addWidget(self.cluster_label_widget)
        self.layout.addWidget(self.scan_label_widget)
    
    def update_gt_label(self, gt_label):
        self.gt_label_widget.setText(f"Ground Truth Label: {gt_label}")
    
    def update_cluster_label(self, cluster_label, color_hex=None):
        self.cluster_label_widget.setText(f"Cluster {cluster_label}")
        if color_hex:
            self.cluster_label_widget.setStyleSheet(f"color: {color_hex}; font-weight: bold;")
    
    def update_scan_label(self, scan_name):
        self.scan_label_widget.setText(f"Scan: {scan_name}")