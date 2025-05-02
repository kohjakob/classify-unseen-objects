import numpy as np
from typing import Any
from pipeline_gui.models.data_model import DataModel
from pipeline_gui.views.main_view import MainView

class ViewController:
    def __init__(self, model, view):
        self.model: DataModel = model
        self.view: MainView = view
        
        # Connect button handlers
        self.view.connect_button_handlers(
            self.handle_next_instance,
            self.handle_next_cluster,
            self.handle_prev_scan,
            self.handle_next_scan
        )
        
        # Connect UMAP click handler
        self.view.umap_view.set_point_click_handler(self.handle_umap_click)
        
        # Initialize with first scan if available
        if self.model.available_scannet_scenes:
            self.model.load_scannet_scene_data(self.model.available_scannet_scenes[0])
            self.update_view()
        
    def handle_next_instance(self):
        if self.model.move_to_next_instance():
            self.update_view()
    
    def handle_next_cluster(self):
        if self.model.move_to_next_cluster():
            self.update_view()
    
    def handle_prev_scan(self):
        if self.model.move_to_previous_scan():
            self.update_view()
    
    def handle_next_scan(self):
        if self.model.move_to_next_scan():
            self.update_view()
    
    def handle_umap_click(self, x, y):
        if self.model.umap_embedding is not None:
            # Find closest point in the UMAP embedding
            distances = np.linalg.norm(self.model.umap_embedding - np.array([x, y]), axis=1)
            closest_index = np.argmin(distances)
            
            if self.model.select_point(closest_index):
                self.update_view()
    
    def update_view(self):
        # Update point cloud views
        selected_instance_points, selected_instance_colors = self.model.get_current_point_data()
        
        if selected_instance_points is not None and selected_instance_colors is not None:
            # Update instance view
            self.view.instance_view.update_point_cloud(
                instance_points=selected_instance_points, 
                scene_points=None,
                instance_colors=selected_instance_colors,
                scene_colors=None,
                add_axes=True
            )
            
            # Update scene view
            self.view.scene_view.update_point_cloud(
                instance_points=selected_instance_points,
                scene_points=self.model.scene_points, 
                instance_colors=selected_instance_colors,
                scene_colors=None,
                add_axes=True
            )
            self.view.scene_view.set_background_color(0.1, 0.1, 0.1)
        
        # Update UMAP plot
        label_to_color, current_label = self.view.umap_view.plot(
            self.model.umap_embedding, 
            self.model.instance_labels, 
            self.model.current_scene_idx
        )
        
        # Update info panel
        if current_label is not None and current_label in label_to_color:
            cluster_color = label_to_color[current_label]
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(cluster_color[0] * 255), 
                int(cluster_color[1] * 255), 
                int(cluster_color[2] * 255)
            )
            self.view.info_panel.update_cluster_label(self.model.current_cluster, hex_color)
        
        self.view.info_panel.update_gt_label(self.model.get_current_gt_label())
        self.view.info_panel.update_scan_label(self.model.get_current_scan_name())