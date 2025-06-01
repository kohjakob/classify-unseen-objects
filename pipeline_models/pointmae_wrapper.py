import sys
import os
import importlib
import numpy as np
import torch

from pipeline_conf.conf import CONFIG, DEVICE

sys.path.insert(0, CONFIG.pointmae)
from external.PointMAE.models.Point_MAE import PointTransformer
from external.PointMAE.tools.builder import load_model
from external.PointMAE.segmentation.logger import get_logger
import external.PointMAE.utils.registry as registry
from external.PointMAE.utils.config import cfg_from_yaml_file

class PointMAE_Wrapper():
    def __init__(self):

        # Continue with initialization
        MODELS = registry.Registry('models')
        MODELS.register_module("PointTransformer", False, PointTransformer)
        
        # Load config
        config_path = CONFIG.pointmae_config_file_path
        config = cfg_from_yaml_file(config_path)
        
        # Build model
        self.model = MODELS.build(config.model)
        
        # Load pretrained checkpoint  
        load_model(self.model, CONFIG.pointmae_checkpoint, logger=get_logger("main"))
        self.model.eval()
        self.model.to(DEVICE)

    def extract_features(self, points):
        # Add batch dimension => shape (1, N, 3)
        points_downsampled_expanded = np.expand_dims(points, axis=0).astype(np.float32)
        points_tensor = torch.from_numpy(points_downsampled_expanded).to(DEVICE)
        points_tensor = points_tensor.float()

        with torch.no_grad():
            output = self.model.forward(points_tensor)
        features = output.cpu().numpy().squeeze(0)

        return features
