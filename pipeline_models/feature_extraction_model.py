import sys
import os
from pipeline_conf.conf import DEVICE, PATHS

class FeatureExtractionModel_PointMAE():
    def __init__(self):
        sys.path.insert(0, PATHS.pointmae)
        from external.PointMAE.utils import registry
        from external.PointMAE.models.Point_MAE import PointTransformer
        MODELS = registry.Registry('models')
        MODELS.register_module("PointTransformer", False, PointTransformer)
        from external.PointMAE.utils.config import cfg_from_yaml_file
        from external.PointMAE.models import build_model_from_cfg
        from external.PointMAE.tools.builder import load_model
        from external.PointMAE.segmentation.logger import get_logger
        import torch

        # Load config
        config_path = PATHS.pointmae_config_file_path
        config = cfg_from_yaml_file(config_path)
        
        # Build PointTransformer (PointTransformer(nn.Module) is registerd in Registry)
        self.model = MODELS.build(config.model)


        # Load pretrained checkpoint
        load_model(self.model, PATHS.pointmae_checkpoint, logger = get_logger("main"))
        self.model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, pts):
        return self.model.forward(pts)