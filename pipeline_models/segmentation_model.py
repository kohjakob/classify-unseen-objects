
import torch
import os
import sys
import hydra

from external.UnScene3D.utils.utils import load_checkpoint_with_missing_or_exsessive_keys
from pipeline_conf.conf import DEVICE, PATHS

sys.path.append(PATHS.unscene)

class InstanceSegmentationModel_UnScene3D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load config
        with hydra.initialize(config_path=PATHS.unscene3d_config_dir, version_base="1.1"):
            cfg = hydra.compose(config_name=PATHS.unscene3d_config_file)
        cfg.general.checkpoint = PATHS.unscene3d_checkpoint
        cfg.general.train_on_segments = False
        cfg.general.eval_on_segments = False
        
        # Build InstanceSegmentation
        self.model = hydra.utils.instantiate(cfg.model)

        # UnScene3D uses hydra and needs relative paths to load yaml
        # Therfor we change dir
        os.chdir(PATHS.base_dir)
        
        # Load pretrained checkpoint
        _, self.model = load_checkpoint_with_missing_or_exsessive_keys(cfg, self.model)
        
        self.model = self.model.to(DEVICE)
        self.model.eval()  
        os.chdir(PATHS.base_dir)

    def forward(self, x, raw_coordinates=None):
        return self.model(x, raw_coordinates=raw_coordinates)