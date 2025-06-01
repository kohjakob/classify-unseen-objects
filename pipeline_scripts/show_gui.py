import sys
import os
from PyQt5.QtWidgets import QApplication
import cv2
from pipeline_conf.conf import CONFIG

# WICHTIG: OpenCVs eigenes Qt-Plugin-Verzeichnis darf nicht verwendet werden
#cv2_plugin_path = os.path.join(os.environ["CONDA_PREFIX"], "lib", "python3.10", "site-packages", "cv2", "qt", "plugins")
#if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ and os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] == cv2_plugin_path:
#    del os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]

from pipeline_gui.models.data_model import DataModel
from pipeline_gui.views.main_view import MainView
from pipeline_gui.controllers.view_controller import ViewController

def main(instance_detection_mode):
    app = QApplication(sys.argv)
    model = DataModel(instance_detection_mode)
    view = MainView()
    controller = ViewController(model, view)
    
    view.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    # Check command line arguments
    # Usage: python3 show_gui.py <instance_detection_mode>
    if len(sys.argv) < 2 or sys.argv[1] not in CONFIG.instance_detection_mode_dict:
        
        print("Usage: python show_gui.py <instance_detection_mode>")
        print("Available instance detection modes:", list(CONFIG.instance_detection_mode_dict.keys()))
        sys.exit(1)

    instance_detection_mode = CONFIG.instance_detection_mode_dict[sys.argv[1]]["name"]
    main(instance_detection_mode)