import sys
import os

# WICHTIG: OpenCVs eigenes Qt-Plugin-Verzeichnis darf nicht verwendet werden
cv2_plugin_path = os.path.join(os.environ["CONDA_PREFIX"], "lib", "python3.10", "site-packages", "cv2", "qt", "plugins")
if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ and os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] == cv2_plugin_path:
    del os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]

# Danach PyQt5 importieren
from PyQt5.QtWidgets import QApplication

# Dann erst cv2
import cv2

from pipeline_gui.models.data_model import DataModel
from pipeline_gui.views.main_view import MainView
from pipeline_gui.controllers.view_controller import ViewController

def main(instance_detection_mode):
    # ======= Load MVC components =======
    app = QApplication(sys.argv)
    model = DataModel(instance_detection_mode)
    view = MainView()
    controller = ViewController(model, view)
    
    # ======= Start GUI =======
    view.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    if sys.argv[1] == "gt" or sys.argv[1] == "unscene3d":
        main(sys.argv[1])
    else:
        raise ValueError("Specify instance detection mode!")

