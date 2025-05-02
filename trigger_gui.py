import sys
from PyQt5.QtWidgets import QApplication

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

