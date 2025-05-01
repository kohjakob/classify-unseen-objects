import sys
from PyQt5.QtWidgets import QApplication

from pipeline_gui.models.data_model import DataModel
from pipeline_gui.views.main_view import MainView
from pipeline_gui.controllers.view_controller import ScanViewController

def main():
    app = QApplication(sys.argv)
    
    # Create MVC components
    model = DataModel()
    view = MainView()
    controller = ScanViewController(model, view)
    
    view.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()