import sys
from FTIR_qt_interface import Ui_MainWindow as GUI
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)
UIWindow = GUI()
UIWindow.show()

app.exec_()
