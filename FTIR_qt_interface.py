import sys, os
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QAction, QFileDialog, QPushButton, QLineEdit
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from PIL import Image
plt.rcParams.update({"image.cmap": 'Greys_r', 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, "font.size": 16}) # set defult values of graphs

class UI(QMainWindow):
    def __init__(self): # What happens when an instance of this class is created?
        super(UI,self).__init__() # Inherit methods (class functions) and attributes (class variables) from QMainWindow. 
        QMainWindow.__init__(self) # Initialise the UI as you would initialise QMainWindow.

        self.parent_directory = os.path.dirname(__file__)
        loadUi(os.path.join(self.parent_directory, "FTIR_qt_interface.ui"),self) # Using a relative path (ie. loadUi("FTIR_qt_interface.ui",self)) doesn't work because the current working directory has changed. IDK why it changed.

        # define figures
        self.figure_2d_image = plt.figure()

        # define canvases
        self.canvas_2d_image = FigureCanvas(self.figure_2d_image)

        # define box layouts inside the frames
        self.box_2d_image = QVBoxLayout(self.frame_2d_image)

        # put the canvases inside the box layouts
        self.box_2d_image.addWidget(self.canvas_2d_image)

        # connect buttons to methods. (so that they actually do something!)
        self.button_2d_loadimage.clicked.connect(self.method_2d_loadimage)

    def method_2d_loadimage(self):
        str_image_path, image_extention = QFileDialog.getOpenFileName(self, "Load Image", self.parent_directory, "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.csv *.fts)") # allow common image files as well as .csv and .fts. These cover all of the cameras that I've tested so far.

        if str_image_path: # if a file is selected
            self.lineedit_2d_loadimage.setText(str_image_path)
            image_path = os.path.normpath(str_image_path)
            self.array_2d_image = Image.open(image_path) # WILL NOT WORK FOR ALL IMAGES. I should make an image opening function in FTIR_functions
            self.method_2d_image() # now that the data is defined. we can plot it. :)

    def method_2d_image(self): # method_2d_image is a method, not a function because it is associated with the class instance. (It takes self as an argument.)
        self.figure_2d_image.clear() # erase previous plot
        axs = self.figure_2d_image.subplots() # add axes
        im = axs.imshow(self.array_2d_image) # plot image. # ENSURE THAT THE DATA IS DEFINED BEFORE ATTEMPTING TO PLOT IT
        self.canvas_2d_image.draw() # instead of fig.show(), we need to update the figure on the canvas.

app = QApplication(sys.argv) # tell Qt about the current working enviroment. (ie. python 3.11 for windows 10)
UIWindow = UI()
UIWindow.show()

app.exec_()

# %%%%%%%%%%
import os

string = "some\\file\\path.txt"

lastSlash = string.rfind(('/'))
data = os.path.join(string[0:lastSlash+1],string[lastSlash+1:])

data2 = os.path.normpath(string)

print(data)
print(data2)

# %%
