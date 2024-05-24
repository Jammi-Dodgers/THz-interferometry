import sys, os, copy
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QAction, QFileDialog, QPushButton, QLineEdit
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import FTIR_functions as FTIR
plt.rcParams.update({"image.cmap": 'Greys_r', 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, "font.size": 16, 'image.origin': 'lower'}) # set defult values of graphs

class UI(QMainWindow):
    def __init__(self): # What happens when an instance of this class is created?
        super(UI,self).__init__() # Inherit methods (class functions) and attributes (class variables) from QMainWindow. 
        QMainWindow.__init__(self) # Initialise the UI as you would initialise QMainWindow.

        self.parent_directory = os.path.dirname(__file__)
        loadUi(os.path.join(self.parent_directory, "FTIR_qt_interface.ui"),self) # Using a relative path (ie. loadUi("FTIR_qt_interface.ui",self)) doesn't work because the current working directory has changed. IDK why it changed.

        ### initialise other variables
        self.list_2d_bg = []
        self.list_2d_bgpaths = []

        ### define figures
        self.figure_2d_image = plt.figure()
        self.figure_2d_bg = plt.figure()
        self.figure_2d_processed = plt.figure()
        self.figure_2d_ft = plt.figure()
        self.figure_2d_fringes = plt.figure()

        ### define canvases
        self.canvas_2d_image = FigureCanvas(self.figure_2d_image)
        self.canvas_2d_bg = FigureCanvas(self.figure_2d_bg)
        self.canvas_2d_processed = FigureCanvas(self.figure_2d_processed)
        self.canvas_2d_ft = FigureCanvas(self.figure_2d_ft)
        self.canvas_2d_fringes = FigureCanvas(self.figure_2d_fringes)

        ### define box layouts inside the frames
        self.box_2d_image = QVBoxLayout(self.frame_2d_image)
        self.box_2d_bg = QVBoxLayout(self.frame_2d_bg)
        self.box_2d_processed = QVBoxLayout(self.frame_2d_processed)
        self.box_2d_ft = QVBoxLayout(self.frame_2d_ft)
        self.box_2d_fringes = QVBoxLayout(self.frame_2d_fringes)

        ### put the canvases inside the box layouts
        self.box_2d_image.addWidget(self.canvas_2d_image)
        self.box_2d_bg.addWidget(self.canvas_2d_bg)
        self.box_2d_processed.addWidget(self.canvas_2d_processed)
        self.box_2d_ft.addWidget(self.canvas_2d_ft)
        self.box_2d_fringes.addWidget(self.canvas_2d_fringes)

        ### connect buttons to methods. (so that they actually do something!)
        self.button_2d_loadimage.clicked.connect(self.method_2d_loadimage)
        self.button_2d_loadbg.clicked.connect(self.method_2d_loadbg)
        self.button_2d_bgsub.clicked.connect(self.method_2d_bgsub)
        self.button_2d_reset.clicked.connect(self.method_2d_reset)
        self.button_2d_badfilter.clicked.connect(self.method_2d_badfilter)
        self.vslider_2d_processed.sliderReleased.connect(self.method_2d_collapse)
        self.hslider_2d_processed.sliderReleased.connect(self.method_2d_collapse)
        self.combo_2d_average.currentTextChanged.connect(self.method_2d_collapse)
        self.button_2d_saveimage.clicked.connect(self.method_2d_saveimage)
        self.button_2d_savefringes.clicked.connect(self.method_2d_savefringes)

    ############################ BUTTONS ##############################

    def method_2d_loadimage(self): # plot_2d_image is a method, not a function because it is associated with the class instance. (It takes self as an argument.)
        str_image_path, _ = QFileDialog.getOpenFileName(self, "Load Image", self.parent_directory, "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.csv *.fts)") # allow common image files as well as .csv and .fts. These cover all of the cameras that I've tested so far.

        if str_image_path: # if a file is selected
            self.lineedit_2d_loadimage.setText(str_image_path)
            image_path = os.path.normpath(str_image_path)
            self.array_2d_image = FTIR.open_image(image_path)
            self.plot_2d_image() # now that the data is defined. we can plot it. :)

            self.method_2d_reset() # loading a new image should act as an implict reset.

    def method_2d_loadbg(self):
        str_image_path, _ = QFileDialog.getOpenFileName(self, "Load Background Image", self.parent_directory, "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.csv *.fts)") # allow common image files as well as .csv and .fts. These cover all of the cameras that I've tested so far.

        if str_image_path: # if a file is selected
            self.list_2d_bgpaths.append(str_image_path)
            self.ptextedit_2d_loadbg.setPlainText("\n".join(self.list_2d_bgpaths))

            image_path = os.path.normpath(str_image_path)
            background_array = FTIR.open_image(image_path)
            self.list_2d_bg.append(background_array)
            self.plot_2d_bg() # now that the data is defined. we can plot it. :)

    def method_2d_reset(self):
        if not FTIR.is_defined('array_2d_image', self):
            print("Failed to reset. No image to reset!")
            return
       
        self.array_2d_processed = copy.copy(self.array_2d_image)
        self.plot_2d_processed() # plot the data

    def method_2d_bgsub(self):
        equation = self.lineedit_2d_bgsub.text() 
        symbols = {char for char in equation if char.isupper()} # SET COMPREHENTION
        bg_indexes = [FTIR.ALPHABET.index(symbol) -1 for symbol in symbols]

        equation = equation.replace("A", "self.array_2d_image")
        for symbol, bg_index in zip(symbols, bg_indexes):
            equation = equation.replace(symbol, "self.list_2d_bg[{0:d}]".format(bg_index))

        try:
            self.array_2d_processed = eval(equation)
        except NameError:
            print("Failed to background subtract. Equation is invalid.")
            return
        except IndexError:
            print("Failed to background subtract. At least one image is missing.")
            return
        
        if np.any(np.isnan(self.array_2d_processed)):
            print("WARNING! Nans detected in processed image.")

        self.plot_2d_processed() # plot the data

    def method_2d_badfilter(self):
        if not FTIR.is_defined('array_2d_processed', self):
            print("Failed to filter. No image to filter!")
            return
        try:
            bad_pixel_percentage = float(self.lineedit_2d_badfilter.text())
        except ValueError:
            print("Failed to filter. '{0:}' could not be converted to a float.".format(self.lineedit_2d_badfilter.text()))
            return
        
        self.array_2d_processed = FTIR.dead_pixel_filter(self.array_2d_processed, bad_pixel_percentage)
        self.plot_2d_processed() # plot the data

    def method_2d_collapse(self):
        if not FTIR.is_defined('array_2d_processed', self):
            print("Failed to update processed image graph. No data to plot.")
            return
        
        self.plot_2d_processed() # plot the data

        if self.combo_2d_average.currentText() == "Slice row":
            row_number = int(self.array_2d_processed.shape[0] *self.float_2d_vslider)
            self.array_2d_fringes = self.array_2d_processed[row_number,:]
        elif self.combo_2d_average.currentText() == "Slice column":
            column_number = int(self.array_2d_processed.shape[1] *self.float_2d_hslider)
            self.array_2d_fringes = self.array_2d_processed[:,column_number]
        elif self.combo_2d_average.currentText() == "Average rows":
            self.array_2d_fringes = np.mean(self.array_2d_processed, axis= 0)
        elif self.combo_2d_average.currentText() == "Average columns":
            self.array_2d_fringes = np.mean(self.array_2d_processed, axis= 1)
        else:
            print("Failed to collapse to 1d. The method '{0:}' was not understood".format(self.combo_2d_average.currentText()))
            return
        
        self.plot_2d_fringes()

    def method_2d_saveimage(self):
        if not FTIR.is_defined('array_2d_processed', self):
            print("Failed to save processed image. No data to save.")
            return

        str_image_path, _ = QFileDialog.getSaveFileName(self,"Save File",self.parent_directory,"Image Files (*.png *.tif *.tiff)")

        if str_image_path: # if a file is selected
            self.lineedit_2d_saveimage.setText(str_image_path)
            image_path = os.path.normpath(str_image_path)
            FTIR.save_image(image_path, self.array_2d_processed)

    def method_2d_savefringes(self):
        if not FTIR.is_defined('array_2d_fringes', self):
            print("Failed to save processed fringes. No data to save.")
            return

        str_image_path, _ = QFileDialog.getSaveFileName(self,"Save File",self.parent_directory,"Data Files (*.txt *.csv *.npy)")

        if str_image_path: # if a file is selected
            self.lineedit_2d_savefringes.setText(str_image_path)
            image_path = os.path.normpath(str_image_path)
            FTIR.save_fringes(image_path, self.array_2d_fringes)


    ############################## PLOTS ###############################

    def plot_2d_image(self):
        self.figure_2d_image.clear() # erase previous plot
        axs = self.figure_2d_image.subplots() # add axes
        axs.imshow(self.array_2d_image) # plot image. # ENSURE THAT THE DATA IS DEFINED BEFORE ATTEMPTING TO PLOT IT
        axs.set(title= "A")
        self.canvas_2d_image.draw() # instead of fig.show(), we need to update the figure on the canvas.

    def plot_2d_bg(self):
        self.figure_2d_bg.clear() # erase previous plot
        axs = self.figure_2d_bg.subplots(1, len(self.list_2d_bg)) # add axes
        axs = np.atleast_1d(axs) # make into an array if not an array already
        for ax, background, label in zip(axs, self.list_2d_bg, FTIR.ALPHABET[1:]):
            ax.imshow(background) # plot image. # ENSURE THAT THE DATA IS DEFINED BEFORE ATTEMPTING TO PLOT IT
            ax.set(title= label)
        self.canvas_2d_bg.draw() # instead of fig.show(), we need to update the figure on the canvas.

    def plot_2d_processed(self):
        self.figure_2d_processed.clear() # erase previous plot
        axs = self.figure_2d_processed.subplots() # add axes
        im = axs.imshow(self.array_2d_processed) # plot image. # ENSURE THAT THE DATA IS DEFINED BEFORE ATTEMPTING TO PLOT IT

        if self.combo_2d_average.currentText() == "Slice row":
            self.float_2d_vslider = self.vslider_2d_processed.value() /(self.vslider_2d_processed.maximum() +1)
            axs.hlines(self.float_2d_vslider, 0, 1, colors= "tab:blue", transform= axs.transAxes)
        elif self.combo_2d_average.currentText() == "Slice column":
            self.float_2d_hslider = self.hslider_2d_processed.value() /(self.hslider_2d_processed.maximum() +1)
            axs.vlines(self.float_2d_hslider, 0, 1, colors= "tab:blue", transform= axs.transAxes)

        self.canvas_2d_processed.draw() # instead of fig.show(), we need to update the figure on the canvas.

        self.method_2d_ft() # calculate its ft

    def plot_2d_ft(self):
        self.figure_2d_ft.clear() # erase previous plot
        axs = self.figure_2d_ft.subplots() # add axes
        im = axs.imshow(np.abs(np.fft.fftshift(self.array_2d_ft)), extent= (-1,1,-1,1), norm= "log", cmap= "magma_r") # plot image. # ENSURE THAT THE DATA IS DEFINED BEFORE ATTEMPTING TO PLOT IT
        self.canvas_2d_ft.draw() # Will crash if the ft is all nans. This can happen if there are any nans in the processed data.

    def plot_2d_fringes(self):
        self.figure_2d_fringes.clear()
        axs = self.figure_2d_fringes.subplots()
        axs.plot(self.array_2d_fringes)
        self.canvas_2d_fringes.draw()

    ############################### AUTOMATIC METHODS ###################################

    def method_2d_ft(self):
        if not FTIR.is_defined('array_2d_processed', self):
            print("Failed to Fourier transform. No image to transform!")
            return
        
        self.array_2d_ft = np.fft.fft(self.array_2d_processed)
        self.plot_2d_ft()


### Create instance of the class so that we can execute it.

app = QApplication(sys.argv) # tell Qt about the current working enviroment. (ie. python 3.11 for windows 10)
UIWindow = UI()
UIWindow.show()

app.exec_()