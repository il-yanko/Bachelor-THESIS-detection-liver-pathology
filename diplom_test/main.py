#!/usr/bin/python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtGui

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import matplotlib.image as mpimg

import numpy as np
import random
import time
import sys



qtCreatorFile = "design/diplom.ui"  # Enter file here.

class MatplotlibWidget(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        loadUi(qtCreatorFile, self)
        self.setWindowTitle("Дипломна робота студента групи БС-52 Янкового І.О. ")
        self.buttonLoader.clicked.connect(self.choose_file)
        self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))
        self.setWindowIcon(QtGui.QIcon("logo.png"))


    def choose_file(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Оберіть зображення", "",
                                                  "All Files (*);;Image files (*.bmp *.png)", options=options)
        if fileName:
            self.img = mpimg.imread(fileName)
            print(fileName)
            self.MplWidget.canvas.axes.clear()
            self.MplWidget.canvas.axes.imshow(self.img)
            self.MplWidget.canvas.axes.set_title('Обране зображення')
            self.MplWidget.canvas.draw()


if __name__ == "__main__":
    app = QApplication([])
    window = MatplotlibWidget()
    window.show()
    sys.exit(app.exec_())

    #sys.exit()
