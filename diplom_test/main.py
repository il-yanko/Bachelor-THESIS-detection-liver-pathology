#!/usr/bin/python3
# -*- coding: utf-8 -*-

from PyQt5.uic import loadUi
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

#from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import matplotlib.image as mpimg

import sys

import radiomics_single as rs

qtCreatorFile = "design/diplom.ui"  # Enter file here.

class MatplotlibWidget(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        loadUi(qtCreatorFile, self)
        self.FlagLoaded = False
        self.setWindowTitle("Texture Analysis for Diffuse Liver Diseases")
        self.buttonLoader.clicked.connect(self.choose_file)
        self.buttonAnalyze.clicked.connect(self.analyze)
        #self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))
        self.setWindowIcon(QIcon("app.ico"))

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        helpMenu = mainMenu.addMenu('Help')

        buttonLoaderMenu = QAction('Download', self)
        buttonLoaderMenu.setShortcut('Ctrl+D')
        buttonLoaderMenu.setStatusTip('Download the region of the interest')
        buttonLoaderMenu.triggered.connect(self.choose_file)
        fileMenu.addAction(buttonLoaderMenu)

        buttonAnalyzeMenu = QAction('Analysis', self)
        buttonAnalyzeMenu.setShortcut('Ctrl+A')
        buttonAnalyzeMenu.setStatusTip('Analyse the loaded region of the interest')
        buttonAnalyzeMenu.triggered.connect(self.analyze)
        fileMenu.addAction(buttonAnalyzeMenu)

        buttonExit = QAction('Quit', self)
        buttonExit.setShortcut('Ctrl+Q')
        buttonExit.setStatusTip('Quit out of application')
        buttonExit.triggered.connect(sys.exit)
        fileMenu.addAction(buttonExit)

        buttonLaunch = QAction('How to run', self)
        buttonLaunch.setStatusTip('Get info about how to run the application')
        self.msgBox1 = QMessageBox(self)
        self.msgBox1.setIcon(QMessageBox.Information)
        self.msgBox1.setWindowTitle("How to run")
        self.msgBox1.setText("To run the classifier:\n1) push the button <Choose an image>\n2) push the button <Analyse>")
        buttonLaunch.triggered.connect(self.msgBox1.exec_)
        helpMenu.addAction(buttonLaunch)



        buttonInfo = QAction('Application', self)
        buttonInfo.setStatusTip('Get info about the application')
        self.msgBox2 = QMessageBox(self)
        self.msgBox2.setIcon(QMessageBox.Information)
        self.msgBox2.setWindowTitle("Application")
        self.msgBox2.setText("This application give an ability to load ROI and predict a probable presence of diffuse liver diseases.")
        buttonInfo.triggered.connect(self.msgBox2.exec_)
        helpMenu.addAction(buttonInfo)

        buttonInfo = QAction('Developer', self)
        buttonInfo.setStatusTip('Get info about the developer')
        self.msgBox3 = QMessageBox(self)
        self.msgBox3.setIcon(QMessageBox.Information)
        self.msgBox3.setWindowTitle("Developer")
        self.msgBox3.setText("This application was developed by Illia Yankovyi, the student of the 4th year"
                            "\nNTUU Igor Sikorsky Kyiv Polytechnic Institute:"
                            "\nFaculty of Biomedical Engineering (FBME)\n"
                            "\nAcademic unit:BS-52 group\n"
                            "\nSupervisor: Nastenko I., M.D., Candidate of Engineering Sciences, Senior Research Fellow.")
        buttonInfo.triggered.connect(self.msgBox3.exec_)
        helpMenu.addAction(buttonInfo)

        self.labelTitle.setText('Classifier of Diffuse Liver Diseases')
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.labelTitle.setFont(font)
        self.labelTitle.setAlignment(Qt.AlignCenter)
        self.buttonAnalyze.setText('Analyze Image')
        self.buttonLoader.setText('Download Image')
        self.labelResult.setText('To get a prediction:\n\n1) Download the region of interest;\n2) Run the analysis.')

    def analyze(self):
        if (self.FlagLoaded):
            self.labelResult.setText(rs.signle_prediction(self.path))
        else:
            self.labelResult.setText("Image was not chosen!\n\nPlease choose the image\nbefore running the Analysis")
            self.msgBox4 = QMessageBox(self)
            self.msgBox4.setIcon(QMessageBox.Warning)
            self.msgBox4.setWindowTitle("Error! Image was not chosen.")
            self.msgBox4.setText(
                "Image was not chosen! Please choose the image before running the Analysis.")
            self.msgBox4.exec_()


    def choose_file(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Choose an image", "",
                                                  "Image (*.bmp *.png *.jpeg *.jpg)", options=options)
        extensions = ['png', 'jpg', 'jpeg', 'bmp']
        fileExtension = (fileName.split('.'))[-1].lower()
        if fileName:
            if fileExtension in extensions:
                self.path = fileName
                self.img = mpimg.imread(self.path)
                self.MplWidget.canvas.axes.clear()
                self.MplWidget.canvas.axes.imshow(self.img)
                self.MplWidget.canvas.axes.set_title('Chosen image')
                self.MplWidget.canvas.draw()
                self.FlagLoaded = True
            else:
                self.labelResult.setText("Chosen filetype is not supported.\nSupported filetypes:\nBMP, PNG, JPEG, JPG")
                self.msgBox5 = QMessageBox(self)
                self.msgBox5.setIcon(QMessageBox.Warning)
                self.msgBox5.setWindowTitle("Error! Chosen filetype is not supported.")
                self.msgBox5.setText(
                    "Chosen filetype is not supported.\nSupported filetypes:\nBMP, PNG, JPEG, JPG.")
                self.msgBox5.exec_()

if __name__ == "__main__":
    app = QApplication([])
    window = MatplotlibWidget()
    window.show()
    sys.exit(app.exec_())