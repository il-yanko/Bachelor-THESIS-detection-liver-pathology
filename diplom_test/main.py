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

import radiomics_single as rs

qtCreatorFile = "design/diplom.ui"  # Enter file here.

class MatplotlibWidget(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        loadUi(qtCreatorFile, self)
        self.FlagLoaded = False
        self.setWindowTitle("Дипломна робота студента групи БС-52 Янкового І.О. ")
        self.buttonLoader.clicked.connect(self.choose_file)
        self.buttonAnalyze.clicked.connect(self.analyze)
        #self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))
        self.setWindowIcon(QtGui.QIcon("logo.png"))

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('Файл')
        helpMenu = mainMenu.addMenu('Допомога')

        buttonLoaderMenu = QAction('Завантаження', self)
        buttonLoaderMenu.setShortcut('Ctrl+L')
        buttonLoaderMenu.setStatusTip('Завантажити область інтересу для подальшого аналізу')
        buttonLoaderMenu.triggered.connect(self.choose_file)
        fileMenu.addAction(buttonLoaderMenu)

        buttonAnalyzeMenu = QAction('Аналіз', self)
        buttonAnalyzeMenu.setShortcut('Ctrl+A')
        buttonAnalyzeMenu.setStatusTip('Проаналізувати завантажене зображення')
        buttonAnalyzeMenu.triggered.connect(self.analyze)
        fileMenu.addAction(buttonAnalyzeMenu)

        buttonExit = QAction('Вихід', self)
        buttonExit.setShortcut('Ctrl+Q')
        buttonExit.setStatusTip('Вийти з додатку')
        buttonExit.triggered.connect(sys.exit)
        fileMenu.addAction(buttonExit)

        buttonInfo = QAction('Додаток', self)
        buttonInfo.setShortcut('Ctrl+I')
        buttonInfo.setStatusTip('Отримати інформацію про додаток')
        self.msgBox = QMessageBox(self)
        self.msgBox.setIcon(QMessageBox.Information)
        self.msgBox.setWindowTitle("Додаток")
        self.msgBox.setText("Цей програмний додаток забезпечує завантаження області інтересу та прогнозування можливих дифузних патологій у пацієнта.")
        buttonInfo.triggered.connect(self.msgBox.exec_)
        helpMenu.addAction(buttonInfo)

        buttonInfo = QAction('Розробник', self)
        buttonInfo.setShortcut('Ctrl+D')
        buttonInfo.setStatusTip('Отримати інформацію про розробника')
        self.msgBox = QMessageBox(self)
        self.msgBox.setIcon(QMessageBox.Information)
        self.msgBox.setWindowTitle("Додаток")
        self.msgBox.setText("Цей програмний додаток було розроблено студентом 4 курсу Янковим І.О.\n"
                            "\nНТТУ КПІ ім. Ігоря Сікорського:\n"
                            "Факультет Біомедичної Інженерії (ФБМІ)\n"
                            "Академічна одиниця: група БС-52\n"
                            "Науковий керівник: Настенко Є.А.")
        buttonInfo.triggered.connect(self.msgBox.exec_)
        helpMenu.addAction(buttonInfo)

    def analyze(self):
        if (self.FlagLoaded):
            self.labelResult.setText(rs.signle_predition(self.path))
        else:
            self.labelResult.setText("Зображення не було обрано")
    def choose_file(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Оберіть зображення", "",
                                                  "All Files (*);;Image files (*.bmp *.png)", options=options)
        if fileName:
            self.path = fileName
            self.img = mpimg.imread(self.path)
            self.MplWidget.canvas.axes.clear()
            self.MplWidget.canvas.axes.imshow(self.img)
            self.MplWidget.canvas.axes.set_title('Обране зображення')
            self.MplWidget.canvas.draw()
            self.FlagLoaded = True


if __name__ == "__main__":
    app = QApplication([])
    window = MatplotlibWidget()
    window.show()
    sys.exit(app.exec_())