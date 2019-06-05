#!/usr/bin/python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtGui

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
        self.setWindowTitle("Дипломна робота студента групи БС-52 Янкового І.О. ")
        self.buttonLoader.clicked.connect(self.choose_file)
        self.buttonAnalyze.clicked.connect(self.analyze)
        #self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))
        self.setWindowIcon(QtGui.QIcon("app.ico"))

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

        buttonLaunch = QAction('Запуск', self)
        buttonLaunch.setStatusTip('Отримати інформацію про запуск класифікатора')
        self.msgBox1 = QMessageBox(self)
        self.msgBox1.setIcon(QMessageBox.Information)
        self.msgBox1.setWindowTitle("Запуск")
        self.msgBox1.setText("Для запуску класифікатора:\n1) натисніть кнопку <Обрати Зображення>\n2) натисніть кнопку <Аналізувати>")
        buttonLaunch.triggered.connect(self.msgBox1.exec_)
        helpMenu.addAction(buttonLaunch)



        buttonInfo = QAction('Додаток', self)
        buttonInfo.setStatusTip('Отримати інформацію про додаток')
        self.msgBox2 = QMessageBox(self)
        self.msgBox2.setIcon(QMessageBox.Information)
        self.msgBox2.setWindowTitle("Додаток")
        self.msgBox2.setText("Цей програмний додаток забезпечує завантаження області інтересу та прогнозування можливих дифузних патологій у пацієнта.")
        buttonInfo.triggered.connect(self.msgBox2.exec_)
        helpMenu.addAction(buttonInfo)

        buttonInfo = QAction('Розробник', self)
        buttonInfo.setStatusTip('Отримати інформацію про розробника')
        self.msgBox3 = QMessageBox(self)
        self.msgBox3.setIcon(QMessageBox.Information)
        self.msgBox3.setWindowTitle("Додаток")
        self.msgBox3.setText("Цей програмний додаток було розроблено студентом 4 курсу Янковим І.О.\n"
                            "\nНТТУ КПІ ім. Ігоря Сікорського:\n"
                            "Факультет Біомедичної Інженерії (ФБМІ)\n"
                            "Академічна одиниця: група БС-52\n"
                            "Науковий керівник: Настенко Є.А.")
        buttonInfo.triggered.connect(self.msgBox3.exec_)
        helpMenu.addAction(buttonInfo)

    def analyze(self):
        if (self.FlagLoaded):
            self.labelResult.setText(rs.signle_prediction(self.path))
        else:
            self.labelResult.setText("Зображення не було обрано")
    def choose_file(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Оберіть зображення", "",
                                                  "Зображення (*.bmp *.png *.jpeg *.jpg)", options=options)
        extensions = ['png', 'jpg', 'jpeg', 'bmp']
        fileExtension = (fileName.split('.'))[-1].lower()
        if fileName:
            if fileExtension in extensions:
                self.path = fileName
                self.img = mpimg.imread(self.path)
                self.MplWidget.canvas.axes.clear()
                self.MplWidget.canvas.axes.imshow(self.img)
                self.MplWidget.canvas.axes.set_title('Обране зображення')
                self.MplWidget.canvas.draw()
                self.FlagLoaded = True
            else:
                self.labelResult.setText("Обраний тип файлу не підтримується.\nТипи файлів, що підтримуються:\nBMP, PNG, JPEG, JPG")


if __name__ == "__main__":
    app = QApplication([])
    window = MatplotlibWidget()
    window.show()
    sys.exit(app.exec_())