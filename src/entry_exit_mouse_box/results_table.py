from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QApplication, QMainWindow,
                            QTableWidget, QTableWidgetItem, QFileDialog)
from PyQt5.QtGui import QColor
import sys


class ResultsTable(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Data Table')
        self.initUI()

    def initUI(self):
        # Central widget
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)

        # Layout
        self.layout = QVBoxLayout(self.centralWidget)

        # Table
        self.table = QTableWidget()
        self.layout.addWidget(self.table)  # Add table to layout

        # Assume we have some data structure holding CSV-like data
        columnHeaders = ['Column 1', 'Column 2', 'Column 3']
        rowHeaders = ['Row 1', 'Row 2']
        rowData = [['Row1-Col1', 'Row1-Col2', 'Row1-Col3'],
                   ['Row2-Col1', 'Row2-Col2', 'Row2-Col3'],
                   # Add more rows as needed
                  ]

        self.table.setColumnCount(len(columnHeaders))
        self.table.setRowCount(len(rowData))
        self.table.setHorizontalHeaderLabels(columnHeaders)
        self.table.setVerticalHeaderLabels(rowHeaders)

        for row, data in enumerate(rowData):
            for column, value in enumerate(data):
                item = QTableWidgetItem(value)
                # Set background color for the cell
                item.setBackground(QColor(255, 255, 200))  # Light yellow background
                self.table.setItem(row, column, item)

        # Export Button
        self.exportButton = QPushButton('Export Data')
        self.exportButton.clicked.connect(self.exportData)
        self.layout.addWidget(self.exportButton)

    def exportData(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            print(f"Exporting data to {fileName}")
            # Implement the actual export functionality here
            # This could involve iterating over the table's items and writing them to a CSV file


"""

Chaque type de donnée devrait avoir son conteneur spécifique.
La gestion de la donnée est déléguée à cette classe, qui a une méthode show au besoin.

- Pour la visibilité des souris, chaque colonne représente une boite qui sera un booléen de visibilité.
  On part sur du rouge si invisible et bleu si visible.

- Pour le bilan d'entrées et sorties, chaque colonne représente une boite qui sera un compteur d'entrées et sorties.
  La première ligne correspond au nombre d'entrées et sorties total.

- Pour le temps passé dans chaque espace pour chaque souris.
  Chaque colonne représente une boite qui sera un compteur de temps.
  Chaque ligne représente une session.
  Chaque boite a deux colonnes, une pour le temps en intérieur, et l'autre pour le temps en extérieur.

"""