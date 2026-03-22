import sys
import torch
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

from model import load_model
from preprocessing import preprocess_image
from explainability import generate_gradcam
from report import generate_report

classes = ["No DR","Mild","Moderate","Severe","Proliferative"]


class DRApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Diabetic Retinopathy Detection System")
        self.setGeometry(200, 100, 1100, 700)

        self.model = load_model()

        self.initUI()

    def initUI(self):

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout()

        # Title
        title = QLabel("Explainable AI Framework for Diabetic Retinopathy Detection")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 20, QFont.Bold))

        main_layout.addWidget(title)

        # Image layout
        image_layout = QHBoxLayout()

        self.image_label = QLabel("Fundus Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(450, 400)
        self.image_label.setStyleSheet("border:2px solid gray;")

        self.heatmap_label = QLabel("Explainability Heatmap")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setFixedSize(450, 400)
        self.heatmap_label.setStyleSheet("border:2px solid gray;")

        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.heatmap_label)

        main_layout.addLayout(image_layout)

        # Prediction label
        self.result_label = QLabel("Prediction: ")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 14))

        main_layout.addWidget(self.result_label)

        # Buttons
        button_layout = QHBoxLayout()

        self.upload_btn = QPushButton("Upload Fundus Image")
        self.upload_btn.clicked.connect(self.upload_image)

        self.report_btn = QPushButton("Generate Report")
        self.report_btn.clicked.connect(self.generate_pdf)

        self.upload_btn.setStyleSheet("""
            QPushButton{
                background-color:#3498db;
                color:white;
                padding:10px;
                font-size:14px;
            }
        """)

        self.report_btn.setStyleSheet("""
            QPushButton{
                background-color:#27ae60;
                color:white;
                padding:10px;
                font-size:14px;
            }
        """)

        button_layout.addWidget(self.upload_btn)
        button_layout.addWidget(self.report_btn)

        main_layout.addLayout(button_layout)

        main_widget.setLayout(main_layout)

    def upload_image(self):

        file_path, _ = QFileDialog.getOpenFileName()

        if file_path:

            pixmap = QPixmap(file_path).scaled(450,400, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

            tensor = preprocess_image(file_path)

            logits = self.model(tensor.float())

            probs = torch.softmax(logits, dim=1)

            conf, pred = torch.max(probs,1)

            self.confidence = round(conf.item()*100,2)

            self.prediction = classes[pred.item()]

            heatmap_path = "heatmap.jpg"

            generate_gradcam(self.model, tensor, heatmap_path)

            heatmap_pixmap = QPixmap(heatmap_path).scaled(450,400, Qt.KeepAspectRatio)

            self.heatmap_label.setPixmap(heatmap_pixmap)

            self.result_label.setText(
                f"Prediction: {self.prediction} | Confidence: {self.confidence}%"
            )

    def generate_pdf(self):

        generate_report(self.prediction, self.confidence, "heatmap.jpg")

        QMessageBox.information(self, "Report", "PDF Report Generated!")


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = DRApp()

    window.show()

    sys.exit(app.exec_())