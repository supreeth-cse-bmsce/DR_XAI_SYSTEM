from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import datetime

def generate_report(prediction, confidence, heatmap_path):
    doc = SimpleDocTemplate("DR_Report.pdf")
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Diabetic Retinopathy Detection Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Prediction: {prediction}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {confidence}%", styles["Normal"]))
    elements.append(Paragraph(f"Date: {datetime.datetime.now()}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Image(heatmap_path, width=4*inch, height=4*inch))

    doc.build(elements)