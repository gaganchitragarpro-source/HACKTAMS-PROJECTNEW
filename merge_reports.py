import os
from PIL import Image
from pypdf import PdfWriter, PdfReader

# Convert PNG to a temporary PDF
img = Image.open("bridge_analytical_report.png")
if img.mode == 'RGBA':
    img = img.convert('RGB')
temp_pdf_path = "temp_bridge_image.pdf"
img.save(temp_pdf_path, "PDF", resolution=100.0)

# Merge the original PDF with the temporary PDF
merger = PdfWriter()
merger.append("bridge_analytical_report.pdf")
merger.append(temp_pdf_path)

merger.write("bridge_analytical_report_merged.pdf")
merger.close()

# Overwrite and clean up
os.replace("bridge_analytical_report_merged.pdf", "bridge_analytical_report.pdf")
os.remove("bridge_analytical_report.png")
os.remove(temp_pdf_path)

print("✅ Successfully merged reports!")
