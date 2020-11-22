import tkinter as tk
import os
from QRGenerator import QRCodeClassic
import svgwrite as svg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        root.title("QR-Code Generator")

    def create_widgets(self):
        self.text = tk.Text(self, width=50, height=10)
        self.text.pack()
        self.btnGenerate = tk.Button(self)
        self.btnGenerate["text"] = "Generate And Save SVG"
        self.btnGenerate["command"] = self.generateSVG
        self.btnGenerate.pack(side="right")

        self.btnSave = tk.Button(self)
        self.btnSave["text"] = "Save PDF And PNG"
        self.btnSave["command"] = self.saveImg
        self.btnSave.pack(side="right")

        self.quit = tk.Button(self, text="QUIT", fg="red",command=self.master.destroy)
        self.quit.pack(side="left")

        self.btnDelete = tk.Button(self)
        self.btnDelete["text"] = "Delete"
        self.btnDelete["command"] = self.Delete
        self.btnDelete.pack(side="left")

    def generateSVG(self):
        a = QRCodeClassic.generate_qr(self.text.get("1.0","end-1c"), QRCodeClassic.Ecc(0,1), -1)
        b = a.to_svg_str(5)
        print(b)
        bb = b[b.find('d="')+3:]
        bbb = bb[:bb.find('"')]

        dwg = svg.Drawing(filename='QR-Code.svg', size=('100%','100%'),viewBox="0 0 31 31", stroke="none")
        dwg.add(dwg.path(d=bbb))
        dwg.save()

    def saveImg(self):
        drawing = svg2rlg("QR-Code.svg")
        renderPDF.drawToFile(drawing, "QR-Code.pdf")
        renderPM.drawToFile(drawing, "QR-Code.png", fmt="PNG")

    def Delete(self):
        os.remove("QR-Code.svg")
        os.remove("QR-Code.png")
        os.remove("QR-Code.pdf")



root = tk.Tk()
app = Application(master=root)
app.mainloop()

