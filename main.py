"""
    This is main class of Project Gold price predicition
"""
import tkinter as tk
from lib.GUI.StockGraphicGui import StockGraphicGui

if __name__ == "__main__":
    root = tk.Tk()
    gui = StockGraphicGui(root)
    root.mainloop()
