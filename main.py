"""
    This is main class of Project Gold price predicition
"""
import tkinter as tk
from lib.GUI.GoldPricePrediction import CryptoGraphicGui

if __name__ == "__main__":
    root = tk.Tk()
    gui = CryptoGraphicGui(root)
    root.mainloop()
