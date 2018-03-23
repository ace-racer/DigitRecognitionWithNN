import tkinter
from tkinter import *

root = tkinter.Tk()

# Add widgets
T = Text(root, height=2, width=60)
T.pack()
T.insert(END, "MNIST Handwritten image recognizer\n")



root.mainloop()