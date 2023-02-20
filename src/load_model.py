import tensorflow as tf
import numpy as np
from tkinter import *
from tkinter import filedialog
import tkinter as tk
from PIL import Image,ImageTk
import os
import time

def showImage():
    fileName=filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Select Image File",
        filetype=(("JPG File", "*.jpg"),("PNG File", "*.png"),("All File", "how are you .txt"))
    )

    img = Image.open(fileName)
    img = ImageTk.PhotoImage(img)
    lbl.configure(image=img)
    lbl.image=img

    IMG_SIZE = 200

    global testimg
    testimg = Image.open(fileName)
    testimg = testimg.convert('L')
    testimg = testimg.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)

def findClass(result):
    max = 0
    position = 0
    counter = 1
    for i in result[0]:
        if(i > max):
            max = i
            position = counter
        counter = counter + 1

    #print("max value of result array: " , max)
    #print("class: " , position)

    className = ""
    predictedLength = ""

    if position == 1:
        #print("level-1")
        className = "level-1"
        predictedLength = "0-20 cm"
    elif position == 2:
        #print("level-2")
        className = "level-2"
        predictedLength = "20-28 cm"
    elif position == 3:
        #print("level-3")
        className = "level-3"
        predictedLength = "Over 28 cm"

    return className,predictedLength

def runModel():
    # load the pre-trained network
    print("[INFO] loading pre-trained network...")
    start = time.time()
    model = tf.keras.models.load_model('model1.h5')
    IMG_SIZE = 200
    result = model.predict(np.array(testimg).reshape(-1, IMG_SIZE, IMG_SIZE, 1))
    #print("Prediction rates : " , result)

    className, predictedLength = findClass(result)
    end = time.time()
    print("Final class: ", className)
    print("Predicted Length: ", predictedLength)
    print("The time of execution of above program is :", (end-start), "seconds")
    resultStr = "Predicted Length: " + predictedLength
    etiket.configure(text=resultStr)


root = Tk()
fram = Frame(root)
fram.pack(side = BOTTOM,padx=15, pady=15)
lbl = Label(root)
lbl.pack()

btn = Button(fram, text="Select Image", command=showImage)
btn.pack(side = tk.LEFT)

btn2 = Button(fram, text="Run", command=runModel)
btn2.pack(side = tk.LEFT, padx=11)

btn3 = Button(fram, text="Exit", command=lambda:exit())
btn3.pack(side = tk.LEFT, padx=12)

root.title("Bean Classifier")
root.geometry("850x700")

etiket = tk.Label(text="Select bean image", font="Verdana 22 bold")
etiket.pack(side = tk.BOTTOM)


root.mainloop()
