import numpy as np
import cv2
from cv2 import dnn
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


def colorize(file_path):
    # Paths to load the model
    DIR = r"C:/Users/india/.ipython/black and white image colorization"
    PROTOTXT = os.path.join(DIR, r"models/colorization_deploy_v2.prototxt")
    POINTS = os.path.join(DIR, r"models/pts_in_hull.npy")
    MODEL = os.path.join(DIR, r"models/colorization_release_v2.caffemodel")

    # Load the input image
    
    img = cv2.imread(file_path)
    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    
    # Load the Model
    print("Load model")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)
    
    
    # Load centers for ab channel quantization used for rebalancing.
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    
    print("Colorizing the image")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    
    L = cv2.split(lab_img)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    
    colorized = (255 * colorized).astype("uint8")
    
    cv2.imshow("Original", img)
    cv2.imshow("Colorized", colorized)
    cv2.waitKey(0)
    return original, colorized

def open_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            original, colorized = colorize(file_path)
            display_images(original, colorized)
        except Exception as e:
            messagebox.showerror("Error", str(e))

def display_images(original, colorized):
    # Convert images from BGR to RGB
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)

    # Convert to PIL Images
    original_pil = Image.fromarray(original_rgb)
    colorized_pil = Image.fromarray(colorized_rgb)

    # Resize images to fit the GUI window
    original_pil = original_pil.resize((300, 300))
    colorized_pil = colorized_pil.resize((300, 300))

    # Convert to ImageTk
    original_tk = ImageTk.PhotoImage(original_pil)
    colorized_tk = ImageTk.PhotoImage(colorized_pil)

    # Update labels
    original_label.configure(image=original_tk)
    original_label.image = original_tk
    colorized_label.configure(image=colorized_tk)
    colorized_label.image = colorized_tk

# Initialize the main window
root = tk.Tk()
root.title("Black and White Image Colorization")

# Create and place the buttons and labels
open_button = tk.Button(root, text="Open Image", command=open_file)
open_button.pack(pady=10)

# Start the GUI event loop
root.mainloop()