import tkinter
from tkinter import *
import MNISTImages
import random as rd
from PIL import Image, ImageTk



NUM_TEST_IMAGES = 10000
TEST_IMAGES_LOCATION = "H:\\KE4102-Sam\\code\\MNIST-data\\test_images\\test_image_{0}.png"
labels = []

def get_next_image_imageid_location():
    nextid = rd.randint(0, NUM_TEST_IMAGES - 1)
    # Create the image
    MNISTImages.generate_test_image(nextid)

    # Get the location of the image
    image_location = TEST_IMAGES_LOCATION.format(nextid)

    # Return the nextid and the image location pair
    return nextid, image_location


def generate_click():
    image_id, image_location = get_next_image_imageid_location()
    image = Image.open(image_location)

    # remove all existing labels from the UI
    for label in labels:
        label.destroy()

    photo = ImageTk.PhotoImage(image)
    label = Label(image=photo)
    label.image = photo  # keep a reference!
    label.grid()
    labels.append(label)





# GUI code below
root = tkinter.Tk()

#  Add a text header
T = Text(root, height=2, width=60)
T.insert(END, "MNIST Handwritten image recognizer\n")
T.grid()

# Add a button to generate the next image
generate_button = Button(root, command = generate_click, text = "Generate next image")
generate_button.grid()



root.mainloop()


