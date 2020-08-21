import numpy as np
import matplotlib.pyplot as plt


def visualize(*args):
    f, ax_array = plt.subplots(1, len(args), sharey=True)
    for i, arg in enumerate(args):
        ax_array[i].imshow(arg)
    plt.show()

def overlay(image1, image2, image_2_is_mask=False, mask_colour=[1,0,0]):
    image1, image2 = np.asarray(image1), np.asarray(image2)
    overlay_img = np.copy(image1)

    if image_2_is_mask:
        idx = np.where(image2 == 1)
        overlay_img[idx] = mask_colour
    else:
        pass

    plt.imshow(overlay_img)
    plt.show()
    return overlay_img

def plot_history(history):
    plt.plot(history['train'], label='train loss')
    plt.plot(history['val'], label='val loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()