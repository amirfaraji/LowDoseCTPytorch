import numpy as np
import matplotlib.pyplot as plt


def visualize(*args):
    """
    Summary:
        Visualize multiple images in a subplot. Dimensions are always (1, number of images).

    Args:
        *args: multiple array images.
    """

    f, ax_array = plt.subplots(1, len(args), sharey=True)
    for i, arg in enumerate(args):
        ax_array[i].imshow(arg)

    plt.show()


def overlay_mask(image, mask, mask_colour: list = [1, 0, 0]) -> np.array:
    """
    Summary:
        Overlays a mask on a multichannel image and displays the overlayed image.

    Args:
        image: Must be a multichannel image.
        mask: Mask for the image.
        mask_colour (list, optional): [description]. Defaults to [1, 0, 0].

    Returns:
        overlay_img (np.array): Optional return of the overlayed image with mask.
    """

    image, mask = np.asarray(image), np.asarray(mask)
    overlay_img = np.copy(image)

    idx = np.where(mask == 1)
    overlay_img[idx] = mask_colour

    plt.imshow(overlay_img)
    plt.show()

    return overlay_img


def plot_history(history: list, label: str = 'loss'):
    """
    Summary:
        Plotting the history of the training and val loss or metric.

    Args:
        history (list): History of training and val loss or metric.
    """

    plt.plot(history['train'], label=f'train {label}')
    plt.plot(history['val'], label=f'val {label}')
    plt.ylabel(label)
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
