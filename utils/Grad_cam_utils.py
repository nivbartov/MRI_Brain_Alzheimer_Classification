
import matplotlib.pyplot as plt
import cv2
import numpy as np

def tensor_to_img(tensor):
    """Converts a PyTorch tensor to a NumPy image.

    Args:
        tensor (Tensor): Image tensor of shape (C, H, W) or (B, C, H, W).

    Returns:
        np.array: Converted image as a NumPy array in (H, W, C) format.
    """
    # Check if tensor has batch dimension
    if tensor.ndim == 4:  # Shape (B, C, H, W)
        tensor = tensor.squeeze(0)  # Remove batch dimension

    # Detach from computation graph, permute to (H, W, C), and move to CPU
    img = tensor.detach().permute(1, 2, 0).cpu().numpy()  # Detach to avoid gradient issues

    # Clip values to [0, 1] range (for visualization) and convert to uint8
    img = np.clip(img, 0, 1)  # Ensure pixel values are in the correct range
    img = (img * 255).astype(np.uint8)  # Scale to [0, 255]

    return img

def Generate_cam(img, grad_cam, true_class, transformations=None, plot=True, save=False, save_path='cam_img'):

    heatmap, predicted_class = grad_cam.forward(img)
    array_img = tensor_to_img(img)  # Assuming tensor_to_img handles the conversion correctly

    applied_img, heatmap = apply_cam(array_img, heatmap)

    if plot:
        plot_cam(array_img, heatmap, applied_img, true_class, predicted_class, save, save_path)
    else:
        return array_img, heatmap, applied_img

def plot_cam(array_img, heatmap, applied_img, true_class, predicted_class, save, save_path):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(array_img)
    plt.title("Image, class = " + str(true_class.item()))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(applied_img)
    plt.title("Image with heatmap, predicted class = " + str(predicted_class.item()))
    plt.axis('off')
    
    if save:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

def apply_cam(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    applied_img = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)

    return applied_img, heatmap
