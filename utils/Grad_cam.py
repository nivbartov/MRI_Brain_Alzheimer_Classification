import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
class Grad_cam():
    def __init__(self, model, last_conv_layer):
        self.model = model
        self.gradients = None
        self.feature_maps = None
        self.set_last_conv(last_conv_layer)

    def forward(self, img):
        # self.model.zero_grad()
        img.requires_grad_()
        self.model.eval()
        predictions = self.model(img)
        predicted_class = predictions.argmax()
        
        # Get the gradients with respect to the predicted class
        predictions[..., predicted_class].backward()
        
        # Get the mean over height and width in all feature maps
        gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Weight the feature maps by the gradients
        weighted_feature_maps = gradients * self.feature_maps
        
        # Get the mean of all feature maps
        heatmap = torch.mean(weighted_feature_maps, dim=1).squeeze()

        # Clip the heatmap to retain only positive influences
        heatmap = torch.clip(heatmap, 0)
        
        # Normalize the heatmap
        heatmap /= torch.max(heatmap)
        
        return heatmap.detach().cpu().numpy(), predicted_class

    def save_grads(self, module, input, grad_output):
        self.gradients = grad_output[0]

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output
    
    def set_last_conv(self, last_conv_layer):
        """Set the specified convolutional layer and register hooks."""
        if not isinstance(last_conv_layer, torch.nn.Conv2d):
            raise ValueError("The last_conv_layer must be an instance of torch.nn.Conv2d.")

        # Save feature maps of the specified conv layer
        last_conv_layer.register_forward_hook(self.save_feature_maps)
        
        # Save the gradients, from the last linear layer to conv
        last_conv_layer.register_backward_hook(self.save_grads)
        


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
    
    class_names = ['VeryMildDemented', 'NonDemented', 'ModerateDemented', 'MildDemented']
    # Map class indices to class names
    true_class_name = class_names[true_class.item()]
    predicted_class_name = class_names[predicted_class.item()]
    
    plt.figure(figsize=(10, 6))

    # Set the title to the predicted class name with adjusted padding
    plt.suptitle("Predicted class: " + predicted_class_name, y=0.76, fontweight='bold')  # Adjust y value as needed

    plt.subplot(1, 3, 1)
    plt.imshow(array_img)
    plt.title("Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(applied_img)
    plt.title("Image with Heatmap")
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

def display_correct_grad_cam_images(testloader, grad_cam, device, num_images_to_show=5):
    correct_count = 0
    
    for images, labels in testloader:
        # Move images and labels to the same device as the model
        images, labels = images.to(device), labels.to(device)
        
        # Process each image in the batch individually
        for i in range(images.size(0)):
            img = images[i].unsqueeze(0)  # Add batch dimension for a single image
            true_class = labels[i]
            
            # Get Grad-CAM heatmap and predicted class
            heatmap, predicted_class = grad_cam.forward(img)
            
            # Check if the prediction is correct
            if predicted_class == true_class:
                # Generate and plot all three images for the correctly classified image
                Generate_cam(img=img, grad_cam=grad_cam, true_class=true_class, plot=True)
                
                correct_count += 1
                
                # Stop after showing the specified number of correctly classified images
                if correct_count >= num_images_to_show:
                    return  # Exit the function after showing the required images

    print(f"Displayed {correct_count} correctly classified images.")