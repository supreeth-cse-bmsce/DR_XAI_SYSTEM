import torch
import cv2
import numpy as np

def generate_gradcam(model, image_tensor, save_path):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.layer4[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1)
    output[:, pred_class].backward()

    grad = gradients[0]
    act = activations[0]

    weights = grad.mean(dim=[2,3], keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze().detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cv2.imwrite(save_path, heatmap)

    return pred_class.item()