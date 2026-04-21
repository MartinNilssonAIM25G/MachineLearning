# CNN Interpretability with LayerCAM

This notebook explores interpretability in convolutional neural networks using **LayerCAM** applied to a pretrained **VGG16** model.

The goal is to visualize which parts of an image influence the model’s predictions and how these activations change across different convolutional layers.

## Examples used

The notebook analyzes several cases:

- **Tibetan mastiff vs Shiba Inu (Tifa)** – a clear positive vs negative example.
- **Green mamba vs a visually similar snake** – examining how the model focuses on distinguishing features.
- **CD player vs Sega Dreamcast** – an example where the object is not present in ImageNet and is mapped to the closest known class.

## Multilayer analysis

For selected images, attribution maps are extracted from:

- Early convolution layer (`features.2`)
- Middle convolution layer (`features.14`)
- Late convolution layer (`features.28`)

This shows how feature representations evolve from simple edge detection to more semantic object parts.

## Requirements

Main libraries used:

- Python
- PyTorch
- torchvision
- torchcam
- matplotlib
- Pillow

## References

Simonyan, K., & Zisserman, A. (2015).  
*Very Deep Convolutional Networks for Large-Scale Image Recognition.*

Deng, J. et al. (2009).  
*ImageNet: A Large-Scale Hierarchical Image Database.*