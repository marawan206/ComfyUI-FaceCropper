# Face Cropper Node (2:3 Ratio)

## Overview
The **Face Cropper Node** (`MarwanFaceCropping`) is a custom image processing node designed for **ComfyUI**. It takes an input image and crops it to a **2:3 aspect ratio**, ensuring that most of the subject remains in the frame while maintaining the correct proportions.

## Features
- Automatically crops images to a 2:3 ratio.
- Maintains the maximum possible height while adjusting width.
- Handles various image sizes and shapes, including non-square images.
- Includes error handling for incorrect input shapes.

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/marawan206/ComfyUI-FaceCropper.git
   cd face-cropper-node
   ```
2. Install dependencies if needed:
   ```sh
   pip install torch pytest
   ```
3. Add the node to your **ComfyUI** workflow.

## Usage
The node expects an input image tensor of shape `(batch, height, width, channels)`, where:
- `batch` is the number of images.
- `height` remains unchanged.
- `width` is adjusted to maintain a **2:3 ratio**.

Example:
```python
import torch
from face_cropper_node import MarwanFaceCropping

image = torch.rand(1, 1024, 1024, 3)  # Example input image
node = MarwanFaceCropping()
cropped, = node.execute(image)
print(cropped.shape)  # Output shape should be (1, 1024, 682, 3)
```

## Running Tests
To verify functionality, run:
```sh
pytest test.py
```

## Contributing
Feel free to submit issues or pull requests to improve the node.

## License
MIT License. See `LICENSE` for details.

