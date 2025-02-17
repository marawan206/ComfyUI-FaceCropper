import torch
import torch.nn.functional as F
from typing import Tuple

class MarwanFaceCropping:
    """
    A custom node to crop an image to a 2:3 aspect ratio
    """
    CATEGORY = "Image Processing"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    NODE_DISPLAY_NAME = "Nodogoro Cropper (2:3 Ratio)"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    def execute(self, image: torch.Tensor) -> Tuple[torch.Tensor]:
        try:
            # Print input shape for debugging
            print(f"Original input shape: {image.shape}")

            # Get dimensions
            batch, height, width, channels = image.shape
            
            # Calculate crop width for 2:3 ratio (keeping height constant)
            target_width = int(height * (2/3))
            
            # Calculate crop boundaries (center crop)
            start_x = (width - target_width) // 2
            end_x = start_x + target_width
            
            # Perform the crop
            cropped = image[:, :, start_x:end_x, :]
            
            print(f"Final output shape: {cropped.shape}")
            
            # Verify we have the correct ratio
            assert cropped.shape[2] == int(cropped.shape[1] * (2/3)), "Output is not 2:3 ratio"
            
            return (cropped,)
            
        except Exception as e:
            print(f"Error in cropping: {str(e)}")
            print(f"Input shape: {image.shape}")
            return (image,)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "MarwanFaceCropping": MarwanFaceCropping
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarwanFaceCropping": "Face Cropper (2:3 Ratio)"
}