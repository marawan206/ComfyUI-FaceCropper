import pytest
import torch
from face_cropper_node import NodoFaceCropping

def test_valid_crop():
    node = NodoFaceCropping()
    image = torch.rand(1, 1024, 1024, 3)  # Batch size 1, square image
    cropped, = node.execute(image)
    
    assert cropped.shape[1] == 1024, "Height should remain the same"
    assert cropped.shape[2] == int(1024 * (2/3)), "Width should be 2:3 ratio of height"

def test_invalid_input_shape():
    node = NodoFaceCropping()
    image = torch.rand(1024, 1024, 3)  # Missing batch dimension
    
    with pytest.raises(ValueError):
        node.execute(image)

def test_non_square_input():
    node = NodoFaceCropping()
    image = torch.rand(1, 800, 1200, 3)  # Non-square image
    cropped, = node.execute(image)
    
    assert cropped.shape[1] == 800, "Height should remain the same"
    assert cropped.shape[2] == int(800 * (2/3)), "Width should be 2:3 ratio of height"

def test_small_image():
    node = NodoFaceCropping()
    image = torch.rand(1, 300, 400, 3)  # Small image
    cropped, = node.execute(image)
    
    assert cropped.shape[1] == 300, "Height should remain the same"
    assert cropped.shape[2] == int(300 * (2/3)), "Width should be 2:3 ratio of height"

def test_large_image():
    node = NodoFaceCropping()
    image = torch.rand(1, 2048, 4096, 3)  # Large image
    cropped, = node.execute(image)
    
    assert cropped.shape[1] == 2048, "Height should remain the same"
    assert cropped.shape[2] == int(2048 * (2/3)), "Width should be 2:3 ratio of height"
