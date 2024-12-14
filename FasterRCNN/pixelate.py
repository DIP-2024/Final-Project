from PIL import Image

def pixelate_box(image, box, pixel_size=10):
    """
    Pixelates a region of an image defined by a bounding box.
    
    Args:
        image (PIL.Image): Input image.
        box (tuple): (xmin, ymin, xmax, ymax) coordinates of the bounding box.
        pixel_size (int): Size of the pixels for the pixelation effect.
    
    Returns:
        PIL.Image: Image with pixelated region.
    """
    xmin, ymin, xmax, ymax = map(int, box)
    cropped = image.crop((xmin, ymin, xmax, ymax))
    small = cropped.resize((cropped.width // pixel_size, cropped.height // pixel_size), Image.NEAREST)
    pixelated = small.resize(cropped.size, Image.NEAREST)
    image.paste(pixelated, (xmin, ymin))
    return image
