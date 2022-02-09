
import numpy as np
from PIL import Image

# --------------- RAW PROCESSING ---------------
# ----------------------------------------------
class Cube():
    def __init__(self, np_array) -> None:
        self.value = np_array
    
    def as_nparray(self):
        return self.value
    
    def rotate_face(self, axe:str):
        """ axes = 'x, 'y' and 'z' | cube shape assumed = (z, y, x)
            -> Rotates the face of the cube 90 degrees over the axes selected
        """
        z_elements, y_elements, x_elements = self.value.shape
        
        rotated_cube = []
        if axe == 'x': 
            # Cube shape to achieve = (y, z, x)
            expected_shape = (y_elements, z_elements, x_elements)
            for z in range(z_elements-1, -1, -1):
                for y in range(y_elements):
                    if z == z_elements-1: rotated_cube.append([[0]*x_elements]*z_elements)
                    rotated_cube[y][z_elements-1-z] = self.value[z][y_elements-1-y]
            rotated_cube = np.array(rotated_cube)
            assert rotated_cube.shape == expected_shape
            self.value = np.array(rotated_cube)
            self.vflip_slices(); self.hflip_slices()
        if axe == 'y':
            ...
        if axe == 'z':
            ...
            
        return self
    
    def resize_slices(self, size:tuple[int,int]):
        resized = []
        for i in range(self.value.shape[0]):
            img_obj = Image.fromarray(self.value[i]).resize(size)
            resized.append(np.array(img_obj))
        self.value = np.array(resized)
                    
        return self
    
    def project(self):
        _, y_elements, x_elements = self.value.shape
        max_slice_vals = np.zeros((y_elements, x_elements))
        for y in range(y_elements):
            for x in range(x_elements):
                transposed = np.transpose(self.value)
                pixel_max = np.max(transposed[x][y])
                max_slice_vals[y][x] = int(pixel_max)
        p = np.array(max_slice_vals)
        norm_projected = (((p - p.min()) / (p.max() - p.min())) * 255.9).astype(np.uint8)
        self.value = norm_projected
        
        return self   
    
    def vflip_slices(self):
        vflipped = []
        for slice_ in self.value:
            vflipped.append(np.flipud(slice_))
        self.value = np.array(vflipped)
        
    def hflip_slices(self):
        hflipped = []
        for slice_ in self.value:
            hflipped.append(np.fliplr(slice_))
        self.value = np.array(hflipped)

class RawProcessingError(Exception):
    pass

def process_oct(raw_path:str, width_pixels:int, height_pixels:int, num_images:int=1, 
                    vertical_flip:bool=False, resize:tuple[int, int]=None) -> Cube:
    """ Returns Numpy array.
    
        -> reads cube with bit_depth=16, mode='unsigned'
    """
    if num_images < 1:
        raise RawProcessingError("'num_images' can't be less than 1")
    
    # En binario con 16 bits representamos del 0 - 255
    # En hexadecimal con 2 byte representamos del 0 - 255 (FF) (La info de un pixel)
    bit_depth = 16
    binary_hex_ratio = 16/2
    hex_depth = int(bit_depth/binary_hex_ratio)
    pixel_length = hex_depth
    slice_pixels = width_pixels*height_pixels
    slice_length = slice_pixels*pixel_length

    cube_data = []
    with open(raw_path, 'rb') as raw_file:
        volume:str = raw_file.read()
        if len(volume) < slice_length*num_images:
            msg = "'num_images' is incorrect (too much images with that image size)"
            raise RawProcessingError(msg)
        for i in range(num_images):
            raw_slice = volume[i*slice_length:(i+1)*slice_length]
            # Usamos Image.frombytes porque lo lee muy rapido (optimizado), usando bucles normales tarda mucho
            slice_ = Image.frombytes(mode="I;16", size=(width_pixels, height_pixels), data=raw_slice)
            if resize is not None: slice_ = slice_.resize(resize)
            slice_ = np.array(slice_)
            if vertical_flip: slice_ = np.flipud(slice_)
            cube_data.append(slice_)
    cube_data = np.array(cube_data)
    
    return Cube(cube_data)