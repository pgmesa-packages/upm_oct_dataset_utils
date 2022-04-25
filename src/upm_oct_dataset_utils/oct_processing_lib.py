
import math
import cv2
import numpy as np
from PIL import Image

# --------------- RAW PROCESSING ---------------
# ----------------------------------------------
class Cube():
    def __init__(self, np_array) -> None:
        self.value = np_array
    
    def as_nparray(self):
        return self.value
    
    def rotate_face(self, axis:str):
        """ axes = 'x, 'y' and 'z' | cube shape assumed = (z, y, x)
            -> Rotates the face of the cube 90 degrees over the axes selected
        """
        z_elements, y_elements, x_elements = self.value.shape
        
        rotated_cube = []
        if axis == 'x': 
            # Cube shape to achieve = (y, z, x)
            expected_shape = (y_elements, z_elements, x_elements)
            for z in range(z_elements-1, -1, -1):
                for y in range(y_elements):
                    if z == z_elements-1: rotated_cube.append([[0]*x_elements]*z_elements)
                    rotated_cube[y][z_elements-1-z] = self.value[z][y_elements-1-y]
            rotated_cube = np.array(rotated_cube)
            assert rotated_cube.shape == expected_shape
            c = Cube(np.array(rotated_cube))
            c = c.vflip_slices()
            c = c.hflip_slices()
        if axis == 'y':
            ...
        if axis == 'z':
            ...
            
        return c
    
    def resize_slices(self, size:tuple[int,int]):
        resized = []
        for i in range(self.value.shape[0]):
            img_obj = Image.fromarray(self.value[i]).resize(size)
            resized.append(np.array(img_obj))
        c = Cube(np.array(resized))
                    
        return c
    
    def project(self):
        _, y_elements, x_elements = self.value.shape
        max_slice_vals = np.zeros((y_elements, x_elements))
        for y in range(y_elements):
            for x in range(x_elements):
                transposed = np.transpose(self.value)
                pixel_max = np.max(transposed[x][y])
                max_slice_vals[y][x] = int(pixel_max)
        p = np.array(max_slice_vals)
        c = Cube(p)
        
        return c   
    
    def vflip_slices(self):
        vflipped = []
        for slice_ in self.value:
            vflipped.append(np.flipud(slice_))
        return Cube(np.array(vflipped))
        
    def hflip_slices(self):
        hflipped = []
        for slice_ in self.value:
            hflipped.append(np.fliplr(slice_))
        return Cube(np.array(hflipped))
        
def reconstruct_OCTA(cube:Cube, kernel_size=(2,2), max_thickness_perc:float=None,
                            signal_threshold:float=0.4, smooth_lines:int=1, claheLimit:float=None):
    cube_array = cube.value
    _, y_elements, x_elements = cube_array.shape

    OCTA_reconstructed = np.zeros((y_elements, x_elements))

    # Dividimos en sectores, filtramos las capas de la imagen consideradas como ruido y 
    # de las capas restantes nos quedamos un porcentaje de profundidad
    x_step = int(np.ceil(x_elements/kernel_size[0]))
    y_step = int(np.ceil(y_elements/kernel_size[1]))
    # print(x_elements, y_elements)
    # print(x_step, y_step)
    for j in range(kernel_size[1]):
        for i in range(kernel_size[0]):
            y_q_init = y_step*j; y_q_end = y_step*(j+1)
            x_q_init = x_step*i; x_q_end = x_step*(i+1)
            # print(y_q_init, y_q_end)
            # print(x_q_init, x_q_end)
            q = cube_array[:, y_q_init:y_q_end, x_q_init:x_q_end]
            avgs = []
            for l in q:
                avg =  np.average(l)
                avgs.append(avg)
            layers = []
            for k, avg in enumerate(avgs):
                if avg > max(avgs)*signal_threshold:
                    layers.append(q[k])
            valid_layers = np.array(layers)
            if max_thickness_perc is not None:
                valid_layers = np.array(layers[:round(max_thickness_perc*len(layers))])
            q_recons = Cube(valid_layers).project().as_nparray()
            OCTA_reconstructed[y_step*j:y_step*(j+1), x_step*i:x_step*(i+1)] = q_recons
    
    if smooth_lines > 0:
        # --- Hacemos que los bordes entre sectores sean transiciones suaves
        # Smooth entre columnas
        for i in range(kernel_size[0]-1):
            column_index = x_step*(i+1)
            band = OCTA_reconstructed[:, column_index-(1*smooth_lines):column_index+(1*smooth_lines)]
            avg_column = np.average(band, axis=1)
            stacked_column = np.stack((avg_column,)*smooth_lines*2, axis=1)
            OCTA_reconstructed[:, column_index-(1*smooth_lines):column_index+(1*smooth_lines)] = stacked_column
        # Smooth entre filas
        for j in range(kernel_size[1]-1):
            row_index = y_step*(i+1)
            band = OCTA_reconstructed[:, row_index-(1*smooth_lines):row_index+(1*smooth_lines)]
            avg_row = np.average(band, axis=1)
            stacked_row = np.stack((avg_row,)*smooth_lines*2, axis=1)
            OCTA_reconstructed[:, row_index-(1*smooth_lines):row_index+(1*smooth_lines)] = stacked_row
    
    if claheLimit is not None:
        clahe = cv2.createCLAHE(clipLimit=claheLimit)
        OCTA_reconstructed = clahe.apply(np.uint16(OCTA_reconstructed))
                
    return OCTA_reconstructed

def reconstruct_ONH_OCTA(cube:Cube, kernel_size=(2,2), max_thickness_perc:float=0.5,
                            signal_threshold:float=0.5, smooth_lines:int=1, claheLimit:float=10):
    onh_recons = reconstruct_OCTA(
        cube, kernel_size=kernel_size, max_thickness_perc=max_thickness_perc,
        signal_threshold=signal_threshold, smooth_lines=smooth_lines, claheLimit=claheLimit
    )
    # projected = cube.project()
    # onh_recons = np.around((projected.value + onh_recons)/2)
     
    return onh_recons
    
    
def reconstruct_Angiography_OCTA(cube:Cube, kernel_size=(4,4), max_thickness_perc:float=0.4,
                            signal_threshold:float=0.4, smooth_lines:int=1, claheLimit:float=10):
    ang_recons = reconstruct_OCTA(
        cube, kernel_size=kernel_size, max_thickness_perc=max_thickness_perc,
        signal_threshold=signal_threshold, smooth_lines=smooth_lines, claheLimit=claheLimit
    )
    
    return ang_recons


def norm_volume(volume, bit_depth:int):
    maxim = math.pow(2, bit_depth) - 1
    norm_v = ((volume / maxim) * 255.9).astype(np.uint8)
    return norm_v

class RawProcessingError(Exception):
    pass

def process_oct(raw_path:str, width_pixels:int, height_pixels:int, num_images:int=1, 
                    vertical_flip:bool=True, resize:tuple[int, int]=None, reverse:bool=True) -> Cube:
    """ Returns Numpy array.
    
        -> reads cube with bit_depth=16, mode='unsigned'
    """
    if num_images < 1:
        raise RawProcessingError("'num_images' can't be less than 1")
    
    # En binario con 16 bits representamos del 0 - 65535
    # En hexadecimal con 2 byte representamos del 0 - 65535 (FFFF) (La info de un pixel)
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
    cube_data = norm_volume(cube_data, bit_depth=bit_depth)
    
    if reverse: cube_data = np.flip(cube_data, axis=1)
    
    return Cube(cube_data)