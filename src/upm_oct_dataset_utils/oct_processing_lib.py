
import math
import cv2
from scipy import signal
from scipy.fft import fft, fftfreq
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

def reconstruct_OCTA(cube:Cube, kernel_size=(2,2), strides=(1,1), smooth_lines:int=1, claheLimit:float=None):
    """ Imagen con valores entre 0-255"""
    assert kernel_size[0] >= strides[0] and kernel_size[1] >= strides[1]
    cube_array = cube.value
    _, y_elements, x_elements = cube_array.shape

    OCTA_reconstructed = cube.project().as_nparray()

    # Dividimos en sectores, filtramos las capas de la imagen consideradas como ruido y
    # de las capas restantes nos quedamos un porcentaje de profundidad
    x_step = strides[0]; x_overlap = kernel_size[0] - x_step
    y_step = strides[1]; y_overlap = kernel_size[1] - y_step
    if x_overlap == 0:
        num_steps_x = x_elements//x_step
        x_left = x_elements%x_step
    else:
        print(int((x_overlap*x_elements)/(x_elements-kernel_size[0])))
        num_steps_x = (x_elements - kernel_size[0])//x_step
        x_left = (x_elements - kernel_size[0])%x_step
    if y_overlap == 0:
        num_steps_y = y_elements//y_step
        y_left = y_elements%y_step
    else:
        num_steps_y = (y_elements - kernel_size[1])//y_step
        y_left = (y_elements - kernel_size[1])%y_step
    # print(x_elements, y_elements)
    # print(x_step, y_step)
    for j in range(num_steps_x+1):
        for i in range(num_steps_y+1):
            print("STARTTT:", j, i)
            y_q_init = y_step*j; y_q_end = y_q_init+kernel_size[1]
            x_q_init = x_step*i; x_q_end = x_q_init+kernel_size[0]
            # if i+1 <= x_left:
            #     x_q_init += i+1
            #     x_q_end += i+2
            # else:
            #     x_q_init += x_left
            #     x_q_end += x_left
            # if j+1 <= y_left:
            #     y_q_init += j+1
            #     y_q_end += j+2
            # else:
            #     y_q_init += y_left
            #     y_q_end += y_left
            print("X", x_q_init, x_q_end)
            print("Y", y_q_init, y_q_end)
            q = cube_array[:, y_q_init:y_q_end, x_q_init:x_q_end]
            avgs = []; stds = []; x_num = []
            for index, l in enumerate(q):
                #print(l)
                avg =  np.average(l); avgs.append(avg)
                std = np.std(l); stds.append(std)
                x_num.append(index)
            
            # window = signal.general_gaussian(51, p=0.5, sig=20)
            # filtered = signal.fftconvolve(window, stds, mode='full')
            # filtered = (np.average(stds) / np.average(filtered)) * filtered
            # print(len(filtered), len(stds))
            # filtered = np.roll(filtered, -25)
            # print(len(filtered), len(stds), len(x_num))
            # stds = filtered
            # stds = np.array(stds)
            # stds[stds < 10] = 0
            # avgs_grad2 = np.gradient(np.gradient(avgs))
            
            peaks, _ = signal.find_peaks(stds, prominence=3, distance=25, width=5)
            valleis, _ = signal.find_peaks(np.array(stds)*-1, prominence=3, distance=25, width=5)

            
            if len(peaks) > 3 or len(peaks) < 2 or len(peaks) != len(valleis)+1:
                # Si el analisis no ha detectado los picos que necesitamos
                # no hacemos el analisis, devolvemos la proyeccion tal cual
                print("Peaks", len(peaks), peaks, "| Valleis", len(valleis), valleis)
                continue
                # plt.subplot(1,2,2)
                # plt.scatter(x_num, stds)
                # plt.scatter(peaks, np.array(stds)[peaks])
                # plt.scatter(valleis, np.array(stds)[valleis])
                # plt.show()
                # Filtramos los slice que son puro ruido
                layers = [q[index] for index, val in enumerate(stds) if val > 10]
                q_recons = Cube(np.array(layers)).project().as_nparray()
            else:
                first_layers_group_1 = q[:valleis[0]]; stds2 = stds[:valleis[0]]
                q1 = Cube(first_layers_group_1).project().as_nparray()
                if smooth_lines == 2:
                    first_layers_group_2 = q[:peaks[1]]
                    q2 = Cube(first_layers_group_2).project().as_nparray()
                    x_center = round(x_elements/2); y_center = round(y_elements/2)
                    R_to_middle = math.sqrt(x_center**2 + y_center**2)
                    R_max = R_to_middle - math.sqrt((kernel_size[0]/2)**2 + (kernel_size[1]/2)**2)
                    x_q_center = round((x_q_end-x_q_init)/2) + x_q_init; y_q_center = round((y_q_end-y_q_init)/2) + y_q_init
                    R = math.sqrt((x_q_center-x_center)**2 + (y_q_center-y_center)**2)
                    print(R_to_middle, R_max, x_q_center, y_q_center, R) 
                    #q1 = np.array(q1)*1/()
                    # w2 = np.log((R_max/3)/(R+0.0001)); w2_max = np.log((R_max/3)/(0+0.0001))
                    # w1 = w2_max - w2
                    w1 = 1; w2 = 0
                    if R < R_max*0.4: 
                        w2 = 1; w1 = R/R_max
                    # if w1 < 0: w1 = 0
                    print(w1, w2)
                    q_recons = np.around(((w1*q1)+(w2*q2))/(w1+w2))
                    assert np.max(q_recons) <= 255 and np.min(q_recons) >= 0
                else:
                     q_recons = q1
                # if x_q_init == 150 and y_q_init == 150:
                #     plt.subplot(1,2,2)
                #     plt.scatter(x_num, stds)
                #     plt.scatter(peaks, np.array(stds)[peaks])
                #     plt.scatter(valleis, np.array(stds)[valleis])
                #     plt.show()
                #     plt.imshow(q_recons)
                #     plt.show()
                if len(peaks) == 3 and len(valleis) == 2:
                    # Si entramos aqui estamos en la excavacion del nervio optico
                    second_layers_group = q[valleis[1]:peaks[2]]; stds2 += stds[valleis[1]:]
                    if smooth_lines == 2:
                        second_layers_group = q[peaks[1]:peaks[2]]
                    q2_recons = Cube(second_layers_group).project().as_nparray()
                    #layers = [q[index] for index, val in enumerate(stds) if val > 10]
                    #layers = np.concatenate((first_layers_group,second_layers_group), axis=0)
                    #q_recons = Cube(layers).project().as_nparray()
                    q_recons = np.around((q_recons+q2_recons)/2)  
                # layers = [layers[index] for index, val in enumerate(stds2) if val > 10]
            print(i, j)
            last_q = OCTA_reconstructed[y_q_init:y_q_end, x_q_init:x_q_end]    
            OCTA_reconstructed[y_q_init:y_q_end, x_q_init:x_q_end] = np.around((q_recons+last_q)/2)

    if claheLimit is not None:
        clahe = cv2.createCLAHE(clipLimit=claheLimit)
        OCTA_reconstructed = clahe.apply(np.uint16(OCTA_reconstructed))

    return OCTA_reconstructed

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def get_mins(array) -> tuple:
    mins = []; locations = []
    for i, elem in enumerate(array):
        if i+1 < len(array) and elem < array[i-1] and elem < array[i+1]:
            mins.append(elem); locations.append(i)

    return locations, mins

def get_maxims(array) -> tuple:
    maxims = []; locations = []
    for i, elem in enumerate(array):
        if i+1 < len(array) and elem > array[i-1] and elem > array[i+1]:
            maxims.append(elem); locations.append(i)

    return locations, maxims

def reconstruct_ONH_OCTA(cube:Cube, kernel_size=(16,16), strides=(1,1), 
                            smooth_lines:int=1, claheLimit:float=10, avg_with_projection:bool=True):
    onh_recons = reconstruct_OCTA(
        cube, kernel_size=kernel_size, strides=strides,
        smooth_lines=smooth_lines, claheLimit=claheLimit
    )

    return onh_recons


def reconstruct_Angiography_OCTA(cube:Cube, kernel_size=(16,16), max_thickness_perc:float=0.1,strides=(1,1), 
                            signal_threshold:float=0.20, smooth_lines:int=1, claheLimit:float=10):
    ang_recons = reconstruct_OCTA(
        cube, kernel_size=kernel_size, strides=strides,
        smooth_lines=smooth_lines, claheLimit=claheLimit
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