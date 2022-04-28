
import math

import cv2
import tqdm
from scipy import signal
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
                max_slice_vals[y][x] = pixel_max
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

def reconstruct_OCTA(cube:Cube, kernel_size=(2,2), strides=(1,1),
                        bit_depth=16, central_depth:float=None, show_progress:bool=True):
    """
    TODO:
    - Adaptar el kernel y strides si no coincide justo con proporciones de la imagen
    - Realizar ultimo filtrado que disminuya transiciones entre proyecciones de distintas regiones
    """
    assert kernel_size[0] >= strides[0] and kernel_size[1] >= strides[1]
    cube_array = norm_volume(cube.value, bit_depth=bit_depth, max_value=1)
    print(np.min(cube_array),np.max(cube_array))
    assert np.min(cube_array) >= 0 and np.max(cube_array) <= 1
    _, y_elements, x_elements = cube_array.shape

    OCTA_reconstructed = Cube(cube_array).project().as_nparray()

    # Dividimos en sectores, filtramos las capas de la imagen consideradas como ruido y
    # de las capas restantes nos quedamos un porcentaje de profundidad
    x_step = strides[0]; x_overlap = kernel_size[0] - x_step
    y_step = strides[1]; y_overlap = kernel_size[1] - y_step
    # Distancia en Polares al centro de la imagen
    x_center = round(x_elements/2); y_center = round(y_elements/2)
    R_to_middle = math.sqrt(x_center**2 + y_center**2)
    R_max = R_to_middle - math.sqrt((kernel_size[0]/2)**2 + (kernel_size[1]/2)**2)
    
    if x_overlap == 0:
        num_steps_x = x_elements//x_step
        x_left = x_elements%x_step
    else:
        num_steps_x = (x_elements - kernel_size[0])//x_step
        x_left = (x_elements - kernel_size[0])%x_step
    if y_overlap == 0:
        num_steps_y = y_elements//y_step
        y_left = y_elements%y_step
    else:
        num_steps_y = (y_elements - kernel_size[1])//y_step
        y_left = (y_elements - kernel_size[1])%y_step
        
    if show_progress:
        pbar = tqdm .tqdm(total=(num_steps_x+1)*(num_steps_y+1), desc="Reconstructing OCTA", unit=" conv")
    for j in range(num_steps_x+1):
        for i in range(num_steps_y+1):
            x_q_init = x_step*i; x_q_end = x_q_init+kernel_size[0]
            y_q_init = y_step*j; y_q_end = y_q_init+kernel_size[1]
            
            def _reconstruct_quadrant(cube_array, x_q_init, x_q_end, y_q_init, y_q_end, timeout=5):
                if timeout == 0: return None, x_q_init, x_q_end, y_q_init, y_q_end
                q = cube_array[:, y_q_init:y_q_end, x_q_init:x_q_end]
                avgs = []; stds = []; x_num = []
                for index, l in enumerate(q):
                    avg =  np.average(l); avgs.append(avg)
                    std = np.std(l); stds.append(std)
                    x_num.append(index)


                def butter_lowpass(cutoff, fs=None, order=5):
                    nyq = 0.5 * fs
                    normal_cutoff = cutoff / nyq
                    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
                    return b, a

                def butter_lowpass_filter(data, cutoff, fs=None, order=5):
                    b, a = butter_lowpass(cutoff, fs=fs, order=order)
                    y = signal.lfilter(b, a, data)
                    return y


                # Setting standard filter requirements.
                order = 6
                fs = 30.0 # 400      
                cutoff = 3.667  

                # b, a = butter_lowpass(cutoff,fs=fs, order=6)

                # # Plotting the frequency response.
                # w, h = signal.freqz(b, a, worN=8000)
                # plt.subplot(2, 1, 1)
                # plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
                # plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
                # plt.axvline(cutoff, color='k')
                # plt.xlim(0, 0.5*fs)
                # plt.title("Lowpass Filter Frequency Response")
                # plt.xlabel('Frequency [Hz]')
                # plt.grid()
                

                # Filtering and plotting
                y = butter_lowpass_filter(stds, cutoff, fs, order)

                # plt.subplot(2, 1, 2)
                # plt.plot(x_num, stds, 'b-', label='data')
                # plt.plot(x_num, y, 'g-', linewidth=2, label='filtered data')
                # plt.xlabel('Time [sec]')
                # plt.grid()
                # plt.legend()

                # plt.subplots_adjust(hspace=0.35)
                # plt.show()
                
                stds = y
                
                prominence = 0.01 # Altura del pico hasta el primer minimo por la izq o derecha
                peaks, _ = signal.find_peaks(stds, prominence=prominence, distance=25, width=2) #prominence=prominence, distance=25, width=3, height=0.07)
                valleys, _ = signal.find_peaks(np.array(stds)*-1, prominence=prominence, distance=25, width=2) # prominence=0.006, distance=25, width=0)
                # signal.find_peaks_cwt
                
                coherence = False
                if (len(peaks) == 3 or len(peaks) == 2) and len(peaks) == len(valleys)+1:
                    for i, m in enumerate(valleys):
                        if not (m > peaks[i] and m < peaks[i+1]):
                            break
                    else:
                        coherence = True
                
                if not coherence:
                    # Si el analisis no ha detectado los picos que necesitamos, no hacemos el analisis
                    # TODO: Ver como procesar estos cubos de los que no se ha podido extraer bien la señal
                    print(peaks, valleys, q.shape, x_q_init, x_q_end, y_q_init, y_q_end)
                    # plt.subplot(1,2,1)
                    # plt.scatter(x_num, avgs)
                    # plt.subplot(1,2,2)
                    # plt.scatter(x_num, stds)
                    # plt.scatter(peaks, np.array(stds)[peaks])
                    # plt.scatter(valleys, np.array(stds)[valleys])
                    # plt.show()
                    inc_perc = 0.2
                    x_q_init = round(x_q_init-(inc_perc*kernel_size[0])); x_q_end = round(x_q_end+(inc_perc*kernel_size[0]))
                    y_q_init = round(y_q_init-(inc_perc*kernel_size[1])); y_q_end = round(y_q_end+(inc_perc*kernel_size[1]))
                    if x_q_init < 0: x_q_init = 0
                    if y_q_init < 0: y_q_init = 0
                    if x_q_end > x_elements: x_q_end = x_elements
                    if y_q_end > y_elements: y_q_end = y_elements
                    q_recons, x_q_init, x_q_end, y_q_init, y_q_end = _reconstruct_quadrant(
                        cube_array, x_q_init, x_q_end, y_q_init, y_q_end, timeout=timeout-1
                    )
                    if q_recons is not None:
                        print("Fixed: ", peaks, valleys, q.shape, x_q_init, x_q_end, y_q_init, y_q_end)
                else:
                    first_layers_group_1 = q[:valleys[0]]
                    # Filtramos ruido 
                    # first_layers_group_1 = [l for i,l in enumerate(first_layers_group_1) if stds[i] > 0.2*np.max(stds)]
                    # first_layers_group_1 = np.array(first_layers_group_1)
                    q1 = Cube(first_layers_group_1).project().as_nparray()
                    
                    if central_depth is not None:
                        first_layers_group_2 = q[:peaks[1]]
                        q2 = Cube(first_layers_group_2).project().as_nparray()
                        x_q_center = round((x_q_end-x_q_init)/2) + x_q_init; y_q_center = round((y_q_end-y_q_init)/2) + y_q_init
                        R = math.sqrt((x_q_center-x_center)**2 + (y_q_center-y_center)**2)

                        w1 = 1; w2 = 0
                        if R < R_max*central_depth: 
                            w2 = 1; w1 = R/R_max
                        
                        q_recons = ((w1*q1)+(w2*q2))/(w1+w2)
                    else:
                        q_recons = q1
                    
                    if len(peaks) == 3 and len(valleys) == 2:
                        # Si entramos aqui probablemente estamos en la excavacion del nervio optico
                        second_layers_group = q[valleys[1]:peaks[2]]
                        q2_recons = Cube(second_layers_group).project().as_nparray()
                        q_recons = Cube(np.array([q_recons, q2_recons])).project().as_nparray()
                    
                return q_recons, x_q_init, x_q_end, y_q_init, y_q_end
                
            q_recons, x_q_init, x_q_end, y_q_init, y_q_end = _reconstruct_quadrant(
                cube_array, x_q_init, x_q_end, y_q_init, y_q_end, timeout=10
            ) 
            if q_recons is not None:
                last_q = OCTA_reconstructed[y_q_init:y_q_end, x_q_init:x_q_end]    
                OCTA_reconstructed[y_q_init:y_q_end, x_q_init:x_q_end] = (q_recons+last_q)/2
            if show_progress: pbar.update(1)

    max_val = math.pow(2, bit_depth) - 1
    OCTA_reconstructed = norm_volume(OCTA_reconstructed, bit_depth=None, max_value=max_val, np_type=np.uint16)

    return OCTA_reconstructed

def norm_volume(volume, bit_depth:int=None, max_value=1, np_type=None):
    """Normalize volume between 0 and max_value"""
    if bit_depth is None:
        maxim = 1
    else:
        maxim = math.pow(2, bit_depth) - 1
    norm_v = ((volume / maxim)*max_value)
    if np_type is not None:
        norm_v = norm_v.astype(np_type)
    
    return norm_v

def get_mins(array) -> tuple:
    mins = []; locations = []
    for i, elem in enumerate(array):
        if i+1 < len(array) and elem < array[i-1] and elem < array[i+1]:
            mins.append(elem); locations.append(i)

    return locations, mins

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

class RawProcessingError(Exception):
    pass

def process_oct(raw_path:str, width_pixels:int, height_pixels:int, num_images:int=1,
                    vertical_flip:bool=True, resize:tuple[int, int]=None, reverse:bool=True) -> Cube:
    """ Returns Numpy array.

        -> reads cube with bit_depth=16, mode='unsigned'
        -> Volume values will be between 0 and 65535
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

    if reverse: cube_data = np.flip(cube_data, axis=1)

    return Cube(cube_data)