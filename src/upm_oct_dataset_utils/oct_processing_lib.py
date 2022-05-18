
import math
from copy import deepcopy
from functools import cmp_to_key
import concurrent.futures as conc

import cv2
import tqdm
from scipy import signal
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage.segmentation as seg 
import skimage.color as color

from .visualization_lib import show_image

# --------------- RAW PROCESSING ---------------
# ----------------------------------------------
class Cube():
    def __init__(self, np_array) -> None:
        np_arr_cp = deepcopy(np_array)
        if type(np_arr_cp) is list:
            np_arr_cp = np.array(np_arr_cp)
        self.value = np_arr_cp

    def as_nparray(self):
        return self.value

    def rotate_face(self, axis:int=1):
        """ axes = 'x, 'y' and 'z' | cube shape assumed = (z, y, x)
            -> Rotates the face of the cube 90 degrees over the axes selected
        """
        rotated_cube = []
        
        for i in range(self.value.shape[axis]):
            if axis == 0:
                rotated_cube.append(self.value[i, :, :])
            elif axis == 1:
                rotated_cube.append(self.value[:, i, :])
            elif axis == 2:
                rotated_cube.append(self.value[:, :, i])
                
        c = Cube(rotated_cube)

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
    """
    assert kernel_size[0] >= strides[0] and kernel_size[1] >= strides[1]
    cube_array = norm_volume(cube.value, bit_depth=bit_depth, max_value=1)
    assert np.min(cube_array) >= 0 and np.max(cube_array) <= 1
    print(np.max(cube_array))
    cube_array[cube_array < 0.50] = 0
    from .visualization_lib import animate_volume
    animate_volume(cube_array, colorbar=True)
            
    _, y_elements, x_elements = cube_array.shape        
            
    OCTA_reconstructed = np.zeros(cube_array.shape[1:])# Cube(cube_array).project().as_nparray()

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
                x_q_size = x_q_end-x_q_init; y_q_size = y_q_end-y_q_init
                x_q_half = int((x_q_size)/2); y_q_half = int((y_q_size)/2)
                
                q = cube_array[:, y_q_init:y_q_end, x_q_init:x_q_end]
                avgs = []; stds = []; x_num = []
                for index, l in enumerate(q):
                    avg =  np.average(l); avgs.append(avg)
                    std = np.std(l); stds.append(std)
                    x_num.append(index)      
                    
                
                # Filtramos ruido y variaciones bruscas iniciales
                stds =  butter_lowpass_filter(stds, cutoff=3.667, fs=30, order=6)
                
                # plt.subplot(1,2,1)
                # plt.scatter(x_num, avgs)
                # plt.subplot(1,2,2)
                # plt.scatter(x_num, stds)
                # plt.show()
 
                prominence = 0.01 # Altura del pico hasta el primer minimo por la izq o derecha
                peaks, _ = signal.find_peaks(stds, prominence=prominence, distance=25, width=2) #prominence=prominence, distance=25, width=3, height=0.07)
                valleys, _ = signal.find_peaks(np.array(stds)*-1, prominence=prominence, distance=25, width=2) # prominence=0.006, distance=25, width=0)
                
                coherence = False
                if (len(peaks) == 3 or len(peaks) == 2) and len(peaks) == len(valleys)+1:
                    for i, m in enumerate(valleys):
                        if not (m > peaks[i] and m < peaks[i+1]):
                            break
                    else:
                        coherence = True
                
                if not coherence:
                    # Si el analisis no ha detectado los picos que necesitamos, no hacemos el analisis
                    # Cambiamos el kernel y el tamaño del cuadrante y lo volvemos a intentar (corregimos)
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
                else:
                    first_layers_group_1 = q[:valleys[0]]
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
            
            qs_recons = _reconstruct_quadrant(
                cube_array, x_q_init, x_q_end, y_q_init, y_q_end, timeout=10
            )
            q_recons, x_q_init, x_q_end, y_q_init, y_q_end = qs_recons
            if q_recons is not None:
                last_q = OCTA_reconstructed[y_q_init:y_q_end, x_q_init:x_q_end]    
                OCTA_reconstructed[y_q_init:y_q_end, x_q_init:x_q_end] = (q_recons+last_q)/2
            else:
                print(f"WARNING: quadrant x={x_q_init}:{x_q_end} y={y_q_init}:{y_q_end} could not be processed neither auto-fixe")
            if show_progress: pbar.update(1)
            
    max_val = math.pow(2, bit_depth) - 1
    OCTA_reconstructed = norm_volume(OCTA_reconstructed, bit_depth=None, max_value=max_val, np_type=np.uint16)

    return OCTA_reconstructed

def segment_vascular_layer_old(octa_volume, oct_volume, scale=30, min_size=3000, channel_axis=None, sigma=5):
    seg_volume = []; masks = []
    oct_volume = oct_volume/65535
    # Suavizamos la imagen
    pbar = tqdm.tqdm(total=oct_volume.shape[0], desc="Segmenting volume", unit=" conv")
    for img in oct_volume:
        kernel = np.ones((7,7),np.float32)/49
        img = img*cv2.filter2D(img,-1,kernel)
        #print(img.shape)
        image_felzenszwalb = seg.felzenszwalb(
            img, scale=scale, min_size=min_size, channel_axis=channel_axis, sigma=sigma
        )
        # cmask = seg.chan_vese(img, mu=4, lambda1=10, lambda2=10, tol=1e-3,
        #         max_num_iter=50, dt=0.1, init_level_set="checkerboard"
        # )
        unique_regions = np.unique(image_felzenszwalb).size
        # print("Unique Regions:", unique_regions)
        label0 = unique_regions
        image_felzenszwalb[image_felzenszwalb == 0] = label0

        counts = []
        # Filtered Image
        for i in range(unique_regions):
            c = np.count_nonzero(image_felzenszwalb == i+1)
            counts.append((c,i+1))

        def compare(tp1, tp2):
            c1 = tp1[0]; c2 = tp2[0]
            if c1 > c2: return -1
            elif c1 < c2: return 1
            else: return 0

        s = sorted(counts, key=cmp_to_key(compare))
        # print(s)
        labels = s[:3] # Cogemos las 3 mas grandes
        for _, label in labels:
            if image_felzenszwalb[0][0] == label or image_felzenszwalb[img.shape[0]-1][img.shape[1]-1]==label:
                continue
            break # La ultima que se cargue sera la label seleccionada
        image_felzenszwalb[image_felzenszwalb != label] = 0
        image_felzenszwalb[image_felzenszwalb == label] = 1
        mask = image_felzenszwalb
        masks.append(mask)
        pbar.update(1)
    masks = np.array(masks)
    half_window = 30
    for i, (mask, octa_img) in enumerate(zip(masks, octa_volume)):
        start = i - half_window if i >= half_window else 0
        end = i+half_window if i < masks.shape[0]-half_window else masks.shape[0]
        mask = np.average(masks[start:end], axis=0)
        mask[mask < 0.5] = 0; mask[mask >= 0.5] = 1
        segmented = octa_img*mask
        # show_image([mask, segmented], cmap='gray', multi=True)
        seg_volume.append(segmented)

    seg_volume = np.array(seg_volume, dtype=np.uint16)

    return seg_volume

def segment_vascular_layer(octa_volume, oct_volume, num_threads:int=1, seg_window=10, smooth_kernel_size:int=7):
    #if seg_window % 2 != 0: raise ValueError("seg_window must be an even number")
    if oct_volume.shape[0] % num_threads != 0:
        raise ValueError("Division between num slices and num_threads must be an even number")
    
    oct_volume = oct_volume/65535
    vol_slices = octa_volume.shape[0]
    slcs_per_interval = math.ceil(vol_slices/num_threads)
    
    threads_slices = []
    for i in range(num_threads):
        start = i*slcs_per_interval
        end = (i+1)*slcs_per_interval
        threads_slices.append((octa_volume[start:end], oct_volume[start:end]))
      
    masks = []  
    if num_threads > 1:
        with conc.ThreadPoolExecutor() as executor:
            threads = []
            for octa_slices, oct_slices in threads_slices:
                thread = executor.submit(
                    __get_masks, 
                    oct_slices=oct_slices, smooth_kernel_size=smooth_kernel_size
                )
                threads.append(thread)
            #pbar = tqdm.tqdm(total=oct_slices.shape[0], desc="Segmenting volume", unit=" jobs")
            for thr in threads:
                slc_masks = thr.result()
                print(np.array(slc_masks).shape)
                masks += slc_masks
                print(np.array(masks).shape)
    else:
        slc_masks = __get_masks(threads_slices[0][1], smooth_kernel_size=smooth_kernel_size)
        masks = slc_masks
    
    masks = np.array(masks)
    seg_volume = []
    half_window = 10
    for i, (mask, octa_img) in enumerate(zip(masks, octa_volume)):
        start = i - half_window if i >= half_window else 0
        end = i+half_window if i < masks.shape[0]-half_window else masks.shape[0]
        mask = np.average(masks[start:end], axis=0)
        # masks[i] = mask
        mask[mask < 0.5] = 0; mask[mask >= 0.5] = 1
        segmented = octa_img*mask
        # show_image([mask, segmented], cmap='gray', multi=True)
        seg_volume.append(segmented)
    seg_volume = np.array(seg_volume)
    #print(seg_volume.shape)
    assert seg_volume.shape[0] == octa_volume.shape[0]
    
    return seg_volume

def __get_masks(oct_slices, smooth_kernel_size):
    masks = []
    pbar = tqdm.tqdm(total=oct_slices.shape[0], desc="Segmenting volume", unit=" conv")
    for img in oct_slices:
        
        if smooth_kernel_size is not None:
            k_size = smooth_kernel_size
            kernel = np.ones((k_size,k_size),np.float32)/(k_size*k_size)
            img = cv2.filter2D(img,-1,kernel)
        
        mask = seg.chan_vese(
            img, mu=3, lambda1=90, lambda2=90, tol=1e-3,
            max_num_iter=40, dt=1, init_level_set="checkerboard",
            extended_output=False
        )

        ref_point = mask[mask.shape[0]-1][mask.shape[1]-1]
        if ref_point == 1:
            mask = np.invert(mask)


        image_felzenszwalb = seg.felzenszwalb(
            mask*img, scale=1, min_size=1500, channel_axis=None, sigma=1
        )
        unique_regions = np.unique(image_felzenszwalb).size
        label0 = unique_regions
        image_felzenszwalb[image_felzenszwalb == 0] = label0
        image_felzenszwalb[image_felzenszwalb == 1] = label0+1

        #cp_fel = deepcopy(image_felzenszwalb)
        
        labels_to_use = []
        start_line_y = image_felzenszwalb[:,0]; y_start_val = image_felzenszwalb[0,0]
        for y, val in enumerate(start_line_y):
            if val != y_start_val: # Primera segmentacion por la izquierda
                labels_to_use.append(val)
                break    
        end_line_y = image_felzenszwalb[:,image_felzenszwalb.shape[1]-1]
        y_start_val = image_felzenszwalb[0,image_felzenszwalb.shape[1]-1]
        for y, val in enumerate(end_line_y):
            if val != y_start_val: # Primera segmentacion por la derecha
                if val not in labels_to_use:
                    labels_to_use.append(val)
                break
            
        for label in labels_to_use:
            image_felzenszwalb[image_felzenszwalb == label] = 1  
        image_felzenszwalb[image_felzenszwalb != 1] = 0
        
        final_mask = image_felzenszwalb
        # show_image([mask, img, final_mask], multi=True, subplot_size=(1,3), cmap='gray'); plt.show()
        masks.append(final_mask)
        pbar.update(1)
        
    return masks

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

def butter_lowpass(cutoff, fs=None, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs=None, order=5):
    b, a = butter_lowpass(cutoff, fs=fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def show_lowfilter_response(x, y, order=6, fs=30.0, cutoff=3.667):
    # Setting standard filter requirements.
    b, a = butter_lowpass(cutoff,fs=fs, order=6)

    # Plotting the frequency response.
    w, h = signal.freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    
    #Filtering and plotting
    y_filtered = butter_lowpass_filter(y, cutoff, fs, order)

    plt.subplot(2, 1, 2)
    plt.plot(x, y, 'b-', label='data')
    plt.plot(x, y_filtered, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=0.35)
    plt.show()

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

def process_oct(raw_path:str, width_pixels:int, height_pixels:int, num_images:int=1, horizontal_flip:bool=True,
                    vertical_flip:bool=False, resize:tuple[int, int]=None, reverse:bool=True) -> Cube:
    """ Returns Cube Object.
        --> horizontal_flip and reverse options are True by default due to that cirrus volumes are
        saved backwards

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

    cube_data = np.array(cube_data).astype(np.uint16)

    if reverse: cube_data = np.flip(cube_data, axis=1)
    
    cube = Cube(cube_data)
    if horizontal_flip: cube = cube.hflip_slices()

    return cube



# Reconstruct OCTA Code for better reconstructing not good quadrant signals
# if not not_phase_1 and not test:
#     print("Phase 1", x_q_init, x_q_end, y_q_init, y_q_end, timeout)
#     xfactors = [0,1,1,1,0,-1,-1,-1]; yfactors = [1,1,0,-1,-1,-1,0,1]
#     reconstructed_qs = np.array([[False, False], [False, False]]); quadrants = []
#     for i in range(8):
#         xf = xfactors[i]; yf = yfactors[i]
#         x_start = (x_q_init+(x_q_half*xf)); x_end = (x_q_end+((x_q_size-x_q_half)*xf))
#         y_start = (y_q_init+(y_q_half*yf)); y_end = (y_q_end+((y_q_size-y_q_half)*yf))
#         if x_start < 0: x_start = 0
#         if y_start < 0: y_start = 0
#         if x_end > x_elements: x_end = x_elements
#         if y_end > y_elements: y_end = y_elements
#         sub_q = cube_array[:, y_start:y_end, x_start:x_end]
#         print("Start",x_start, x_end, y_start, y_end,timeout)
#         q_tuple = _reconstruct_quadrant(
#             sub_q, x_start, x_end, y_start, y_end, timeout=timeout-1, not_phase_1=True
#         )
#         q_recons = q_tuple[0]
#         if q_recons is not None:
#             quadrants.append(q_tuple)
#             corner = xf*yf
#             if corner == 0:
#                 if xf == 0:
#                     if yf == 1: reconstructed_qs[0] = [True, True]
#                     elif yf == -1: reconstructed_qs[1] = [True, True]
#                 elif yf == 0:
#                     if xf == 1: reconstructed_qs[:, 1] = True
#                     elif yf == -1: reconstructed_qs[:, 0] = True
#             elif abs(corner) == 1:
#                 if xf > 0: xf = 1
#                 elif xf < 0: xf = 0
#                 if yf > 0: yf = 0
#                 elif yf < 0: yf = 1
#                 reconstructed_qs[yf][xf] = True
#             else: raise Exception("Error")
#     for rq in reconstructed_qs.flatten():
#         if not rq:
#             break
#     else:
#         q_recons = quadrants