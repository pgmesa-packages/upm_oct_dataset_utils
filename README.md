# UPM package for managing the OCT study dataset

Package that offers some functionalities to easily work, create and organize a dataset for an study that works with raw (.img) OCT/OCTA volumes and XML scans analysis, exported from the Cirrus Zeiss 5000 with the Zeiss research license. This code in specific for the UPM multiple sclerosis, NMO and RIS study (2021-2022), but it can serve as a base to fit a great range of other necessities related with the topic

### Instalation
```
pip install upm_oct_dataset_utils
```
### Modules 

#### dataset_classes
Classes that represent a layer of abstraction to easily query the file system tree of the dataset where the images are stored (hard disk, computer ...)
Arquitecure that the tree directory must follow in the raw dataset:
```
- dataset_path
    (groups)
    - CONTROL
        (patients)
        - patient-1
            - IMG
                - PCZMI... .img (exported with Zeiss research licence)
                ...
            - retinography
                - O(S/D)_adqu-date_retinography.jpg
            - XML
                - CZMI... .xml
        - patient-2
            ...
        - ...
    - MS
        ...
    - NMO
        ...
    - RIS
        ...
```

#### oct_processing_lib
To process, read and easily work with the raw (.img) images from Cirrus Zeiss 5000. It also offers some functions to manage and reconstruct OCTA volumes.
The file system tree of the dataset should be as follows 
```
# Reconstruct OCTA from volume and get 2D projected numpy array
data = process_cube(
    data_path, 'OCTA', 'macula', 
    resize=(int(width_scale_factor*1024), 1024) if resize else None
).rotate_face(axe='x').resize_slices((350,350)).project().as_nparray()
```

#### visualization_lib
To visualize OCTA reconstructions and OCT/OCTA volumes with the possibility to animate the volume as a short movie.
The functions admits arrays of images to show and arrays of different volumes to animate at the same time (when multi option is set to True)
```
def show_image(image, title:list[str]=None, subplot_size:tuple[int,int]=None, 
                    cmap:str='jet', colorbar:bool=None, multi:bool=False, show:bool=True):
    ...

def animate_volume(volume, figure=None, title:list[str]=None, subplot_size:tuple[int,int]=None, 
                    cmap:str='jet', colorbar:bool=None, multi:bool=False, t_milisec:int=4000, repeat=True):
    ...
```

#### xml_processing_lib
To process and read 1 or more XML analysis from the Cirrus Zeiss 5000
```
# Remove trash oct info and returns a clean XML with the useful data (removes "TRACKINGDETAILS" field and other minor stuff)
processed_xml:dict = process_xmlscans(xml_path, xml_scans)
```

### Usage Example
```
# Create a query to get all optic-disk OCTA images of left eye of the control group patients from 1 to 7 
raw_dataset = RawDataset(raw_dataset_path)
raw_data_paths:dict = raw_dataset.get_data_paths(
    group='control', patient_num=[1,2,3,4,5,6,7], data_type='OCTA', zone="optic-disk", eye="left"
)
...
# Process all raw images exported with Zeiss research License and save them in standard format in the clean dataset
clean_dataset = CleanDataset(clean_dataset_path)
raw_data_paths:dict = raw_dataset.get_data_paths()
for group, patients_data in raw_data_paths.items():
    for patient, dtypes in patients_data.items()
        for data_type, zones_data in dtypes.items():
            clean_path = clean_dataset.get_dir_path(group=grp, patient_num=p_num, data_type=dtype)
            if not os.path.exists(clean_path): os.mkdir(clean_path)
            for zone, eye_data in zonde_data.items():
                for eye, data_path in eye_data.items():
                    data = process_cube(
                        data_path, data_type, zone, 
                        resize=(int(width_scale_factor*1024), 1024) if resize else None
                    ).rotate_face(axe='x').resize_slices((350,350)).project().as_nparray()
                    data_path = Path(data_path)
                    raw_file_info = raw_dataset.split_file_name(data_path.name, dtype)
                    adq_date = raw_file_info['adquisition_date']
                    adq_name = zone_info['adquisitions_name']
                    file_name = patient+"_"+adq_name[data_type]+"_"+adq_date+"_"+eye_conv+'.tif'
                    file_path = clean_path/file_name
                    tiff.imwrite(file_path, data)

```