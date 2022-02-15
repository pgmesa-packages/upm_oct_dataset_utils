# UPM package for managing the OCT/OCTA study dataset

Package that offers some functionalities to easily work, create, manage and organize a dataset for an study that works with raw (.img) OCT/OCTA volumes and XML scans analysis, exported from the Cirrus Zeiss 5000 with the Zeiss research license. This code in specific for the UPM multiple sclerosis, NMO and RIS study (2021-2022), but it can serve as a base to fit a great range of other necessities related with the topic. The idea is to have two different directories to store data: a raw dataset with the exported data (.img) from the device and a clean dataset where the processed data from the raw dataset will be stored. To train an IA with the clean dataset, is as simple as clean_ds.get-data_paths(-query-) to get all data paths you need and then create for example a tensorflow dataset to load the data durig the training.

### Installation
```
pip install upm_oct_dataset_utils
```
### Modules 

#### dataset_classes
Classes that represent a layer of abstraction to easily query the file system tree of the dataset where the images are stored (hard disk, computer ...)
Default arquitecure that the tree directory must follow in the raw dataset:
```
- dataset_path
    (groups)
    - control
        (patients)
        - patient-1
            - study_20-11-2021
                - IMG
                    - PCZMI... .img (exported with Zeiss research licence)
                    ...
                - retinography
                    - O(S/D)_adqu-date_retinography.jpg
                - XML
                    - CZMI... .xml
            - study_23-1-2022
                ...
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
Create a query to get all optic-disk OCTA images of left eye of the control group patients from 1 to 7 
```
raw_dataset = RawDataset(raw_dataset_path)
raw_data_paths:dict = raw_dataset.get_data_paths(
    group='control', patient_num=[1,2,3,4,5,6,7], data_type='OCTA', zone="optic-disk", eye="left"
)
```
See the state of the datasets, the adquisitions that are missing from each patient study or are pending to be exported or the studies and adquisitions that have not been processed yet (raw_dataset.show_info(-query-) or clean_dataset.show_info(-query-))
```
+ RAW DATASET INFO (Path -> 'D:\study_datasets\raw_dataset')

        - Adquisitions per patient study:
            -> 4 OCT (macular_OD, macular_OS, optic-nerve_OD, optic-nerve_OS)
            -> 4 OCTA (macular_OD, macular_OS, optic-nerve_OD, optic-nerve_OS)
            -> 2 retinographies (OD, OS)
            -> 8 scans XML analysis report

----------------------------------------------------
+ CONTROL GROUP (size=21)
- 'patient-1' (studies=1) has all adquisitions
- 'patient-2' (studies=1) has all adquisitions
- 'patient-3' (studies=1) has all adquisitions
- 'patient-4' (studies=1) has all adquisitions
...
----------------------------------------------------
----------------------------------------------------
+ MS GROUP (size=26)
- 'patient-1' (studies=1) has all adquisitions
- 'patient-2' (studies=1) has all adquisitions
- 'patient-6' (studies=1) has all adquisitions
- 'patient-7' (studies=1) has missing info:
    {
        "study_13-10-2021": {
            "OCTA": "macula left missing",
            "XML": {
                "OCTA_macula_OS": "missing"
            }
        }
    }
...
+ SUMMARY (queried-studies=26):
     -> OCT Cubes => 76/104 (73.08%) -> (28 missing)
     -> OCTA Cubes => 51/104 (49.04%) -> (53 missing)
     -> Retina Images => 24/52 (46.15%) -> (28 missing)
     -> XML scans => 127/208 (61.06%) -> (81 missing)
 -> Global data = 278/468 (59.4%) -> (190 missing)
----------------------------------------------------
----------------------------------------------------
 + NMO GROUP (size=1)
 - 'patient-1' (studies=1) has all adquisitions
 + SUMMARY:
     -> OCT Cubes => 4/4 (100.0%) -> (0 missing)
     -> OCTA Cubes => 4/4 (100.0%) -> (0 missing)
     -> Retina Images => 2/2 (100.0%) -> (0 missing)
     -> XML scans => 8/8 (100.0%) -> (0 missing)
 -> Global data = 18/18 (100.0%) -> (0 missing)
----------------------------------------------------
----------------------------------------------------
 + RIS GROUP (size=0)
     -> This group is empty
----------------------------------------------------
```
Process all raw images exported with Zeiss research License and save them in standard format in the clean dataset
```
clean_dataset = CleanDataset(clean_dataset_path)
raw_data_paths:dict = raw_dataset.get_data_paths()
for group, patients_data in raw_data_paths.items():
    for patient, study_data in patients_data.items():
        clean_dataset.create_patient(grp, patient_num=p_num)
        for study, dtypes in study_data.items():
            clean_dataset.create_study(group, patient_num, study)
            for data_type, zones_data in dtypes.items():
                clean_path = clean_dataset.get_dir_path(group=grp, patient_num=p_num, study=study, data_type=dtype)
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
Clean Dataset directory tree
```
- dataset_path
    (groups)
    - control
        (patients)
        - patient-1
            - study_20-11-2021
                - OCT
                    - patient-1_adqu-type_adqu-date_O(S/D).tiff
                - OCTA
                    ...
                - retinography
                    - patient-1_retinography_adqu-date_O(S/D).jpg
                - patient-1_analysis.json
            - study_23-1-2022
                ...
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
