
import os
import json
import copy
from typing import Union
from pathlib import Path
from functools import cmp_to_key

import pandas_read_xml as pdx
    
class DatasetAccessError(Exception):
    pass

class RawDataset():
    """
    Arquitecure that the tree directory must follow:
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
    """
    groups = {
        'control': {'dir_name': 'CONTROL'}, 
        'MS': {'dir_name': 'MS'},
        'NMO': {'dir_name': 'NMO'},
        'RIS': {'dir_name': 'RIS'},
    }
    data_types = {
        'OCT': {'parent_dir': 'IMG'}, 
        'OCTA': {'parent_dir': 'IMG'},
        'retinography': {'parent_dir': 'retinography'},
        'XML': {'parent_dir': 'XML'}
    }
    file_suffixes = {
        'OCT': 'cube_z.img',
        'OCTA': 'FlowCube_z.img',
        'retinography': '_retinography.jpg',
    }
    file_prefixes = {
        'XML': "CZMI^"
    }
    zones = {
        'macula': {
            'adquisitions_name': {
                'OCT': 'Macular Cube 512x128',
                'OCTA': 'Angiography 6x6 mm'
            }
        }, 
        'optic-nerve': {
            'adquisitions_name': {
                'OCT': 'Optic Disc Cube 200x200',
                'OCTA': 'ONH Angiography 4.5x4.5 mm'
            }
        }
    }
    eyes = {'right': 'OD', 'left': 'OS'}
    
    def __init__(self, dataset_path:str) -> None:
        if dataset_path is not type(Path):
            dataset_path = Path(dataset_path).resolve()
        self.dataset_path = dataset_path
    
    def get_dir_path(self, group:str=None, patient_num:int=None, data_type:str=None) -> Path:
        path = self.dataset_path
        if group is None: return path
        try:
            path = path/self.groups[group]['dir_name']
        except KeyError:
            raise DatasetAccessError(f"'{group}' is not a valid group -> {list(self.groups.keys())}")
        if patient_num is None: return path
        patient = f'patient-{patient_num}'
        path = path/patient
        if not os.path.isdir(path):
            raise DatasetAccessError(f"'{patient}' doesn't exist in '{group}' group")
        if data_type is None: return path
        try:
            path = path/self.data_types[data_type]['parent_dir']
        except KeyError:
            raise DatasetAccessError(f"'{data_type}' is not a valid data_type -> {list(self.data_types.keys())}")
        
        return path
    
    def split_file_name(self, file_name:str, data_type:str) -> dict:
        if data_type == 'OCT' or data_type == 'OCTA':
            headers = ['id', 'modality_info', 'adquisition_date', 'num', 'eye', 'sn', 'cube_type']
            info = file_name.split('_', maxsplit=6)
        elif data_type == 'retinography':
            headers = ['eye', 'adquisition_date', 'modality_info']
            info = file_name.split('_')
            
        return dict(zip(headers, info))
    
    def get_patients(self, group:str, as_int:bool=False) -> list:
        group_path:Path = self.get_dir_path(group=group)
        file_names = os.listdir(group_path)
        patients = []
        for name in file_names:
            if "patient-" in name:
                if as_int:
                    patients.append(int(name.split("-")[1]))
                else:
                    patients.append(name)
        
        def _compare(patient1:str, patient2:str):
            num1 = int(patient1.split("-")[1])
            num2 = int(patient2.split("-")[1])
            if num1 < num2:
                return -1
            elif num1 > num2:
                return 1
            else:
                return 0

        return sorted(patients, key=cmp_to_key(_compare))
    
    def get_data_paths(self, group:Union[str, list[str]]=None, patient_num:Union[int, list[int]]=None, 
                       data_type:Union[str, list[str]]=None, zone:str=None, eye:str=None, _withoutpaths:bool=False) -> Union[dict, Path]:
        
        def _get_dtype(grp:str, p_num:int, d_type:str) -> dict:
            data_type_info = {}
            if d_type == 'OCT' or d_type == 'OCTA':
                data_type_info = self._get_img_paths(grp, p_num, d_type, zone=zone, eye=eye, _withoutpaths=_withoutpaths)
            elif d_type == 'retinography':
                data_type_info = self._get_retinography_paths(grp, p_num, eye=eye, _withoutpaths=_withoutpaths)
            elif d_type == 'XML':
                data_type_info = self._get_xml_paths(grp, p_num, _withoutpaths=_withoutpaths)
        
            return data_type_info
        
        def _get_data_oftype(grp:str, p_num:int, d_type:Union[str, list[str]]=None) -> dict:
            data_types = {}
            if d_type is None:
                for d_type in self.data_types.keys():
                    data_types[d_type] = _get_dtype(grp, p_num, d_type)
            elif type(d_type) is list:
                for dtp in d_type:
                    data_types[dtp] = _get_dtype(grp, p_num, dtp)
            elif type(d_type) is str:
                data_types[d_type] = _get_dtype(grp, p_num, d_type)
            
            # Filtramos los que no tengan info (estan vacíos)
            if _withoutpaths:
                dict_copy = copy.deepcopy(data_types)
                for d, info in dict_copy.items():
                    if not bool(info): data_types.pop(d) 
        
            return data_types
        
        # Vemos que grupos hay que recorrer        
        data = {}     
        if group is None:
            for group in self.groups.keys():
                data[group] = {}
        elif type(group) is list:
            for grp in group:
                data[grp] = {}
        else:
            data[group] = {}
        # Recorremos los grupos
        for grp in data:
            if patient_num is None:
                for patient in self.get_patients(grp):
                    num = patient.split("-")[1]
                    try:
                        data[grp][patient] = _get_data_oftype(grp, num, d_type=data_type)
                    except DatasetAccessError:
                        pass
            elif type(patient_num) is list:
                for num in patient_num:
                   data[grp][f'patient-{num}'] = _get_data_oftype(grp, num, d_type=data_type) 
            else:
                data[grp][f'patient-{patient_num}'] = _get_data_oftype(grp, patient_num, d_type=data_type)
        # Vemos si se nos ha especificado un unico path en concreto para devolver solo ese en vez del dict entero
        if group is not None and type(patient_num) is int and type(data_type) is str:
            try:   
                if data_type == 'OCT' or data_type == 'OCTA':
                    if data_type is not None and zone is not None and eye is not None:  
                        data = data[group][f'patient-{patient_num}'][data_type][zone][eye]
                elif data_type == 'retinography':
                    if data_type is not None and zone is not None and eye is not None:  
                        data = data[group][f'patient-{patient_num}'][data_type][eye]
                elif data_type == 'XML':
                    data = data[group][f'patient-{patient_num}'][data_type]
                    if not bool(data): raise KeyError
            except KeyError:
                raise DatasetAccessError("The path/file specified doesn't exist")
        
        return data
    
    def _get_img_paths(self, group:str, patient_num:int, modality:str, zone:str=None, eye:str=None, _withoutpaths=False) -> dict:
        path = self.get_dir_path(group=group, patient_num=patient_num, data_type=modality)
        img_data = {}; data_without_paths = {}
        if os.path.isdir(path):
            for file_name in os.listdir(path):
                if not file_name.endswith(self.file_suffixes[modality]):
                    continue
                for z, zone_val in self.zones.items():
                    if zone_val['adquisitions_name'][modality] in file_name:
                        break
                else: continue
                if zone is not None and z != zone: continue
                if img_data.get(z, None) is None:
                    img_data[z] = {}; data_without_paths[z] = []
                for e, eye_val in self.eyes.items():
                    if eye_val in file_name: break
                if eye is not None and e != eye: continue
                full_path = str(path/file_name)
                img_data[z][e] = full_path
                data_without_paths[z].append(e)
        
        dict_copy = copy.deepcopy(data_without_paths)
        for z, info in dict_copy.items():
            if not bool(info): data_without_paths.pop(z)
        
        if _withoutpaths: return data_without_paths        
        return img_data
    
    def _get_retinography_paths(self, group:str, patient_num:int, eye:str=None, _withoutpaths=False) -> Union[dict, list]:
        data_type = 'retinography'
        path = self.get_dir_path(group=group, patient_num=patient_num, data_type=data_type)
        img_data = {}; eyes = []
        if os.path.isdir(path):
            for file_name in os.listdir(path):
                if not file_name.endswith(self.file_suffixes[data_type]):
                    continue
                for e, eye_val in self.eyes.items():
                    if eye_val in file_name: break
                if eye is not None and e != eye: continue
                full_path = str(path/file_name)
                img_data[e] = full_path
                eyes.append(e)
        if _withoutpaths: return eyes    
        return img_data
        
    def _get_xml_paths(self, group:str, patient_num:int, _withoutpaths=False) -> Union[dict, list]:
        data_type = 'XML'
        path = self.get_dir_path(group=group, patient_num=patient_num, data_type=data_type)
        data_path = {}; total_scans = []
        if os.path.isdir(path):
            for file_name in os.listdir(path):
                if not file_name.startswith(self.file_prefixes[data_type]):
                    continue
                full_path = str(path/file_name)
                scans = self._get_xml_info(full_path)
                total_scans += scans
                data_path[full_path] = scans
        
        if _withoutpaths: return total_scans
        return data_path
    
    def _get_xml_info(self, file_path:str) -> list:
        xml_info = []
        json_str = pdx.read_xml(file_path).to_json(indent=4)
        try:
            studies = json.loads(json_str)["ExportSchema"]["0"]["PATIENT"]["VISITS"]["STUDY"]
        except:
            raise DatasetAccessError(f"Invalid XML format -> '{file_path}'")
        if type(studies) is dict:
            studies = [studies]
        for i in range(2):
            modality = 'OCT' if i == 0 else 'OCTA'
            for zone, zone_adq in self.zones.items():
                for eye_convention  in self.eyes.values():
                    for study in studies:
                        series = study["SERIES"]
                        if series is None: continue
                        scans = series["SCAN"]
                        if type(scans) is dict:
                            scans = [scans]
                        for scan in scans:
                            adq_name = zone_adq['adquisitions_name'][modality]
                            cond1 = scan['PROTOCOL'] == adq_name and scan["SITE"] == eye_convention
                            cond2 = "ANALYSIS" in scan
                            if cond1 and cond2:
                                xml_info.append(modality+"_"+zone+"_"+eye_convention)
                                break     

        return xml_info
    
    def show_info(self, group:str=None, patient_num:Union[int, list[int]]=None,
                    only_missing_info:bool=False, data_type:list[str]=None, only_summary:bool=False):
        if data_type is not None:
            if type(data_type) is not list:
                raise DatasetAccessError('data_types parameter must be a list of strings')
            for dtype in data_type:
                if dtype not in self.data_types:
                    raise DatasetAccessError(f"'{dtype}' is not a valid data type")
        print(f"+ RAW DATASET INFO (Path -> '{self.dataset_path}')")
        raw_dataset_info = """
        - Adquisitions per patient:
            -> 4 OCT (macular_OD, macular_OS, optic-nerve_OD, optic-nerve_OS)
            -> 4 OCTA (macular_OD, macular_OS, optic-nerve_OD, optic-nerve_OS)
            -> 2 retinographies (OD, OS)
            -> 8 scans XML analysis report
        """
        print(raw_dataset_info)
        data_paths:dict = self.get_data_paths(group=group, patient_num=patient_num)
        for group, group_info in data_paths.items():
            print('----------------------------------------------------')
            msg = f" + {group.upper()} GROUP"
            if patient_num is None:
                msg += f" (size={len(self.get_patients(group))})"
            else:
                msg += F", PATIENT {patient_num}"
            print(msg)
            # Variables to count missing info
            m_oct = 0; m_octa = 0; m_ret = 0; m_xml = 0; num_patients = len(group_info)
            if num_patients == 0:
                print("     -> This group is empty")
            else:
                for patient, p_info in group_info.items():
                    missing_info = {}; has_missing_info = False
                    for dtype in self.data_types:
                        if data_type is not None and dtype not in data_type: continue
                        if dtype == 'retinography':
                            ret_info = p_info.get(dtype, None)
                            if not bool(ret_info):
                                missing_info[dtype] = 'OD and OS missing'
                                has_missing_info = True; m_ret += 2
                            else:
                                for i in range(2):
                                    eye = 'right' if i == 0 else 'left'
                                    eye_info = ret_info.get(eye, None)
                                    if not bool(eye_info):
                                        missing_info[dtype] = f'{eye} missing'
                                        has_missing_info = True; m_ret += 1
                        if dtype == 'OCT' or dtype == 'OCTA':
                            img_info = p_info.get(dtype, None)
                            if not bool(img_info):
                                missing_info[dtype] = '2x macular (OD, OS), 2x optic nerve (OD, OS)'
                                has_missing_info = True;
                                if dtype == 'OCT': m_oct += 4
                                elif dtype == 'OCTA': m_octa += 4
                            else:
                                for i in range(2):
                                    zone = 'macula' if i == 0 else 'optic-nerve'
                                    zone_info = img_info.get(zone, None)
                                    if not bool(zone_info):
                                        missing_info[dtype] = f'2x {zone} (OD, OS)'
                                        has_missing_info = True
                                        if dtype == 'OCT': m_oct += 2
                                        elif dtype == 'OCTA': m_octa += 2
                                    else:
                                        for i in range(2):
                                            eye = 'right' if i == 0 else 'left'
                                            eye_info = zone_info.get(eye, None)
                                            if not bool(eye_info):
                                                missing_info[dtype] = f'{zone} {eye} missing'
                                                has_missing_info = True
                                                if dtype == 'OCT': m_oct += 1
                                                elif dtype == 'OCTA': m_octa += 1
                        if dtype == 'XML':
                            xml_files = p_info.get(dtype, None)
                            if not bool(xml_files):
                                missing_info[dtype] = "all 8 scans analysis report are missing"
                                m_xml += 8; has_missing_info = True
                            else:
                                for i in range(2):
                                    modality = 'OCT' if i == 0 else 'OCTA'
                                    for zone in self.zones:
                                        for eye in self.eyes.values():
                                            scan_name = modality+"_"+zone+"_"+eye
                                            for scans in xml_files.values():
                                                if scan_name in scans: break
                                            else:
                                                if dtype not in missing_info:
                                                    missing_info[dtype] = {}
                                                missing_info[dtype][scan_name] = "missing"
                                                m_xml += 1; has_missing_info = True
                    if not only_summary:
                        if not has_missing_info:
                            if not only_missing_info: 
                                msg = f" - '{patient}' has all adquisitions" 
                                msg += "" if data_type is None else f" of type {data_type}"
                                print(msg)
                        else:
                            print(f" - '{patient}' has missing info:")
                            str_missing_info = json.dumps(missing_info, indent=4)
                            tab = "     "; str_missing_info = str_missing_info.replace('\n', '\n'+tab)
                            print(tab+str_missing_info)
                if data_type is None:
                    summary_dtypes = list(self.data_types.keys())
                else:
                    summary_dtypes = data_type
                # Summary
                print(f" + SUMMARY:")
                # OCT
                total_octs = 0
                if 'OCT' in summary_dtypes:
                    total_octs = num_patients*4
                    oct_perc =  round((total_octs-m_oct)*100/total_octs, 2)
                    print(f'     -> OCT Cubes => {total_octs-m_oct}/{total_octs} ({oct_perc}%) -> ({m_oct} missing)')
                # OCTA
                total_octas = 0;
                if 'OCTA' in summary_dtypes:
                    total_octas = num_patients*4
                    octa_perc =  round((total_octas-m_octa)*100/total_octas, 2)
                    print(f'     -> OCTA Cubes => {total_octas-m_octa}/{total_octas} ({octa_perc}%) -> ({m_octa} missing)')
                # Retinographies
                total_retinos = 0;
                if 'retinography' in summary_dtypes:
                    total_retinos = num_patients*2;
                    ret_perc =  round((total_retinos-m_ret)*100/total_retinos, 2)
                    print(f'     -> Retina Images => {total_retinos-m_ret}/{total_retinos} ({ret_perc}%) -> ({m_ret} missing)')
                # XML scans analysis
                total_xml = 0;
                if 'XML' in summary_dtypes:
                    total_xml = num_patients*8
                    xml_perc =  round((total_xml-m_xml)*100/total_xml, 2)
                    print(f'     -> XML scans => {total_xml-m_xml}/{total_xml} ({xml_perc}%) -> ({m_xml} missing)')
                # Global 
                total = total_octs + total_octas + total_retinos + total_xml
                total_missing = m_oct+m_octa+m_ret+m_xml; percentage = round((total-total_missing)*100/total, 2)
                print(f' -> Global data = {total-total_missing}/{total} ({percentage}%) -> ({total_missing} missing)')
            print('----------------------------------------------------')
            
class CleanDataset():
    """
    Arquitecure that the tree directory must follow:
        - dataset_path
            (groups)
            - CONTROL
                (patients)
                - patient-1
                    - OCT
                        - patient-1_adqu-type_adqu-date_O(S/D).tiff
                    - OCTA
                        ...
                    - retinography
                        - patient-1_retinography_adqu-date_O(S/D).jpg
                    - patient-1_analysis.json
                - patient-2
                    ...
                - ...
            - MS
                ...
            - NMO
                ...
            - RIS
                ...
    """
    
    groups = {
        'control': {'dir_name': 'CONTROL'}, 
        'MS': {'dir_name': 'MS'},
        'NMO': {'dir_name': 'NMO'},
        'RIS': {'dir_name': 'RIS'},
    }
    data_types = {
        'OCT': {'parent_dir': 'OCT'}, 
        'OCTA': {'parent_dir': 'OCTA'},
        'retinography': {'parent_dir': 'retinography'},
        'XML': {'parent_dir': ''}
    }
    
    file_suffixes = {
        'XML': "analysis.json"
    }
    
    zones = {
        'macula': {
            'adquisitions_name': {
                'OCT': 'Macular Cube 512x128',
                'OCTA': 'Angiography 6x6 mm'
            }
        }, 
        'optic-nerve': {
            'adquisitions_name': {
                'OCT': 'Optic Disc Cube 200x200',
                'OCTA': 'ONH Angiography 4.5x4.5 mm'
            }
        }
    }
    eyes = {'right': 'OD', 'left': 'OS'}
    
    def __init__(self, dataset_path:str) -> None:
        if dataset_path is not type(Path):
            dataset_path = Path(dataset_path).resolve()
        self.dataset_path = dataset_path
    
    def get_dir_path(self, group:str=None, patient_num:int=None, data_type:str=None) -> Path:
        path = self.dataset_path
        if group is None: return path
        try:
            path = path/self.groups[group]['dir_name']
        except KeyError:
            raise DatasetAccessError(f"'{group}' is not a valid group -> {list(self.groups.keys())}")
        if patient_num is None: return path
        patient = f'patient-{patient_num}'
        path = path/patient
        if not os.path.isdir(path):
            raise DatasetAccessError(f"'{patient}' doesn't exist in '{group}' group")
        if data_type is None: return path
        try:
            path = path/self.data_types[data_type]['parent_dir']
        except KeyError:
            raise DatasetAccessError(f"'{data_type}' is not a valid data_type -> {list(self.data_types.keys())}")
        
        return path
 
    def create_patient(self, group:str, patient_num:int):
        """Creates a patient directory tree in case it hasn't been created yet"""
        try:
            self.get_dir_path(group=group, patient_num=patient_num)
        except DatasetAccessError:
            patient_path = self.get_dir_path(group=group)/f'patient-{patient_num}'
            os.mkdir(patient_path)
            for dtype in self.data_types:
                dir_name = self.data_types[dtype]['parent_dir']
                if not os.path.exists(patient_path/dir_name):
                    os.mkdir(patient_path/dir_name)
        
    def get_patients(self, group:str) -> list:
        group_path:Path = self.get_dir_path(group=group)
        file_names = os.listdir(group_path)
        patients = []
        for name in file_names:
            if "patient-" in name: 
                patients.append(name)
        
        return patients

    def get_data_paths(self, group:Union[int, list[str]]=None, patient_num:Union[int, list[int]]=None, 
                       data_type:Union[str, list[str]]=None, zone:str=None, eye:str=None) -> Union[dict, Path]:
        
        def _get_dtype(grp:str, p_num:int, d_type:str) -> dict:
            data_type_info = {}
            if d_type == 'OCT' or d_type == 'OCTA':
                data_type_info = self._get_img_paths(grp, p_num, d_type, zone=zone, eye=eye)
            elif d_type == 'retinography':
                data_type_info = self._get_retinography_paths(grp, p_num, eye=eye)
            elif d_type == 'XML':
                data_type_info = self._get_analysis_path(grp, p_num)
        
            return data_type_info
        
        def _get_data_oftype(grp:str, p_num:int, d_type:Union[str, list[str]]=None) -> dict:
            data_types = {}
            if d_type is None:
                for d_type in self.data_types.keys():
                    data_types[d_type] = _get_dtype(grp, p_num, d_type)
            elif type(d_type) is list:
                for dtp in d_type:
                    data_types[dtp] = _get_dtype(grp, p_num, dtp)
            elif type(d_type) is str:
                data_types[d_type] = _get_dtype(grp, p_num, d_type)
        
            return data_types
        
        # Vemos que grupos hay que recorrer        
        data = {}     
        if group is None:
            for group in self.groups.keys():
                data[group] = {}
        elif type(group) is list:
            for grp in group:
                data[grp] = {}
        else:
            data[group] = {}
        # Recorremos los grupos
        for grp in data:
            if patient_num is None:
                for patient in self.get_patients(grp):
                    num = patient.split("-")[1]
                    data[grp][patient] = _get_data_oftype(grp, num, d_type=data_type)
            elif type(patient_num) is list:
                for num in patient_num:
                    data[grp][f'patient-{num}'] = _get_data_oftype(grp, num, d_type=data_type) 
            else:
                data[grp][f'patient-{patient_num}'] = _get_data_oftype(grp, patient_num, d_type=data_type)
        # Vemos si se nos ha especificado un unico path en concreto para devolver solo ese en vez del dict entero
        if group is not None and type(patient_num) is int and type(data_type) is str:
            try:   
                if data_type == 'OCT' or data_type == 'OCTA':
                    if data_type is not None and zone is not None and eye is not None:  
                        data = data[group][f'patient-{patient_num}'][data_type][zone][eye]
                elif data_type == 'retinography':
                    if data_type is not None and zone is not None and eye is not None:  
                        data = data[group][f'patient-{patient_num}'][data_type][eye]
                elif data_type == 'XML':
                    data = data[group][f'patient-{patient_num}'][data_type]
                    if not bool(data): raise KeyError
            except KeyError:
                raise DatasetAccessError("The path/file specified doesn't exist")
        
        return data
    
    def _get_img_paths(self, group:str, patient_num:int, modality:str, zone:str=None, eye:str=None) -> dict:
        path = self.get_dir_path(group=group, patient_num=patient_num, data_type=modality)
        img_data = {}
        if os.path.isdir(path):
            for file_name in os.listdir(path):
                for z, zone_val in self.zones.items():
                    if zone_val['adquisitions_name'][modality] in file_name:
                        break
                else: continue
                if zone is not None and z != zone: continue
                if img_data.get(z, None) is None:
                    img_data[z] = {}
                for e, eye_val in self.eyes.items():
                    if eye_val in file_name: break
                if eye is not None and e != eye: continue
                img_data[z][e] = {}
                full_path = str(path/file_name)
                img_data[z][e] = full_path
                
        return img_data
    
    def _get_retinography_paths(self, group:str, patient_num:int, eye:str=None) -> dict:
        data_type = 'retinography'
        path = self.get_dir_path(group=group, patient_num=patient_num, data_type=data_type)
        img_data = {}
        if os.path.isdir(path):
            for file_name in os.listdir(path):
                for e, eye_val in self.eyes.items():
                    if eye_val in file_name: break
                if eye is not None and e != eye: continue
                img_data[e] = {}
                full_path = str(path/file_name)
                img_data[e] = full_path
                
        return img_data
        
    def _get_analysis_path(self, group:str, patient_num:int) -> dict:
        data_type = 'XML'; name = f'patient-{patient_num}_'+self.file_suffixes[data_type]
        analysis_path = self.get_dir_path(group=group, patient_num=patient_num, data_type=data_type)/name
        analysis = {}
        if os.path.exists(analysis_path):
            analysis[str(analysis_path)] = self._get_analysis_info(analysis_path)
            return analysis
        return {}
    
    def _get_analysis_info(self, file_path:Path) -> list:
        analysis_dict:dict = json.loads(file_path.read_bytes())
        return list(analysis_dict.keys())
    
    def show_info(self, group:str=None, patient_num:Union[int, list[int]]=None,
                    only_missing_info:bool=False, data_type:list[str]=None, only_summary:bool=False):
        if data_type is not None:
            if type(data_type) is not list:
                raise DatasetAccessError('data_types parameter must be a list of strings')
            for dtype in data_type:
                if dtype not in self.data_types:
                    raise DatasetAccessError(f"'{dtype}' is not a valid data type")
        print(f"+ CLEAN DATASET INFO (Path -> '{self.dataset_path}')")
        clean_dataset_info = """
        - Adquisitions per patient:
            -> 4 OCT (macular_OD, macular_OS, optic-nerve_OD, optic-nerve_OS)
            -> 4 OCTA (macular_OD, macular_OS, optic-nerve_OD, optic-nerve_OS)
            -> 2 retinographies (OD, OS)
            -> 8 scans in JSON analysis report
        """
        print(clean_dataset_info)
        data_paths:dict = self.get_data_paths(group=group, patient_num=patient_num)
        for group, group_info in data_paths.items():
            print('----------------------------------------------------')
            msg = f" + {group.upper()} GROUP"
            if patient_num is None:
                msg += f" (size={len(self.get_patients(group))})"
            else:
                msg += F", PATIENT {patient_num}"
            print(msg)
            # Variables to count missing info
            m_oct = 0; m_octa = 0; m_ret = 0; m_xml = 0; num_patients = len(group_info)
            if num_patients == 0:
                print("     -> This group is empty")
            else:
                for patient, p_info in group_info.items():
                    missing_info = {}; has_missing_info = False
                    for dtype in self.data_types:
                        if data_type is not None and dtype not in data_type: continue
                        if dtype == 'retinography':
                            ret_info = p_info.get(dtype, None)
                            if not bool(ret_info):
                                missing_info[dtype] = 'OD and OS missing'
                                has_missing_info = True; m_ret += 2
                            else:
                                for i in range(2):
                                    eye = 'right' if i == 0 else 'left'
                                    eye_info = ret_info.get(eye, None)
                                    if not bool(eye_info):
                                        missing_info[dtype] = f'{eye} missing'
                                        has_missing_info = True; m_ret += 1
                        if dtype == 'OCT' or dtype == 'OCTA':
                            img_info = p_info.get(dtype, None)
                            if not bool(img_info):
                                missing_info[dtype] = '2x macular (OD, OS), 2x optic nerve (OD, OS)'
                                has_missing_info = True;
                                if dtype == 'OCT': m_oct += 4
                                elif dtype == 'OCTA': m_octa += 4
                            else:
                                for i in range(2):
                                    zone = 'macula' if i == 0 else 'optic-nerve'
                                    zone_info = img_info.get(zone, None)
                                    if not bool(zone_info):
                                        missing_info[dtype] = f'2x {zone} (OD, OS)'
                                        has_missing_info = True
                                        if dtype == 'OCT': m_oct += 2
                                        elif dtype == 'OCTA': m_octa += 2
                                    else:
                                        for i in range(2):
                                            eye = 'right' if i == 0 else 'left'
                                            eye_info = zone_info.get(eye, None)
                                            if not bool(eye_info):
                                                missing_info[dtype] = f'{zone} {eye} missing'
                                                has_missing_info = True
                                                if dtype == 'OCT': m_oct += 1
                                                elif dtype == 'OCTA': m_octa += 1
                        if dtype == 'XML':
                            xml_files = p_info.get(dtype, None)
                            if not bool(xml_files):
                                missing_info[dtype] = "all 8 scans analysis report are missing"
                                m_xml += 8; has_missing_info = True
                            else:
                                for i in range(2):
                                    modality = 'OCT' if i == 0 else 'OCTA'
                                    for zone in self.zones:
                                        for eye in self.eyes.values():
                                            scan_name = modality+"_"+zone+"_"+eye
                                            for scans in xml_files.values():
                                                if scan_name in scans: break
                                            else:
                                                if dtype not in missing_info:
                                                    missing_info[dtype] = {}
                                                missing_info[dtype][scan_name] = "missing"
                                                m_xml += 1; has_missing_info = True
                    if not only_summary:
                        if not has_missing_info:
                            if not only_missing_info: 
                                msg = f" - '{patient}' has all adquisitions" 
                                msg += "" if data_type is None else f" of type {data_type}"
                                print(msg)
                        else:
                            print(f" - '{patient}' has missing info:")
                            str_missing_info = json.dumps(missing_info, indent=4)
                            tab = "     "; str_missing_info = str_missing_info.replace('\n', '\n'+tab)
                            print(tab+str_missing_info)
                if data_type is None:
                    summary_dtypes = list(self.data_types.keys())
                else:
                    summary_dtypes = data_type
                # Summary
                print(f" + SUMMARY:")
                # OCT
                total_octs = 0
                if 'OCT' in summary_dtypes:
                    total_octs = num_patients*4
                    oct_perc =  round((total_octs-m_oct)*100/total_octs, 2)
                    print(f'     -> OCT Cubes => {total_octs-m_oct}/{total_octs} ({oct_perc}%) -> ({m_oct} missing)')
                # OCTA
                total_octas = 0;
                if 'OCTA' in summary_dtypes:
                    total_octas = num_patients*4
                    octa_perc =  round((total_octas-m_octa)*100/total_octas, 2)
                    print(f'     -> OCTA Cubes => {total_octas-m_octa}/{total_octas} ({octa_perc}%) -> ({m_octa} missing)')
                # Retinographies
                total_retinos = 0;
                if 'retinography' in summary_dtypes:
                    total_retinos = num_patients*2;
                    ret_perc =  round((total_retinos-m_ret)*100/total_retinos, 2)
                    print(f'     -> Retina Images => {total_retinos-m_ret}/{total_retinos} ({ret_perc}%) -> ({m_ret} missing)')
                # XML scans analysis
                total_xml = 0;
                if 'XML' in summary_dtypes:
                    total_xml = num_patients*8
                    xml_perc =  round((total_xml-m_xml)*100/total_xml, 2)
                    print(f'     -> JSON scans => {total_xml-m_xml}/{total_xml} ({xml_perc}%) -> ({m_xml} missing)')
                # Global 
                total = total_octs + total_octas + total_retinos + total_xml
                total_missing = m_oct+m_octa+m_ret+m_xml; percentage = round((total-total_missing)*100/total, 2)
                print(f' -> Global data = {total-total_missing}/{total} ({percentage}%) -> ({total_missing} missing)')
            print('----------------------------------------------------')