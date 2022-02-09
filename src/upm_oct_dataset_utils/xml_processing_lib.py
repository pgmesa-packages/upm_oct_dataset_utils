
import json
from .dataset_classes import RawDataset
import pandas_read_xml as pdx

class XMLError(Exception):
    pass

def process_xmlscans(xml_path, scans_to_process:list) -> dict:
    xml_df = pdx.read_xml(xml_path)
    json_str = xml_df.to_json(indent=4)
    xml_dict = json.loads(json_str)
    studies = xml_dict["ExportSchema"]["0"].pop("PATIENT")["VISITS"]["STUDY"]
    if type(studies) is dict:
        studies = [studies]
    processed_scans = {}
    for scan in scans_to_process:
        for study in studies:
            series = study["SERIES"]
            if series is None: continue
            scans = series["SCAN"]
            if type(scans) is dict:
                scans = [scans]
            oct_type, zone, eye = scan.split("_")
            protocol = RawDataset.zones[zone]["adquisitions_name"][oct_type]
            for sc in scans:
                if sc['PROTOCOL'] == protocol and sc["SITE"] == eye:
                    sc.pop("TRACKINGDETAILS")
                    processed_scans[scan] = sc        
    return processed_scans