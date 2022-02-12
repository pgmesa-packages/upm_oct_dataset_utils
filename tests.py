
import src.upm_oct_dataset_utils.dataset_classes as ds

grp = ds.NMO; pnum = 1

raw_path = "D:\\study_datasets\\raw_dataset"

raw_ds = ds.RawDataset(raw_path)

# raw_ds.show_info()

clean_path = "D:\\study_datasets\\clean_dataset"

clean_ds = ds.CleanDataset(clean_path)

date = ds.StudyDate(17,11,2021)
# # d = clean_ds.get_study_dir(grp, pnum, date)
# # dir_path = clean_ds.get_dir_path(grp, pnum,d)
# # print(dir_path)
clean_ds.show_info()

