import json
import logging
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import List, Union

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import pydicom.misc
import ray
from nilearn import image
from numpyencoder import NumpyEncoder
from tqdm import tqdm, trange

from ..MEDscan import MEDscan
from ..processing.PETSUVConverter import PETSUVConverter
from ..processing.segmentation import get_roi_from_indexes
from ..utils.get_file_paths import get_file_paths
from ..utils.get_patient_names import get_patient_names
from ..utils.imref import imref3d
from ..utils.json_utils import load_json, save_json
from ..utils.save_MEDscan import save_MEDscan
from .ProcessDICOM import ProcessDICOM


class DataManager(object):
    """Reads all the raw data (DICOM, NIfTI) content and organizes it in instances of the MEDscan class."""


    @dataclass
    class DICOM(object):
        """DICOM data management class that will organize data during the conversion to MEDscan class process"""
        stack_series_rs: List
        stack_path_rs: List
        stack_frame_rs: List
        cell_series_id: List
        cell_path_rs: List
        cell_path_images: List
        cell_frame_rs: List
        cell_frame_id: List


    @dataclass
    class NIfTI(object):
        """NIfTI data management class that will organize data during the conversion to MEDscan class process"""
        stack_path_images: List
        stack_path_roi: List
        stack_path_all: List


    @dataclass
    class Paths(object):
        """Paths management class that will organize the paths used in the processing"""
        _path_to_dicoms: List
        _path_to_niftis: List
        _path_csv: Union[Path, str]
        _path_save: Union[Path, str]
        _path_save_checks: Union[Path, str]
        _path_pre_checks_settings: Union[Path, str]

    def __init__(
            self, 
            path_to_dicoms: List = [],
            path_to_niftis: List = [],
            path_csv: Union[Path, str] = None,
            path_save: Union[Path, str] = None,
            path_save_checks: Union[Path, str] = None,
            path_pre_checks_settings: Union[Path, str] = None,
            save: bool = True,
            n_batch: int = 2
    ) -> None:
        """Constructor of the class DataManager.

        Args:
            path_to_dicoms (Union[Path, str], optional): Full path to the starting directory
                where the DICOM data is located.
            path_to_niftis (Union[Path, str], optional): Full path to the starting directory
                where the NIfTI is located.
            path_csv (Union[Path, str], optional): Full path to the CSV file containing the scans info list.
            path_save (Union[Path, str], optional): Full path to the directory where to save all the MEDscan classes.
            path_save_checks(Union[Path, str], optional): Full path to the directory where to save all 
                the pre-radiomics checks analysis results.
            path_pre_checks_settings(Union[Path, str], optional): Full path to the JSON file of the pre-checks analysis
                parameters.
            save (bool, optional): True to save the MEDscan classes in `path_save`.
            n_batch (int, optional): Numerical value specifying the number of batch to use in the
                parallel computations (use 0 for serial computation).
        
        Returns:
            None
        """
        # Convert all paths to Pathlib.Path
        if path_to_dicoms:
            path_to_dicoms = Path(path_to_dicoms)
        if path_to_niftis:
            path_to_niftis = Path(path_to_niftis)
        if path_csv:
            path_csv = Path(path_csv)
        if path_save:
            path_save = Path(path_save)
        if path_save_checks:
            path_save_checks = Path(path_save_checks)
        if path_pre_checks_settings:
            path_pre_checks_settings = Path(path_pre_checks_settings)

        self.paths = self.Paths(
                path_to_dicoms,
                path_to_niftis,
                path_csv,
                path_save,
                path_save_checks,
                path_pre_checks_settings,
        )
        self.save = save
        self.n_batch = n_batch
        self.__dicom = self.DICOM(
                stack_series_rs=[],
                stack_path_rs=[],
                stack_frame_rs=[],
                cell_series_id=[],
                cell_path_rs=[],
                cell_path_images=[],
                cell_frame_rs=[],
                cell_frame_id=[]
        )
        self.__nifti = self.NIfTI(
                stack_path_images=[],
                stack_path_roi=[],
                stack_path_all=[]
        )
        self.path_to_objects = []
        self.summary = {}
        self.csv_data = None
        self.__studies = []
        self.__institutions = []
        self.__scans = []

    def __find_uid_cell_index(self, uid: Union[str, List[str]], cell: List[str]) -> List: 
        """Finds the cell with the same `uid`. If not is present in `cell`, creates a new position
        in the `cell` for the new `uid`.

        Args:
            uid (Union[str, List[str]]):  Unique identifier of the Series to find.
            cell (List[str]): List of Unique identifiers of the Series.

        Returns:
            Union[List[str], str]: List or string of the uid  
        """
        return [len(cell)] if uid not in cell else[i for i, e in enumerate(cell) if e == uid]

    def __get_list_of_files(self, dir_name: str) -> List:
        """Gets all files in the given directory

        Args:
            dir_name (str): directory name

        Returns:
            List: List of all files in the directory
        """
        list_of_file = os.listdir(dir_name)
        all_files = list()
        for entry in list_of_file:
            full_path = os.path.join(dir_name, entry)
            if os.path.isdir(full_path):
                all_files = all_files + self.__get_list_of_files(full_path)
            else:
                all_files.append(full_path)

        return all_files

    def __get_MEDscan_name_save(self, medscan: MEDscan) -> str:
        """Returns the name that will be used to save the MEDscan instance, based on the values of the attributes.

        Args:
            medscan(MEDscan): A MEDscan class instance.
        
        Returns:
            str: String of the name save.
        """
        series_description = medscan.series_description.translate({ord(ch): '-' for ch in '/\\ ()&:*'})
        name_id = medscan.patientID.translate({ord(ch): '-' for ch in '/\\ ()&:*'})
        # final saving name
        name_complete = name_id + '__' + series_description + '.' + medscan.type + '.npy'
        return name_complete

    def __associate_rt_stuct(self) -> None:
        """Associates the imaging volumes to their mask using UIDs

        Returns:
            None
        """
        print('--> Associating all RT objects to imaging volumes')
        n_rs = len(self.__dicom.stack_path_rs)
        self.__dicom.stack_series_rs = list(dict.fromkeys(self.__dicom.stack_series_rs))
        if n_rs:
            for i in trange(0, n_rs):
                try: 
                    # PUT ALL THE DICOM PATHS WITH THE SAME UID IN THE SAME PATH LIST
                    ind_series_id = self.__find_uid_cell_index(
                                                        self.__dicom.stack_series_rs[i], 
                                                        self.__dicom.cell_series_id)
                    for n in range(len(ind_series_id)):
                        if ind_series_id[n] < len(self.__dicom.cell_path_rs):
                            self.__dicom.cell_path_rs[ind_series_id[n]] += [self.__dicom.stack_path_rs[i]]
                except:
                    ind_series_id = self.__find_uid_cell_index(
                                                        self.__dicom.stack_frame_rs[i], 
                                                        self.__dicom.cell_frame_id)
                    for n in range(len(ind_series_id)):
                        if ind_series_id[n] < len(self.__dicom.cell_path_rs):
                            self.__dicom.cell_path_rs[ind_series_id[n]] += [self.__dicom.stack_path_rs[i]]
        print('DONE')

    def __read_all_dicoms(self) -> None:
        """Reads all the dicom files in the all the paths of the attribute `_path_to_dicoms`

        Returns:
            None
        """
        # SCANNING ALL FOLDERS IN INITIAL DIRECTORY
        print('\n--> Scanning all folders in initial directory...', end='')
        p = Path(self.paths._path_to_dicoms)
        e_rglob = '*.dcm'

        # EXTRACT ALL FILES IN THE PATH TO DICOMS
        if self.paths._path_to_dicoms.is_dir():
            stack_folder_temp = list(p.rglob(e_rglob))
            stack_folder = [x for x in stack_folder_temp if not x.is_dir()]
        elif str(self.paths._path_to_dicoms).find('json') != -1:
            with open(self.paths._path_to_dicoms) as f:
                data = json.load(f)
                for value in data.values():
                    stack_folder_temp = value
            directory_name = str(stack_folder_temp).replace("'", '').replace('[', '').replace(']', '')
            stack_folder = self.__get_list_of_files(directory_name)
        else:
            raise ValueError("The given dicom folder path either doesn't exist or not a folder.")
        # READ ALL DICOM FILES AND UPDATE ATTRIBUTES FOR FURTHER PROCESSING
        for file in tqdm(stack_folder):
            if pydicom.misc.is_dicom(file):
                try:
                    info = pydicom.dcmread(str(file))
                    if info.Modality in ['MR', 'PT', 'CT']:
                        ind_series_id = self.__find_uid_cell_index(
                                                        info.SeriesInstanceUID, 
                                                        self.__dicom.cell_series_id)[0]
                        if ind_series_id == len(self.__dicom.cell_series_id):  # New volume
                            self.__dicom.cell_series_id = self.__dicom.cell_series_id + [info.SeriesInstanceUID]
                            self.__dicom.cell_frame_id += [info.FrameOfReferenceUID]
                            self.__dicom.cell_path_images += [[]]
                            self.__dicom.cell_path_rs = self.__dicom.cell_path_rs + [[]]
                        self.__dicom.cell_path_images[ind_series_id] += [file]
                    elif info.Modality == 'RTSTRUCT':
                        self.__dicom.stack_path_rs += [file]
                        try:
                            series_uid = info.ReferencedFrameOfReferenceSequence[
                                        0].RTReferencedStudySequence[
                                        0].RTReferencedSeriesSequence[
                                        0].SeriesInstanceUID
                        except:
                            series_uid = 'NotFound'
                        self.__dicom.stack_series_rs += [series_uid]
                        try:
                            frame_uid = info.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
                        except:
                            frame_uid = info.FrameOfReferenceUID
                        self.__dicom.stack_frame_rs += [frame_uid]
                    else:
                        print("Modality not supported: ", info.Modality)

                except Exception as e:
                    print(f'Error while reading: {file}, error: {e}\n')
                    continue
        print('DONE')

        # ASSOCIATE ALL VOLUMES TO THEIR MASK
        self.__associate_rt_stuct()

    def process_all_dicoms(self) -> Union[List[MEDscan], None]:
        """This function reads the DICOM content of all the sub-folder tree of a starting directory defined by
        `path_to_dicoms`. It then organizes the data (files throughout the starting directory are associated by
        'SeriesInstanceUID') in the MEDscan class including the region of  interest (ROI) defined by an
        associated RTstruct. All MEDscan classes hereby created are saved in `path_save` with a name
        varying with every scan.

        Returns:
            List[MEDscan]: List of MEDscan instances.
        """
        # Initialize ray
        if ray.is_initialized():
            ray.shutdown()
        ray.init(local_mode=True, include_dashboard=False)

        print('--> Reading all DICOM objects to create MEDscan classes')
        self.__read_all_dicoms()

        print('--> Processing DICOMs and creating MEDscan objects')
        n_scans = len(self.__dicom.cell_path_images)
        if self.n_batch is None:
            n_batch = 1
        elif n_scans < self.n_batch:
            n_batch = n_scans
        else:
            n_batch = self.n_batch

        # Distribute the first tasks to all workers
        pds = [ProcessDICOM(
                        self.__dicom.cell_path_images[i], 
                        self.__dicom.cell_path_rs[i], 
                        self.paths._path_save,
                        self.save)
            for i in range(n_batch)]
        
        ids = [pd.process_files() for pd in pds]

        # Update the path to the created instances
        if self.save:
            for name_save in ray.get(ids):
                if self.paths._path_save:
                    self.path_to_objects.append(str(self.paths._path_save / name_save))
                # Update processing summary
                if name_save.split('_')[0].count('-') >= 2:
                    scan_type = name_save[name_save.find('__')+2 : name_save.find('.')]
                    if name_save.split('-')[0] not in self.__studies:
                        self.__studies.append(name_save.split('-')[0])  # add new study
                    if name_save.split('-')[1] not in self.__institutions:
                        self.__institutions.append(name_save.split('-')[1])  # add new study
                    if name_save.split('-')[0] not in self.summary:
                        self.summary[name_save.split('-')[0]] = {}
                    if name_save.split('-')[1] not  in self.summary[name_save.split('-')[0]]:
                        self.summary[name_save.split('-')[0]][name_save.split('-')[1]] = {}  # add new institution
                    if scan_type not in self.__scans:
                        self.__scans.append(scan_type)
                    if scan_type not in self.summary[name_save.split('-')[0]][name_save.split('-')[1]]:
                        self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type] = []
                    if name_save not in self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type]:
                        self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type].append(name_save)
                else:
                    logging.warning(f"The patient ID of the following file: {name_save} does not respect the MEDiml "\
                        "naming convention 'study-institution-id' (Ex: Glioma-TCGA-001)")

        nb_job_left = n_scans - n_batch

        if not self.save:
            return ray.get(ids)

        # Distribute the remaining tasks
        for _ in trange(n_scans):
            _, ids = ray.wait(ids, num_returns=1)
            if nb_job_left > 0:
                idx = n_scans - nb_job_left
                pd = ProcessDICOM(
                        self.__dicom.cell_path_images[idx], 
                        self.__dicom.cell_path_rs[idx], 
                        self.paths._path_save,
                        self.save)
                ids.extend([pd.process_files()])
                nb_job_left -= 1

            # Update the path to the created instances
            if self.save:
                for name_save in ray.get(ids):
                    if self.paths._path_save:
                        self.path_to_objects.extend(str(self.paths._path_save / name_save))
                    # Update processing summary
                    if name_save.split('_')[0].count('-') >= 2:
                        scan_type = name_save[name_save.find('__')+2 : name_save.find('.')]
                        if name_save.split('-')[0] not in self.__studies:
                            self.__studies.append(name_save.split('-')[0])  # add new study
                        if name_save.split('-')[1] not in self.__institutions:
                            self.__institutions.append(name_save.split('-')[1])  # add new study
                        if name_save.split('-')[0] not in self.summary:
                            self.summary[name_save.split('-')[0]] = {}
                        if name_save.split('-')[1] not  in self.summary[name_save.split('-')[0]]:
                            self.summary[name_save.split('-')[0]][name_save.split('-')[1]] = {}  # add new institution
                        if scan_type not in self.__scans:
                            self.__scans.append(scan_type)
                        if scan_type not in self.summary[name_save.split('-')[0]][name_save.split('-')[1]]:
                            self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type] = []
                        if name_save not in self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type]:
                            self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type].append(name_save)
                    else:
                        logging.warning(f"The patient ID of the following file: {name_save} does not respect the MEDiml "\
                            "naming convention 'study-institution-id' (Ex: Glioma-TCGA-001)")
            else:
                return ray.get(ids)
        print('DONE')

    def __read_all_niftis(self) -> None:
        """Reads all files in the initial path and organizes other path to images and roi
        in the class attributes.

        Returns:
            None.
        """
        print('\n--> Scanning all folders in initial directory')
        if not self.paths._path_to_niftis:
            raise ValueError("The path to the niftis is not defined")
        p = Path(self.paths._path_to_niftis)
        e_rglob1 = '*.nii'
        e_rglob2 = '*.nii.gz'

        # EXTRACT ALL FILES IN THE PATH TO DICOMS
        if p.is_dir():
            self.__nifti.stack_path_all = list(p.rglob(e_rglob1))
            self.__nifti.stack_path_all.extend(list(p.rglob(e_rglob2)))
        else:
            raise TypeError(f"{p} must be a path to a directory")

        all_niftis = list(self.__nifti.stack_path_all)
        for i in trange(0, len(all_niftis)):
            if 'ROI' in all_niftis[i].name.split("."):
                self.__nifti.stack_path_roi.append(all_niftis[i])
            else:
                self.__nifti.stack_path_images.append(all_niftis[i])
        print('DONE')

    def __associate_roi_to_image(
            self,
            image_file: Union[Path, str], 
            medscan: MEDscan, 
            nifti: nib.Nifti1Image,
            path_roi_data: Path = None
        ) -> MEDscan:
        """Extracts all ROI data from the given path for the given patient ID and updates all class attributes with
        the new extracted data.

        Args:
            image_file(Union[Path, str]): Path to the ROI data.
            medscan (MEDscan): MEDscan class instance that will hold the data. 

        Returns:
            MEDscan: Returns a MEDscan instance with updated roi attributes.
        """
        def load_mask(_id, file, medscan):
            # Load the patient's ROI nifti files:
            if file.name.startswith(_id) and 'ROI' in file.name.split("."):
                roi = nib.load(file)
                roi = image.resample_to_img(roi, nifti, interpolation='nearest')
                roi_data = roi.get_fdata()
                roi_name = file.name[file.name.find("(") + 1 : file.name.find(")")]
                name_set = file.name[file.name.find("_") + 2 : file.name.find("(")]
                medscan.data.ROI.update_indexes(key=roi_index, indexes=np.nonzero(roi_data.flatten()))
                medscan.data.ROI.update_name_set(key=roi_index, name_set=name_set)
                medscan.data.ROI.update_roi_name(key=roi_index, roi_name=roi_name)  
            else:
                raise ValueError(f"The ROI file for patient ID: {_id} "
                    f"was not found in the given path: {file} or was not correctly named.")   
        
        image_file = Path(image_file)
        roi_index = 0
        if not path_roi_data:
            if not self.paths._path_to_niftis:
                raise ValueError("The path to the niftis is not defined")
            else:
                path_roi_data = self.paths._path_to_niftis

            for file in self.__nifti.stack_path_roi:
                _id = file.name.split("(")[0] if ("(") in file.name else file.name # id is PatientID__ImagingScanName
                load_mask(_id, file, medscan)
                roi_index += 1
        else:
            _id = image_file.name.split("(")[0] if ("(") in image_file.name else image_file.name # id is PatientID__ImagingScanName

            # Check the path type
            if path_roi_data.is_dir():
                for file in path_roi_data.rglob('*.nii*'):
                    if file.name.startswith(_id) and 'ROI' in file.name.split("."):
                        load_mask(_id, file, medscan)
                        roi_index += 1
            elif path_roi_data.name.startswith(_id) and 'ROI' in path_roi_data.name.split("."):
                load_mask(_id, path_roi_data, medscan)
            else:
                raise ValueError(f"The ROI file for patient ID: {_id} "
                    f"was not found in the given path: {path_roi_data} or was not correctly named.")

        return medscan

    def __associate_spatialRef(self, nifti_file: Union[Path, str], medscan: MEDscan) -> MEDscan:
        """Computes the imref3d spatialRef using a NIFTI file and updates the spatialRef attribute.

        Args:
            nifti_file(Union[Path, str]): Path to the nifti data.
            medscan (MEDscan): MEDscan class instance that will hold the data.

        Returns:
            MEDscan: Returns a MEDscan instance with updated spatialRef attribute.
        """
        # Loading the nifti file :
        nifti = nib.load(nifti_file)
        nifti_data = medscan.data.volume.array

        # spatialRef Creation
        pixel_x = abs(nifti.affine[0, 0])
        pixel_y = abs(nifti.affine[1, 1])
        slices = abs(nifti.affine[2, 2])
        min_grid = nifti.affine[:3, 3] * [-1.0, -1.0, 1.0] # x and y are flipped
        min_x_grid = min_grid[0]
        min_y_grid = min_grid[1]
        min_z_grid = min_grid[2]
        size_image = np.shape(nifti_data)
        spatialRef = imref3d(size_image, abs(pixel_x), abs(pixel_y), abs(slices))
        spatialRef.XWorldLimits = (np.array(spatialRef.XWorldLimits) -
                                (spatialRef.XWorldLimits[0] -
                                    (min_x_grid-pixel_x/2))
                                ).tolist()
        spatialRef.YWorldLimits = (np.array(spatialRef.YWorldLimits) -
                                (spatialRef.YWorldLimits[0] -
                                    (min_y_grid-pixel_y/2))
                                ).tolist()
        spatialRef.ZWorldLimits = (np.array(spatialRef.ZWorldLimits) -
                                (spatialRef.ZWorldLimits[0] -
                                    (min_z_grid-slices/2))
                                ).tolist()

        # Converting the results into lists
        spatialRef.ImageSize = spatialRef.ImageSize.tolist()
        spatialRef.XIntrinsicLimits = spatialRef.XIntrinsicLimits.tolist()
        spatialRef.YIntrinsicLimits = spatialRef.YIntrinsicLimits.tolist()
        spatialRef.ZIntrinsicLimits = spatialRef.ZIntrinsicLimits.tolist()

        # update spatialRef in the volume sub-class
        medscan.data.volume.update_spatialRef(spatialRef)

        return medscan

    def __process_one_nifti(self, nifti_file: Union[Path, str], path_data) -> MEDscan:
        """
        Processes one NIfTI file to create a MEDscan class instance.

        Args:
            nifti_file (Union[Path, str]): Path to the NIfTI file.
            path_data (Union[Path, str]): Path to the data where associated files are located.
        
        Returns:
            MEDscan: MEDscan class instance.
        """
        medscan = MEDscan()
        medscan.patientID = os.path.basename(nifti_file).split("_")[0]
        medscan.type = os.path.basename(nifti_file).split(".")[-3]
        medscan.series_description = nifti_file.name[nifti_file.name.find('__') + 2: nifti_file.name.find('(')]
        medscan.format = "nifti"
        medscan.data.set_orientation(orientation="Axial")
        medscan.data.set_patient_position(patient_position="HFS")
        medscan.data.volume.array = nib.load(nifti_file).get_fdata()
        medscan.data.volume.scan_rot = None
        
        # Update spatialRef
        self.__associate_spatialRef(nifti_file, medscan)
        
        # Assiocate ROI
        medscan = self.__associate_roi_to_image(nifti_file, medscan, nib.load(nifti_file), path_data)

        return medscan
    
    def process_all(self) -> None:
        """Processes both DICOM & NIfTI content to create MEDscan classes
        """
        self.process_all_dicoms()
        self.process_all_niftis()

    def process_all_niftis(self) -> List[MEDscan]:
        """This function reads the NIfTI content of all the sub-folder tree of a starting directory. 
        It then organizes the data in the MEDscan class including the region of  interest (ROI)
        defined by an associated mask file. All MEDscan classes hereby created are saved in a specific path
        with a name specific name varying with every scan.

        Args:
            None.

        Returns:
            List[MEDscan]: List of MEDscan instances.
        """

        # Reading all NIfTI files
        self.__read_all_niftis()

        # Create the MEDscan instances
        print('--> Reading all NIfTI objects (imaging volumes & masks) to create MEDscan classes')
        list_instances = []
        for file in tqdm(self.__nifti.stack_path_images):
            try:
                # Assert the list of instances does not exceed the a size of 10
                if len(list_instances) >= 10:
                    print('The number of MEDscan instances exceeds 10, please consider saving the instances')
                    break
                # INITIALIZE MEDscan INSTANCE AND UPDATE ATTRIBUTES
                medscan = MEDscan()
                medscan.patientID = os.path.basename(file).split("_")[0]
                medscan.type = os.path.basename(file).split(".")[-3]
                medscan.series_description = file.name[file.name.find('__') + 2: file.name.find('(')] if '__' in file.name else ""
                medscan.format = "nifti"
                medscan.data.set_orientation(orientation="Axial")
                medscan.data.set_patient_position(patient_position="HFS")
                medscan.data.volume.array = nib.load(file).get_fdata()
                
                # RAS to LPS
                #medscan.data.volume.convert_to_LPS()
                medscan.data.volume.scan_rot = None
                
                # Update spatialRef
                medscan = self.__associate_spatialRef(file, medscan)
                
                # Get ROI
                medscan = self.__associate_roi_to_image(file, medscan, nib.load(file))

                # SAVE MEDscan INSTANCE
                if self.save and self.paths._path_save:
                    save_MEDscan(medscan, self.paths._path_save)
                else:
                    list_instances.append(medscan)
                
                # Update the path to the created instances
                name_save = self.__get_MEDscan_name_save(medscan)

                # Clear memory
                del medscan

                # Update the path to the created instances
                if self.paths._path_save:
                    self.path_to_objects.append(str(self.paths._path_save / name_save))
                
                # Update processing summary
                if name_save.split('_')[0].count('-') >= 2:
                    scan_type = name_save[name_save.find('__')+2 : name_save.find('.')]
                    if name_save.split('-')[0] not in self.__studies:
                        self.__studies.append(name_save.split('-')[0])  # add new study
                    if name_save.split('-')[1] not in self.__institutions:
                        self.__institutions.append(name_save.split('-')[1])  # add new institution
                    if name_save.split('-')[0] not in self.summary:
                        self.summary[name_save.split('-')[0]] = {}  # add new study to summary
                    if name_save.split('-')[1] not  in self.summary[name_save.split('-')[0]]:
                        self.summary[name_save.split('-')[0]][name_save.split('-')[1]] = {}  # add new institution
                    if scan_type not in self.__scans:
                        self.__scans.append(scan_type)
                    if scan_type not in self.summary[name_save.split('-')[0]][name_save.split('-')[1]]:
                        self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type] = []
                    if name_save not in self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type]:
                        self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type].append(name_save)
                else:
                    if self.save:
                        logging.warning(f"The patient ID of the following file: {name_save} does not respect the MEDiml "\
                            "naming convention 'study-institution-id' (Ex: Glioma-TCGA-001)")
            except Exception as e:
                print(f'Error while processing: {file}, error: {e}\n')
        print('DONE')

        if list_instances:
            return list_instances

    def process_one_dicom(self) -> MEDscan:
        """Processes one DICOM file to create a MEDscan class instance.

        Args:
            path_image (Union[Path, str]): Path to the DICOM image data.
            path_mask (Union[Path, str]): Path to the DICOM mask data.
        
        Returns:
            MEDscan: MEDscan class instance.
        """
        self.__read_all_dicoms()

        # Ensure only one image and one mask are given
        if len(self.__dicom.cell_path_images) > 1 or len(self.__dicom.cell_path_rs) > 1:
            raise ValueError("More than one image or mask found, please make sure the folder provided contains a single scan.")

        medscan = ProcessDICOM(
            self.__dicom.cell_path_images[0],
            self.__dicom.cell_path_rs[0],
            save=False
        ).process_files()

        # SAVE MEDscan INSTANCE
        if self.save and self.paths._path_save:
            save_MEDscan(medscan, self.paths._path_save)

        return medscan

    def process_one_nifti(self, path_image: Union[Path, str], path_mask: Union[Path, str]) -> MEDscan:
        """Processes one NIfTI file to create a MEDscan class instance.

        Args:
            nifti_file (Union[Path, str]): Path to the NIfTI file.
            path_data (Union[Path, str]): Path to the data.
        
        Returns:
            MEDscan: MEDscan class instance.
        """
        medscan = self.__process_one_nifti(path_image, path_mask)

        # SAVE MEDscan INSTANCE
        if self.save and self.paths._path_save:
            save_MEDscan(medscan, self.paths._path_save)

        return medscan
    
    def update_from_csv(self, path_csv: Union[str, Path] = None) -> None:
        """Updates the class from a given CSV and summarizes the processed scans again according to it.

        Args:
            path_csv(optional, Union[str, Path]): Path to a csv file, if not given, will check
                for csv info in the class attributes.
        
        Returns:
            None
        """
        if not (path_csv or self.paths._path_csv):
            print('No csv provided, no updates will be made')
        else:
            if path_csv:
                self.paths._path_csv = path_csv
            # Extract roi type label from csv file name
            name_csv = self.paths._path_csv.name
            roi_type_label = name_csv[name_csv.find('_')+1 : name_csv.find('.')]

            # Create a dictionary
            csv_data = {}
            csv_data[roi_type_label] = pd.read_csv(self.paths._path_csv)
            self.csv_data = csv_data
            self.summarize()

    def summarize(self, retrun_summary: bool = False) -> None:
        """Creates and shows a summary of processed scans organized by study, institution, scan type and roi type

        Args:
            retrun_summary (bool, optional): If True, will return the summary as a dictionary.
        
        Returns:
            None
        """
        def count_scans(summary):
            count = 0
            if type(summary) == dict:
                for study in summary:
                    if type(summary[study]) == dict:
                        for institution in summary[study]:
                            if type(summary[study][institution]) == dict:
                                for scan in self.summary[study][institution]:
                                    count += len(summary[study][institution][scan])
                            else:
                                count += len(summary[study][institution])
                    else:
                        count += len(summary[study])
            elif type(summary) == list:
                count = len(summary)
            return count

        summary_df = pd.DataFrame(columns=['study', 'institution', 'scan_type', 'roi_type', 'count'])

        for study in self.summary:
            summary_df = summary_df.append({
                                        'study': study,
                                        'institution': "",
                                        'scan_type': "",
                                        'roi_type': "",
                                        'count' : count_scans(self.summary)
                                        }, ignore_index=True)
            for institution in self.summary[study]:
                summary_df = summary_df.append({
                                            'study': study,
                                            'institution': institution,
                                            'scan_type': "",
                                            'roi_type': "",
                                            'count' : count_scans(self.summary[study][institution])
                                            }, ignore_index=True)
                for scan in self.summary[study][institution]:
                    summary_df = summary_df.append({
                                                'study': study,
                                                'institution': institution,
                                                'scan_type': scan,
                                                'roi_type': "",
                                                'count' : count_scans(self.summary[study][institution][scan])
                                                }, ignore_index=True)
                    if self.csv_data:
                        roi_count = 0
                        for roi_type in self.csv_data:
                            csv_table = pd.DataFrame(self.csv_data[roi_type])
                            csv_table['under'] = '_'
                            csv_table['dot'] = '.'
                            csv_table['npy'] = '.npy'
                            name_patients = (pd.Series(
                                csv_table[['PatientID', 'under', 'under',
                                        'ImagingScanName',
                                        'dot',
                                        'ImagingModality',
                                        'npy']].fillna('').values.tolist()).str.join('')).tolist()
                            for patient_id in self.summary[study][institution][scan]:
                                if patient_id in name_patients:
                                    roi_count += 1
                            summary_df = summary_df.append({
                                                'study': study,
                                                'institution': institution,
                                                'scan_type': scan,
                                                'roi_type': roi_type,
                                                'count' : roi_count
                                                }, ignore_index=True)
        print(summary_df.to_markdown(index=False))

        if retrun_summary:
            return summary_df

    def __adapt_wildcard_for_format(
            self,
            wildcard: str,
            use_dicoms: bool,
            use_niftis: bool
        ) -> str:
        if use_dicoms:
            for ext in ('.npy', '.nii.gz', '.nii'):
                wildcard = wildcard.replace(ext, '.dcm')
            return wildcard
        if use_niftis:
            for ext in ('.npy', '.dcm'):
                wildcard = wildcard.replace(ext, '.nii*')
            return wildcard
        for ext in ('.dcm', '.nii.gz', '.nii'):
            wildcard = wildcard.replace(ext, '.npy')
        return wildcard

    def __filter_file_paths_by_format(
            self,
            file_paths: List[Path],
            use_dicoms: bool,
            use_niftis: bool
        ) -> List[Path]:
        if use_dicoms:
            return [f for f in file_paths if f.name.endswith('.dcm')]
        if use_niftis:
            return [f for f in file_paths if f.name.endswith('.nii') or f.name.endswith('.nii.gz')]
        return [f for f in file_paths if f.name.endswith('.npy')]

    def __get_pre_radiomics_file_paths(
            self,
            path_data: Union[Path, str],
            wildcard: str,
            use_dicoms: bool,
            use_niftis: bool
        ) -> List[Path]:
        wildcard = self.__adapt_wildcard_for_format(wildcard, use_dicoms, use_niftis)
        if path_data:
            file_paths = get_file_paths(path_data, wildcard)
        elif self.paths._path_save:
            file_paths = get_file_paths(self.paths._path_save, wildcard)
        else:
            raise ValueError("Path data is invalid.")
        return self.__filter_file_paths_by_format(file_paths, use_dicoms, use_niftis)

    def __load_medscan_for_pre_radiomics_checks(
            self,
            file_path: Path,
            path_data: Union[Path, str],
            use_dicoms: bool,
            use_niftis: bool
        ):
        if use_dicoms:
            return self.process_one_dicom(file_path)
        if use_niftis:
            if 'ROI' in file_path.name.split("."):
                return None
            return self.__process_one_nifti(file_path, path_data)
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def __pre_radiomics_checks_dimensions(
            self,
            path_data: Union[Path, str] = None,
            wildcards_dimensions: List[str] = [],
            min_percentile: float = 0.05,
            max_percentile: float = 0.95,
            use_dicoms: bool = False,
            use_niftis: bool = False,
            save: bool = False
        ) -> None:
        """Finds proper voxels dimension options for radiomics analyses for a group of scans

        Args:
            path_data (Path, optional): Path to the MEDscan objects, if not specified will use ``path_save`` from the 
                inner-class ``Paths`` in the current instance.
            wildcards_dimensions(List[str], optional): List of wildcards that determines the scans 
                that will be analyzed. You can learn more about wildcards in
                :ref:`this link <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`.
            min_percentile (float, optional): Minimum percentile to use for the histograms. Defaults to 0.05.
            max_percentile (float, optional): Maximum percentile to use for the histograms. Defaults to 0.95.
            use_dicoms (bool, optional): Load DICOM files (``.dcm``). Defaults to False.
            use_niftis (bool, optional): Load NIfTI files (``.nii`` / ``.nii.gz``). Defaults to False.
            save (bool, optional): If True, will save the results in a json file. Defaults to False.
        
        Returns:
            None.
        """
        xy_dim = {
            "data": [],
            "mean": [],
            "median": [],
            "std": [],
            "min": [],
            "max": [],
            f"p{min_percentile}": [],
            f"p{max_percentile}": []
        }
        z_dim = {
            "data": [],
            "mean": [],
            "median": [],
            "std": [],
            "min": [],
            "max": [],
            f"p{min_percentile}": [],
            f"p{max_percentile}": []
        }
        if type(wildcards_dimensions) is str:
            wildcards_dimensions = [wildcards_dimensions]

        if len(wildcards_dimensions) == 0:
            print("Wildcard is empty, the pre-checks will be aborted")
            return

        # Plotting style
        plt.rcParams["figure.figsize"] = (12, 5)
        plt.rcParams.update({"font.size": 12})
        _HIST_COLOR   = "#4472C4"
        _PMIN_COLOR   = "#E74C3C"
        _PMAX_COLOR   = "#27AE60"
        _MED_COLOR    = "#F39C12"

        # TODO: seperate by studies and scan type (MRscan, CTscan...)
        # TODO: Two summaries (df, list of names saves) -> 
        # name_save = name_save(ROI) : Glioma-Huashan-001__T1.MRscan.npy({GTV})
        file_paths = list()
        for w in range(len(wildcards_dimensions)):
            wildcard = wildcards_dimensions[w]
            file_paths = self.__get_pre_radiomics_file_paths(
                path_data, wildcard, use_dicoms, use_niftis
            )
            n_files = len(file_paths)
            xy_dim["data"] = np.zeros((n_files, 1))
            xy_dim["data"] = np.multiply(xy_dim["data"], np.nan)
            z_dim["data"] = np.zeros((n_files, 1))
            z_dim["data"] = np.multiply(z_dim["data"], np.nan)
            for f in tqdm(range(len(file_paths))):
                try:
                    medscan = self.__load_medscan_for_pre_radiomics_checks(
                        file_paths[f], path_data, use_dicoms, use_niftis
                    )
                    if medscan is None:
                        continue
                    xy_dim["data"][f] = medscan.data.volume.spatialRef.PixelExtentInWorldX
                    z_dim["data"][f]  = medscan.data.volume.spatialRef.PixelExtentInWorldZ
                except Exception as e:
                    print(f'Error while processing: {file_paths[f]}, error: {e}\n')
                    continue
            
            # Safe check
            if all(np.isnan(xy_dim["data"])) or all(np.isnan(z_dim["data"])):
                print(f"All xy-spacing or z-spacing data are NaN for the following wildcard: {wildcard}," \
                    " the pre-checks will be aborted")
                continue

            # Running analysis
            xy_dim["data"] = np.concatenate(xy_dim["data"])
            xy_dim["mean"] = np.mean(xy_dim["data"][~np.isnan(xy_dim["data"])])
            xy_dim["median"] = np.median(xy_dim["data"][~np.isnan(xy_dim["data"])])
            xy_dim["std"] = np.std(xy_dim["data"][~np.isnan(xy_dim["data"])])
            xy_dim["min"] = np.min(xy_dim["data"][~np.isnan(xy_dim["data"])])
            xy_dim["max"] = np.max(xy_dim["data"][~np.isnan(xy_dim["data"])])
            xy_dim[f"p{min_percentile}"] = np.percentile(xy_dim["data"][~np.isnan(xy_dim["data"])], 
                                                        min_percentile)
            xy_dim[f"p{max_percentile}"] = np.percentile(xy_dim["data"][~np.isnan(xy_dim["data"])], 
                                                        max_percentile)
            z_dim["mean"] = np.mean(z_dim["data"][~np.isnan(z_dim["data"])])
            z_dim["median"] = np.median(z_dim["data"][~np.isnan(z_dim["data"])])
            z_dim["std"] = np.std(z_dim["data"][~np.isnan(z_dim["data"])])
            z_dim["min"] = np.min(z_dim["data"][~np.isnan(z_dim["data"])])
            z_dim["max"] = np.max(z_dim["data"][~np.isnan(z_dim["data"])])
            z_dim[f"p{min_percentile}"] = np.percentile(z_dim["data"][~np.isnan(z_dim["data"])], 
                                                        min_percentile)
            z_dim[f"p{max_percentile}"] = np.percentile(z_dim["data"][~np.isnan(z_dim["data"])], max_percentile)
            xy_dim["data"] = xy_dim["data"].tolist()
            z_dim["data"] = z_dim["data"].tolist()

            # Plot: xy-spacing histogram 
            df_xy = pd.DataFrame(xy_dim["data"], columns=['data'])
            del xy_dim["data"]  # no interest in keeping data (we only need statistics)
            min_quant, max_quant, median = (
                df_xy.quantile(min_percentile),
                df_xy.quantile(max_percentile),
                df_xy.median()
            )

            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            ax.hist(
                df_xy['data'].dropna(), bins=20,
                color=_HIST_COLOR, alpha=0.80, edgecolor='white', linewidth=0.6
            )
            ax.axvline(
                min_quant.data, linestyle='--', color=_PMIN_COLOR, linewidth=1.8,
                label=f"p{int(min_percentile * 100)}: {float(min_quant):.3f} mm"
            )
            ax.axvline(
                max_quant.data, linestyle='--', color=_PMAX_COLOR, linewidth=1.8,
                label=f"p{int(max_percentile * 100)}: {float(max_quant):.3f} mm"
            )
            ax.axvline(
                median.data, linestyle='-', color=_MED_COLOR, linewidth=2.0,
                label=f"Median: {float(median.data):.3f} mm"
            )
            ax.set_xlabel("Voxel Spacing (mm)", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title(
                f"Voxels xy-Spacing Distribution — {wildcard}",
                fontsize=13, fontweight='bold', pad=10
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='grey')
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=11)
            fig.tight_layout()
            if save:
                if not self.paths._path_save_checks:
                    print("Warning: save is set to True but the path to save checks is not defined, the figure will not be saved")
                else:
                    fig.savefig(
                        self.paths._path_save_checks / 'Voxels_xy_check.png',
                        dpi=150, bbox_inches='tight'
                    )
            else:
                plt.show()
            plt.close(fig)

            # Plot: z-spacing histogram
            df_z = pd.DataFrame(z_dim["data"], columns=['data'])
            del z_dim["data"]  # no interest in keeping data (we only need statistics)
            min_quant, max_quant, median = (
                df_z.quantile(min_percentile),
                df_z.quantile(max_percentile),
                df_z.median()
            )

            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            ax.hist(
                df_z['data'].dropna(), bins=20,
                color=_HIST_COLOR, alpha=0.80, edgecolor='white', linewidth=0.6
            )
            ax.axvline(
                min_quant.data, linestyle='--', color=_PMIN_COLOR, linewidth=1.8,
                label=f"p{int(min_percentile * 100)}: {float(min_quant):.3f} mm"
            )
            ax.axvline(
                max_quant.data, linestyle='--', color=_PMAX_COLOR, linewidth=1.8,
                label=f"p{int(max_percentile * 100)}: {float(max_quant):.3f} mm"
            )
            ax.axvline(
                median.data, linestyle='-', color=_MED_COLOR, linewidth=2.0,
                label=f"Median: {float(median.data):.3f} mm"
            )
            ax.set_xlabel("Voxel Spacing (mm)", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title(
                f"Voxels z-Spacing Distribution — {wildcard}",
                fontsize=13, fontweight='bold', pad=10
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='grey')
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=11)
            fig.tight_layout()
            if save:
                if not self.paths._path_save_checks:
                    print("Warning: save is set to True but the path to save checks is not defined, the figure will not be saved")
                else:
                    fig.savefig(
                        self.paths._path_save_checks / 'Voxels_z_check.png',
                        dpi=150, bbox_inches='tight'
                    )
            else:
                plt.show()
            plt.close(fig)
                
            # Saving files using wildcard for name
            if save:
                if not self.paths._path_save_checks:
                    print("Warning: save is set to True but the path to save checks is not defined, the checks will not be saved")
                else:
                    wildcard = str(wildcard).replace('*', '').replace('.npy', '.json')
                    save_json(self.paths._path_save_checks / ('xyDim_' + wildcard), xy_dim, cls=NumpyEncoder)
                    save_json(self.paths._path_save_checks / ('zDim_' + wildcard), z_dim, cls=NumpyEncoder)


    def __pre_radiomics_checks_window(
            self,
            path_data: Union[str, Path] = None,
            wildcards_window: List = [], 
            path_csv: Union[str, Path] = None,
            min_percentile: float = 0.05,
            max_percentile: float = 0.95,
            bin_width: int = 0,
            hist_range: list = [],
            compute_suv_map: bool = False,
            use_dicoms: bool = False,
            use_niftis: bool = False,
            save: bool = False
        ) -> None:
        """Finds proper re-segmentation ranges options for radiomics analyses for a group of scans

        Args:
            path_data (Path, optional): Path to the MEDscan objects, if not specified will use ``path_save`` from the 
                inner-class ``Paths`` in the current instance.
            wildcards_window(List[str], optional): List of wildcards that determines the scans 
                that will be analyzed. You can learn more about wildcards in
                :ref:`this link <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`.
            path_csv(Union[str, Path], optional): Path to a csv file containing a list of the scans that will be
                analyzed (a CSV file for a single ROI type).
            min_percentile (float, optional): Minimum percentile to use for the histograms. Defaults to 0.05.
            max_percentile (float, optional): Maximum percentile to use for the histograms. Defaults to 0.95.
            bin_width(int, optional): Width of the bins for the histograms. If not provided, will use the 
                default number of bins in the method 
                :ref:`pandas.DataFrame.hist <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.hist.html>`: 10 bins.
            hist_range(list, optional): Range of the histograms. If empty, will use the minimum and maximum values.
            compute_suv_map(bool, optional): Whether to compute the SUV map for PET scans that do not have SUV values.
            use_dicoms (bool, optional): Load DICOM files (``.dcm``). Defaults to False.
            use_niftis (bool, optional): Load NIfTI files (``.nii`` / ``.nii.gz``). Defaults to False.
            save (bool, optional): If True, will save the results in a json file. Defaults to False.
        
        Returns:
            None.
        """
        # Plotting style
        plt.rcParams["figure.figsize"] = (14, 5)
        plt.rcParams.update({"font.size": 12})
        _HIST_COLOR   = "#4472C4"
        _PMIN_COLOR   = "#E74C3C"
        _PMAX_COLOR   = "#27AE60"
        
        if type(wildcards_window) is str:
            wildcards_window = [wildcards_window]

        if len(wildcards_window) == 0:
            raise ValueError("Wildcard is empty, the pre-checks will be aborted")
        if path_csv is not None:
            self.paths._path_csv = Path(path_csv)
        elif self.paths._path_csv is None:
            raise ValueError("Cannot run pre-radiomics windows checks, please provide a csv file containing the list of scans to " \
            "analyze or set the path_csv attribute in the class.")
        roi_table = pd.read_csv(self.paths._path_csv)
        for w in range(len(wildcards_window)):
            temp_val = []
            temp = []
            file_paths = []
            roi_data= {
                "data": [],
                "mean": [],
                "median": [],
                "std": [],
                "min": [],
                "max": [],
                f"p{min_percentile}": [],
                f"p{max_percentile}": []
            }
            wildcard = wildcards_window[w]
            if not path_data and self.paths._path_save:
                path_data = self.paths._path_save
            file_paths = self.__get_pre_radiomics_file_paths(
                path_data, wildcard, use_dicoms, use_niftis
            )
            n_files = len(file_paths)
            i = 0
            for f in tqdm(range(len(file_paths))):
                try:
                    medscan = self.__load_medscan_for_pre_radiomics_checks(
                        file_paths[f], path_data, use_dicoms, use_niftis
                    )
                    if medscan is None:
                        continue
                    
                    # Compute SUV map if it's a PET scan without SUV values
                    if compute_suv_map:
                        suv_converter = PETSUVConverter(medscan.dicomH)
                        medscan.data.volume.array = suv_converter.compute(np.double(medscan.data.volume.array))

                    # Extract ROI values according to csv file
                    patient_id, sequence, modality = medscan.patientID, medscan.series_description, medscan.type
                    name_roi = roi_table[(roi_table['PatientID'] == patient_id) & 
                                         (roi_table['ImagingScanName'] == sequence) & 
                                         (roi_table['ImagingModality'] == modality)
                                        ]['ROIname'].values
                    if len(name_roi) == 0:
                        print(f"No ROI found for patient {patient_id} with sequence {sequence} and modality {modality} in the csv file, skipping this scan.")
                        continue

                    name_roi = name_roi[0]
                    vol_obj_init, roi_obj_init = get_roi_from_indexes(medscan, name_roi, 'box')
                    temp = vol_obj_init.data[roi_obj_init.data == 1]
                    temp_val.append(len(temp))
                    roi_data["data"].append(np.zeros(shape=(n_files, temp_val[i])))
                    roi_data["data"][i] = temp
                    i+=1
                    del medscan
                    del vol_obj_init
                    del roi_obj_init
                except Exception as e:
                    print(f"Problem with patient {patient_id}, error: {e}")
            
            roi_data["data"] = np.concatenate(roi_data["data"])
            roi_data["mean"] = np.mean(roi_data["data"][~np.isnan(roi_data["data"])])
            roi_data["median"] = np.median(roi_data["data"][~np.isnan(roi_data["data"])])
            roi_data["std"] = np.std(roi_data["data"][~np.isnan(roi_data["data"])])
            roi_data["min"] = np.min(roi_data["data"][~np.isnan(roi_data["data"])])
            roi_data["max"] = np.max(roi_data["data"][~np.isnan(roi_data["data"])])
            roi_data[f"p{min_percentile}"] = np.percentile(roi_data["data"][~np.isnan(roi_data["data"])], 
                                                        min_percentile)
            roi_data[f"p{max_percentile}"] = np.percentile(roi_data["data"][~np.isnan(roi_data["data"])], 
                                                        max_percentile)
            
            # Set bin width if not provided
            if bin_width != 0:
                if hist_range:
                    nb_bins = (round(hist_range[1]) - round(hist_range[0])) // bin_width
                else:
                    nb_bins = (round(roi_data["max"]) - round(roi_data["min"])) // bin_width
            else:
                nb_bins = 10
                if hist_range:
                    bin_width = int((round(hist_range[1]) - round(hist_range[0])) // nb_bins)
                else:
                    bin_width = int((round(roi_data["max"]) - round(roi_data["min"])) // nb_bins)
            nb_bins = int(nb_bins)

            # Set histogram range if not provided
            if not hist_range:
                hist_range = (roi_data["min"], roi_data["max"])

            # re-segment data according to histogram range
            roi_data["data"] = roi_data["data"][(roi_data["data"] > hist_range[0]) & (roi_data["data"] < hist_range[1])]
            df_data = pd.DataFrame(roi_data["data"], columns=['data'])
            del roi_data["data"]  # no interest in keeping data (we only need statistics)

            # Plot: intensity range histogram 
            min_quant, max_quant = (
                df_data.quantile(min_percentile),
                df_data.quantile(max_percentile)
            )

            fig, ax = plt.subplots(1, 1, figsize=(14, 5))
            ax.hist(
                df_data['data'].dropna(), bins=nb_bins,
                range=(hist_range[0], hist_range[1]),
                color=_HIST_COLOR, alpha=0.80, edgecolor='white', linewidth=0.6
            )
            ax.axvline(
                min_quant.data, linestyle='--', color=_PMIN_COLOR, linewidth=1.8,
                label=f"p{int(min_percentile * 100)}: {float(min_quant):.3f}"
            )
            ax.axvline(
                max_quant.data, linestyle='--', color=_PMAX_COLOR, linewidth=1.8,
                label=f"p{int(max_percentile * 100)}: {float(max_quant):.3f}"
            )
            ax.set_xlabel("Intensity Values", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title(
                f"Intensity Range Distribution — {wildcard}  (bin width = {bin_width})",
                fontsize=13, fontweight='bold', pad=10
            )
            _max_x_ticks = 12
            if nb_bins <= _max_x_ticks:
                tick_positions = np.arange(
                    hist_range[0], hist_range[1] + bin_width / 2, bin_width
                )
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(
                    [f"{t:g}" for t in tick_positions],
                    rotation=45, ha='right', fontsize=10
                )
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=_max_x_ticks, min_n_ticks=4))
                x_formatter = ScalarFormatter(useOffset=False)
                x_formatter.set_scientific(False)
                ax.xaxis.set_major_formatter(x_formatter)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            ax.xaxis.set_tick_params(pad=5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='grey')
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=11)
            fig.tight_layout()
            if save:
                if not self.paths._path_save_checks:
                    print("Warning: save is set to True but the path to save checks is not defined, the figure will not be saved")
                else:
                    fig.savefig(
                        self.paths._path_save_checks / f'Intensity_range_check_bw_{bin_width}.png',
                        dpi=150, bbox_inches='tight'
                    )
            else:
                plt.show()
            plt.close(fig)
            
            # save final checks
            if save:
                if not self.paths._path_save_checks:
                    print("Warning: save is set to True but the path to save checks is not defined, the checks will not be saved")
                else:
                    wildcard = str(wildcard).replace('*', '').replace('.npy', '.json')
                    save_json(self.paths._path_save_checks / ('roi_data_' + wildcard), roi_data, cls=NumpyEncoder)

    def pre_radiomics_checks(
            self,
            path_data: Union[str, Path] = None,
            wildcards_dimensions: List = [],
            wildcards_window: List = [],
            path_csv: Union[str, Path] = None,
            min_percentile: float = 0.05,
            max_percentile: float = 0.95,
            bin_width: int = 0,
            hist_range: list = [],
            compute_suv_map : bool = False,
            dimensions_only: bool = False,
            intensity_only: bool = False,
            use_dicoms: bool = False,
            use_niftis: bool = False,
            save: bool = False
        ) -> None:
        """Finds proper dimension and re-segmentation ranges options for radiomics analyses. 

        The resulting files from this method can then be analyzed and used to set up radiomics 
        parameters options in computation methods.

        Args:
            path_data (Path, optional): Path to the MEDscan objects, if not specified will use ``path_save`` from the 
                inner-class ``Paths`` in the current instance.
            wildcards_dimensions(List[str], optional): List of wildcards that determines the scans 
                that will be analyzed. You can learn more about wildcards in
                `this link <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`_.
            wildcards_window(List[str], optional): List of wildcards that determines the scans 
                that will be analyzed. You can learn more about wildcards in
                `this link <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`_.
            path_csv(Union[str, Path], optional): Path to a csv file containing a list of the scans that will be
                analyzed (a CSV file for a single ROI type).
            min_percentile (float, optional): Minimum percentile to use for the histograms. Defaults to 0.05.
            max_percentile (float, optional): Maximum percentile to use for the histograms. Defaults to 0.95.
            bin_width(int, optional): Width of the bins for the histograms. If not provided, will use the 
                default number of bins in the method 
                :ref:`pandas.DataFrame.hist <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.hist.html>`: 10 bins.
            hist_range(list, optional): Range of the histograms. If empty, will use the minimum and maximum values.
            compute_suv_map (bool, optional): Whether to compute the SUV map for PET scans that do not have SUV values.
            dimensions_only (bool, optional): If True, will only perform the dimensions checks. Defaults to False.
            intensity_only (bool, optional): If True, will only perform the intensity checks. Defaults to False.
            use_dicoms (bool, optional): Set to True to load DICOM files (``.dcm``). Cannot be used together with ``use_niftis``.
            use_niftis (bool, optional): Set to True to load NIfTI files (``.nii`` / ``.nii.gz``). Cannot be used together with ``use_dicoms``.
                If both ``use_dicoms`` and ``use_niftis`` are False, numpy files (``.npy``) are loaded by default.
            save (bool, optional): If True, will save the results in a json file. Defaults to False.

        Returns:
            None
        """
        if dimensions_only and intensity_only:
            raise ValueError("dimensions_only and intensity_only cannot both be True. Please select one option.")

        if use_dicoms and use_niftis:
            raise ValueError("use_dicoms and use_niftis cannot both be True. Please select one file format.")

        # Initialization
        path_study = Path.cwd()

        # Load params
        if not self.paths._path_pre_checks_settings:
            if not wildcards_dimensions or not wildcards_window:
                raise ValueError("path to pre-checks settings is None.\
                    wildcards_dimensions and wildcards_window need to be specified")
        else:
            settings = self.paths._path_pre_checks_settings
            settings = load_json(settings)
            settings = settings['pre_radiomics_checks']  

            # Setting up paths
            if 'path_save_checks' in settings and settings['path_save_checks']:
                self.paths._path_save_checks = Path(settings['path_save_checks']) 
            if 'path_csv' in settings and settings['path_csv']:
                self.paths._path_csv = Path(settings['path_csv']) 

            # Wildcards of groups of files to analyze for dimensions in path_data.
            # See for example: https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html
            # Keep the cell empty if no dimension checks are to be performed.
            if not intensity_only and not wildcards_dimensions:
                wildcards_dimensions = []
                for i in range(len(settings['wildcards_dimensions'])):
                    wildcards_dimensions.append(settings['wildcards_dimensions'][i])

            # ROI intensity window checks params
            if not dimensions_only and not wildcards_window:
                wildcards_window = []
                for i in range(len(settings['wildcards_window'])):
                    wildcards_window.append(settings['wildcards_window'][i])

        # PRE-RADIOMICS CHECKS
        if not self.paths._path_save_checks:
            if (path_study / 'checks').exists():
                self.paths._path_save_checks = Path(path_study / 'checks')
            else:
                os.mkdir(path_study / 'checks')
                self.paths._path_save_checks = Path(path_study / 'checks')
        else:
            if self.paths._path_save_checks.name != 'checks':
                if (self.paths._path_save_checks / 'checks').exists():
                    self.paths._path_save_checks /= 'checks'
                else:
                    os.mkdir(self.paths._path_save_checks / 'checks')
                    self.paths._path_save_checks = Path(self.paths._path_save_checks / 'checks')

        # Initializing plotting params
        plt.rcParams["figure.figsize"] = (20,20)
        plt.rcParams.update({'font.size': 22})
        
        start = time()
        print('\n\n************************* PRE-RADIOMICS CHECKS *************************', end='')

        # 1. PRE-RADIOMICS CHECKS -- DIMENSIONS
        if not intensity_only:
            start1 = time()
            print('\n--> PRE-RADIOMICS CHECKS -- DIMENSIONS ... ', end='')
            self.__pre_radiomics_checks_dimensions(
                                            path_data, 
                                            wildcards_dimensions, 
                                            min_percentile, 
                                            max_percentile,
                                            use_dicoms,
                                            use_niftis,
                                            save)
            print('DONE', end='')
            time1 = f"{time() - start1:.2f}"
            print(f'\nElapsed time: {time1} sec', end='')

        # 2. PRE-RADIOMICS CHECKS - WINDOW
        if not dimensions_only:
            start2 = time()
            print('\n\n--> PRE-RADIOMICS CHECKS -- WINDOW ... \n', end='')
            self.__pre_radiomics_checks_window(
                                            path_data, 
                                            wildcards_window, 
                                            path_csv,
                                            min_percentile, 
                                            max_percentile,
                                            bin_width,
                                            hist_range,
                                            compute_suv_map,
                                            use_dicoms=use_dicoms,
                                            use_niftis=use_niftis,
                                            save=save)
            print('DONE', end='')
            time2 = f"{time() - start2:.2f}"
            print(f'\nElapsed time: {time2} sec', end='')

        time_elapsed = f"{time() - start:.2f}"
        print(f'\n\n--> TOTAL TIME FOR PRE-RADIOMICS CHECKS: {time_elapsed} seconds')
        print('-------------------------------------------------------------------------------------')

    def perform_mr_imaging_summary(self, 
                                wildcards_scans: List[str],
                                path_data: Path = None,
                                path_save_checks: Path = None,
                                min_percentile: float = 0.05,
                                max_percentile: float = 0.95
                                ) -> None:
        """
        Summarizes MRI imaging acquisition parameters. Plots summary histograms
        for different dimensions and saves all acquisition parameters locally in JSON files.

        Args:
            wildcards_scans (List[str]): List of wildcards that determines the scans 
                that will be analyzed (Only MRI scans will be analyzed). You can learn more about wildcards in
                `this link <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`_.
                For example: ``[\"STS*.MRscan.npy\"]``.
            path_data (Path, optional): Path to the MEDscan objects, if not specified will use ``path_save`` from the 
                inner-class ``Paths`` in the current instance.
            path_save_checks (Path, optional): Path where to save the checks, if not specified will use the one 
                in the current instance.
            min_percentile (float, optional): Minimum percentile to use for the histograms. Defaults to 0.05.
            max_percentile (float, optional): Maximum percentile to use for the histograms. Defaults to 0.95.
        
        Returns:
            None.
        """
        # Initializing data structures
        class param:
            dates = []
            manufacturer = []
            scanning_sequence = []
            class years:
                data = []

            class fieldStrength:
                data = []

            class repetitionTime:
                data = []

            class echoTime:
                data = []

            class inversionTime:
                data = []

            class echoTrainLength:
                data = []

            class flipAngle:
                data = []

            class numberAverages:
                data = []

            class xyDim:
                data = []

            class zDim:
                data = []
        
        if len(wildcards_scans) == 0:
            print('wildcards_scans is empty')

        # wildcards checks:
        no_mr_scan = True
        for wildcard in wildcards_scans:
            if 'MRscan' in wildcard:
                no_mr_scan = False
        if no_mr_scan:
            raise ValueError(f"wildcards: {wildcards_scans} does not include MR scans. (Only MR scans are supported)")
        
        # Initialization
        if path_data is None:
            if self.paths._path_save:
                path_data = Path(self.paths._path_save)
            else:
                print("No path to data was given and path save is None.")
                return 0
        
        if not path_save_checks:
            if self.paths._path_save_checks:
                path_save_checks = Path(self.paths._path_save_checks)
            else:
                if (Path(os.getcwd()) / "checks").exists():
                    path_save_checks = Path(os.getcwd()) / "checks"
                else:
                    path_save_checks = (Path(os.getcwd()) / "checks").mkdir()
        # Looping through all the different wildcards
        for i in tqdm(range(len(wildcards_scans))):
            wildcard = wildcards_scans[i]
            file_paths = get_file_paths(path_data, wildcard)
            n_files = len(file_paths)
            param.dates = np.zeros(n_files)
            param.years.data = np.zeros((n_files, 1))
            param.years.data = np.multiply(param.years.data, np.NaN)
            param.manufacturer = [None] * n_files
            param.scanning_sequence = [None] * n_files
            param.fieldStrength.data = np.zeros((n_files, 1))
            param.fieldStrength.data = np.multiply(param.fieldStrength.data, np.NaN)
            param.repetitionTime.data = np.zeros((n_files, 1))
            param.repetitionTime.data = np.multiply(param.repetitionTime.data, np.NaN)
            param.echoTime.data = np.zeros((n_files, 1))
            param.echoTime.data = np.multiply(param.echoTime.data, np.NaN)
            param.inversionTime.data = np.zeros((n_files, 1))
            param.inversionTime.data = np.multiply(param.inversionTime.data, np.NaN)
            param.echoTrainLength.data = np.zeros((n_files, 1))
            param.echoTrainLength.data = np.multiply(param.echoTrainLength.data, np.NaN)
            param.flipAngle.data = np.zeros((n_files, 1))
            param.flipAngle.data = np.multiply(param.flipAngle.data, np.NaN)
            param.numberAverages.data = np.zeros((n_files, 1))
            param.numberAverages.data = np.multiply(param.numberAverages.data, np.NaN)
            param.xyDim.data = np.zeros((n_files, 1))
            param.xyDim.data = np.multiply(param.xyDim.data, np.NaN)
            param.zDim.data = np.zeros((n_files, 1))
            param.zDim.data = np.multiply(param.zDim.data, np.NaN)
            
            # Loading and recording data
            for f in tqdm(range(n_files)):
                file = file_paths[f]

                #Open file for warning
                try:
                    warn_file = open(path_save_checks / 'imaging_summary_mr_warnings.txt', 'a')
                except IOError:
                    print("Could not open warning file")

                # Loading Data
                try:
                    print(f'\nCurrently working on: {file}', file = warn_file)
                    with open(path_data / file, 'rb') as fe: medscan = pickle.load(fe)

                    # Example of DICOM header
                    info = medscan.dicomH[1]
                    # Recording dates (info.AcquistionDates)
                    try:
                        param.dates[f] = info.AcquisitionDate
                    except AttributeError:
                        param.dates[f] = info.StudyDate
                    # Recording years
                    try:
                        y = str(param.dates[f])  # Only the first four characters represent the years
                        param.years.data[f] = y[0:4]
                    except Exception as e:
                        print(f'Cannot read years of: {file}. Error: {e}', file = warn_file)
                    # Recording manufacturers
                    try:
                        param.manufacturer[f] = info.Manufacturer
                    except Exception as e:
                        print(f'Cannot read manufacturer of: {file}. Error: {e}', file = warn_file)
                    # Recording scanning sequence
                    try:
                        param.scanning_sequence[f] = info.scanning_sequence
                    except Exception as e:
                        print(f'Cannot read scanning sequence of: {file}. Error: {e}', file = warn_file)
                    # Recording field strength
                    try:
                        param.fieldStrength.data[f] = info.MagneticFieldStrength
                    except Exception as e:
                        print(f'Cannot read field strength of: {file}. Error: {e}', file = warn_file)
                    # Recording repetition time
                    try:
                        param.repetitionTime.data[f] = info.RepetitionTime
                    except Exception as e:
                        print(f'Cannot read repetition time of: {file}. Error: {e}', file = warn_file)
                    # Recording echo time
                    try:
                        param.echoTime.data[f] = info.EchoTime
                    except Exception as e:
                        print(f'Cannot read echo time of: {file}. Error: {e}', file = warn_file)
                    # Recording inversion time
                    try:
                        param.inversionTime.data[f] = info.InversionTime
                    except Exception as e:
                        print(f'Cannot read inversion time of: {file}. Error: {e}', file = warn_file)
                    # Recording echo train length
                    try:
                        param.echoTrainLength.data[f] = info.EchoTrainLength
                    except Exception as e:
                        print(f'Cannot read echo train length of: {file}. Error: {e}', file = warn_file)
                    # Recording flip angle
                    try:
                        param.flipAngle.data[f] = info.FlipAngle
                    except Exception as e:
                        print(f'Cannot read flip angle of: {file}. Error: {e}', file = warn_file)
                    # Recording number of averages
                    try:
                        param.numberAverages.data[f] = info.NumberOfAverages
                    except Exception as e:
                        print(f'Cannot read number averages of: {file}. Error: {e}', file = warn_file)
                    # Recording xy spacing
                    try:
                        param.xyDim.data[f] = medscan.data.volume.spatialRef.PixelExtentInWorldX
                    except Exception as e:
                        print(f'Cannot read x spacing of: {file}. Error: {e}', file = warn_file)
                    # Recording z spacing
                    try:
                        param.zDim.data[f] = medscan.data.volume.spatialRef.PixelExtentInWorldZ
                    except Exception as e:
                        print(f'Cannot read z spacing of: {file}', file = warn_file)
                except Exception as e:
                    print(f'Cannot read file: {file}. Error: {e}', file = warn_file)

                warn_file.close()

            # Summarize data
                # Summarizing years
            df_years = pd.DataFrame(param.years.data, 
                                    columns=['years']).describe(percentiles=[min_percentile, max_percentile], 
                                    include='all')
                # Summarizing field strength
            df_fs = pd.DataFrame(param.fieldStrength.data, 
                                columns=['fieldStrength']).describe(percentiles=[min_percentile, max_percentile], 
                                include='all')
                # Summarizing  repetition time
            df_rt = pd.DataFrame(param.repetitionTime.data, 
                                columns=['repetitionTime']).describe(percentiles=[min_percentile, max_percentile], 
                                include='all')
                # Summarizing echo time
            df_et = pd.DataFrame(param.echoTime.data, 
                                columns=['echoTime']).describe(percentiles=[min_percentile, max_percentile], 
                                include='all')
                # Summarizing inversion time
            df_it = pd.DataFrame(param.inversionTime.data, 
                                columns=['inversionTime']).describe(percentiles=[min_percentile, max_percentile], 
                                include='all')
                # Summarizing echo train length
            df_etl = pd.DataFrame(param.echoTrainLength.data, 
                                columns=['echoTrainLength']).describe(percentiles=[min_percentile, max_percentile], 
                                include='all')
                # Summarizing flip  angle
            df_fa = pd.DataFrame(param.flipAngle.data, 
                                columns=['flipAngle']).describe(percentiles=[min_percentile, max_percentile], 
                                include='all')
                # Summarizing number of  averages
            df_na = pd.DataFrame(param.numberAverages.data, 
                                columns=['numberAverages']).describe(percentiles=[min_percentile, max_percentile], 
                                include='all')
                # Summarizing xy-spacing
            df_xy = pd.DataFrame(param.xyDim.data, 
                                columns=['xyDim'])
                # Summarizing z-spacing
            df_z = pd.DataFrame(param.zDim.data, 
                                columns=['zDim'])

            # Plotting xy-spacing histogram
            ax = df_xy.hist(column='xyDim')
            min_quant, max_quant, average = df_xy.quantile(min_percentile), df_xy.quantile(max_percentile), param.xyDim.data.mean()
            for x in ax[0]:
                x.axvline(min_quant.xyDim, linestyle=':', color='r', label=f"Min Percentile: {float(min_quant):.3f}")
                x.axvline(max_quant.xyDim, linestyle=':', color='g', label=f"Max Percentile: {float(max_quant):.3f}")
                x.axvline(average, linestyle='solid', color='gold', label=f"Average: {float(average):.3f}")
                x.grid(False)
                plt.title(f"MR xy-spacing imaging summary for {wildcard}")
                plt.legend()
                plt.show()
            
            # Plotting z-spacing histogram
            ax = df_z.hist(column='zDim')
            min_quant, max_quant, average = df_z.quantile(min_percentile), df_z.quantile(max_percentile), param.zDim.data.mean()
            for x in ax[0]:
                x.axvline(min_quant.zDim, linestyle=':', color='r', label=f"Min Percentile: {float(min_quant):.3f}")
                x.axvline(max_quant.zDim, linestyle=':', color='g', label=f"Max Percentile: {float(max_quant):.3f}")
                x.axvline(average, linestyle='solid', color='gold', label=f"Average: {float(average):.3f}")
                x.grid(False)
                plt.title(f"MR z-spacing imaging summary for {wildcard}")
                plt.legend()
                plt.show()
            
            # Summarizing xy-spacing
            df_xy = df_xy.describe(percentiles=[min_percentile, max_percentile], include='all')
            # Summarizing  z-spacing
            df_z = df_z.describe(percentiles=[min_percentile, max_percentile], include='all')
            
            # Saving data
            name_save = wildcard.replace('*', '').replace('.npy', '')
            save_name = 'imagingSummary__' + name_save + ".json"
            df_all = [df_years, df_fs, df_rt, df_et, df_it, df_etl, df_fa, df_na, df_xy, df_z]
            df_all = df_all[0].join(df_all[1:])
            df_all.to_json(path_save_checks / save_name, orient='columns', indent=4)

    def perform_ct_imaging_summary(self, 
                                wildcards_scans: List[str],
                                path_data: Path = None,
                                path_save_checks: Path = None,
                                min_percentile: float = 0.05,
                                max_percentile: float = 0.95
                                ) -> None:
        """
        Summarizes CT imaging acquisition parameters. Plots summary histograms
        for different dimensions and saves all acquisition parameters locally in JSON files.

        Args:
            wildcards_scans (List[str]): List of wildcards that determines the scans 
                that will be analyzed (Only MRI scans will be analyzed). You can learn more about wildcards in
                `this link <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`_.
                For example: ``[\"STS*.CTscan.npy\"]``.
            path_data (Path, optional): Path to the MEDscan objects, if not specified will use ``path_save`` from the 
                inner-class ``Paths`` in the current instance.
            path_save_checks (Path, optional): Path where to save the checks, if not specified will use the one 
                in the current instance.
            min_percentile (float, optional): Minimum percentile to use for the histograms. Defaults to 0.05.
            max_percentile (float, optional): Maximum percentile to use for the histograms. Defaults to 0.95.
        
        Returns:
            None.
        """

        class param:
            manufacturer = []
            dates = []
            kernel = []

            class years:
                data = []
            class voltage:
                data = []
            class exposure:
                data = []
            class xyDim:
                data = []
            class zDim:
                data = []

        if len(wildcards_scans) == 0:
            print('wildcards_scans is empty')

        # wildcards checks:
        no_mr_scan = True
        for wildcard in wildcards_scans:
            if 'CTscan' in wildcard:
                no_mr_scan = False
        if no_mr_scan:
            raise ValueError(f"wildcards: {wildcards_scans} does not include CT scans. (Only CT scans are supported)")

        # Initialization
        if path_data is None:
            if self.paths._path_save:
                path_data = Path(self.paths._path_save)
            else:
                print("No path to data was given and path save is None.")
                return 0
        
        if not path_save_checks:
            if self.paths._path_save_checks:
                path_save_checks = Path(self.paths._path_save_checks)
            else:
                if (Path(os.getcwd()) / "checks").exists():
                    path_save_checks = Path(os.getcwd()) / "checks"
                else:
                    path_save_checks = (Path(os.getcwd()) / "checks").mkdir()

        # Looping through all the different wildcards
        for i in tqdm(range(len(wildcards_scans))):
            wildcard = wildcards_scans[i]
            file_paths = get_file_paths(path_data, wildcard)
            n_files = len(file_paths)
            param.dates = np.zeros(n_files)
            param.years.data = np.zeros(n_files)
            param.years.data = np.multiply(param.years.data, np.NaN)
            param.manufacturer = [None] * n_files
            param.voltage.data = np.zeros(n_files)
            param.voltage.data = np.multiply(param.voltage.data, np.NaN)
            param.exposure.data = np.zeros(n_files)
            param.exposure.data = np.multiply(param.exposure.data, np.NaN)
            param.kernel = [None] * n_files
            param.xyDim.data = np.zeros(n_files)
            param.xyDim.data = np.multiply(param.xyDim.data, np.NaN)
            param.zDim.data = np.zeros(n_files)
            param.zDim.data = np.multiply(param.zDim.data, np.NaN)
        
            # Loading and recording data
            for f in tqdm(range(n_files)):
                file = file_paths[f]

                # Open file for warning
                try:
                    warn_file = open(path_save_checks / 'imaging_summary_ct_warnings.txt', 'a')
                except IOError:
                    print("Could not open file")

                # Loading Data
                try:
                    with open(path_data / file, 'rb') as fe: medscan = pickle.load(fe)
                    print(f'Currently working on: {file}', file=warn_file)
                    
                    # DICOM header
                    info = medscan.dicomH[1]

                    # Recording dates
                    try:
                        param.dates[f] = info.AcquisitionDate
                    except AttributeError:
                        param.dates[f] = info.StudyDate
                        # Recording years
                    try:
                        years = str(param.dates[f])  # Only the first four characters represent the years
                        param.years.data[f] = years[0:4]
                    except Exception as e:
                        print(f'Cannot read dates of : {file}. Error: {e}', file=warn_file)
                        # Recording manufacturers
                    try:
                        param.manufacturer[f] = info.Manufacturer
                    except Exception as e:
                        print(f'Cannot read Manufacturer of: {file}. Error: {e}', file=warn_file)
                        # Recording voltage
                    try:
                        param.voltage.data[f] = info.KVP
                    except Exception as e:
                        print(f'Cannot read voltage of: {file}. Error: {e}', file=warn_file)
                        # Recording exposure
                    try:
                        param.exposure.data[f] = info.Exposure
                    except Exception as e:
                        print(f'Cannot read exposure of: {file}. Error: {e}', file=warn_file)
                        # Recording reconstruction kernel
                    try:
                        param.kernel[f] = info.ConvolutionKernel
                    except Exception as e:
                        print(f'Cannot read Kernel of: {file}. Error: {e}', file=warn_file)
                        # Recording xy spacing
                    try:
                        param.xyDim.data[f] = medscan.data.volume.spatialRef.PixelExtentInWorldX
                    except Exception as e:
                        print(f'Cannot read x spacing of: {file}. Error: {e}', file=warn_file)
                        # Recording z spacing
                    try:
                        param.zDim.data[f] = medscan.data.volume.spatialRef.PixelExtentInWorldZ
                    except Exception as e:
                        print(f'Cannot read z spacing of: {file}. Error: {e}', file=warn_file)
                except Exception as e:
                    print(f'Cannot load file: {file}', file=warn_file)

                warn_file.close()

            # Summarize data
                # Summarizing years
            df_years = pd.DataFrame(param.years.data, columns=['years']).describe(percentiles=[min_percentile, max_percentile], include='all')
                # Summarizing voltage
            df_voltage = pd.DataFrame(param.voltage.data, columns=['voltage']).describe(percentiles=[min_percentile, max_percentile], include='all')
                # Summarizing exposure
            df_exposure = pd.DataFrame(param.exposure.data, columns=['exposure']).describe(percentiles=[min_percentile, max_percentile], include='all')
                # Summarizing kernel
            df_kernel = pd.DataFrame(param.kernel, columns=['kernel']).describe(percentiles=[min_percentile, max_percentile], include='all')
                # Summarize xy spacing
            df_xy = pd.DataFrame(param.xyDim.data, columns=['xyDim']).describe(percentiles=[min_percentile, max_percentile], include='all')
                # Summarize z spacing
            df_z = pd.DataFrame(param.zDim.data, columns=['zDim']).describe(percentiles=[min_percentile, max_percentile], include='all')
                # Summarizing xy-spacing
            df_xy = pd.DataFrame(param.xyDim.data, columns=['xyDim'])
                # Summarizing z-spacing
            df_z = pd.DataFrame(param.zDim.data, columns=['zDim'])

            # Plotting xy-spacing histogram
            ax = df_xy.hist(column='xyDim')
            min_quant, max_quant, average = df_xy.quantile(min_percentile), df_xy.quantile(max_percentile), param.xyDim.data.mean()
            for x in ax[0]:
                x.axvline(min_quant.xyDim, linestyle=':', color='r', label=f"Min Percentile: {float(min_quant):.3f}")
                x.axvline(max_quant.xyDim, linestyle=':', color='g', label=f"Max Percentile: {float(max_quant):.3f}")
                x.axvline(average, linestyle='solid', color='gold', label=f"Average: {float(average):.3f}")
                x.grid(False)
                plt.title(f"CT xy-spacing imaging summary for {wildcard}")
                plt.legend()
                plt.show()
            
            # Plotting z-spacing histogram
            ax = df_z.hist(column='zDim')
            min_quant, max_quant, average = df_z.quantile(min_percentile), df_z.quantile(max_percentile), param.zDim.data.mean()
            for x in ax[0]:
                x.axvline(min_quant.zDim, linestyle=':', color='r', label=f"Min Percentile: {float(min_quant):.3f}")
                x.axvline(max_quant.zDim, linestyle=':', color='g', label=f"Max Percentile: {float(max_quant):.3f}")
                x.axvline(average, linestyle='solid', color='gold', label=f"Average: {float(average):.3f}")
                x.grid(False)
                plt.title(f"CT z-spacing imaging summary for {wildcard}")
                plt.legend()
                plt.show()
            
            # Summarizing xy-spacing
            df_xy = df_xy.describe(percentiles=[min_percentile, max_percentile], include='all')
            # Summarizing  z-spacing
            df_z = df_z.describe(percentiles=[min_percentile, max_percentile], include='all')

            # Saving data
            name_save = wildcard.replace('*', '').replace('.npy', '')
            save_name = 'imagingSummary__' + name_save + ".json"
            df_all = [df_years, df_voltage, df_exposure, df_kernel, df_xy, df_z]
            df_all = df_all[0].join(df_all[1:])
            df_all.to_json(path_save_checks / save_name, orient='columns', indent=4)

    def perform_imaging_summary(self, 
                                wildcards_scans: List[str],
                                path_data: Path = None,
                                path_save_checks: Path = None,
                                min_percentile: float = 0.05,
                                max_percentile: float = 0.95
                                ) -> None:
        """
        Summarizes CT and MR imaging acquisition parameters. Plots summary histograms
        for different dimensions and saves all acquisition parameters locally in JSON files.

        Args:
            wildcards_scans (List[str]): List of wildcards that determines the scans 
                that will be analyzed (CT and MRI scans will be analyzed). You can learn more about wildcards in
                `this link <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`_.
                For example: ``[\"STS*.CTscan.npy\", \"STS*.MRscan.npy\"]``.
            path_data (Path, optional): Path to the MEDscan objects, if not specified will use ``path_save`` from the 
                inner-class ``Paths`` in the current instance.
            path_save_checks (Path, optional): Path where to save the checks, if not specified will use the one 
                in the current instance.
            min_percentile (float, optional): Minimum percentile to use for the histograms. Defaults to 0.05.
            max_percentile (float, optional): Maximum percentile to use for the histograms. Defaults to 0.95.
        
        Returns:
            None.
        """
        # MR imaging summary
        wildcards_scans_mr = [wildcard for wildcard in wildcards_scans if 'MRscan' in wildcard]
        if len(wildcards_scans_mr) == 0:
            print("Cannot perform imaging summary for MR, no MR scan wildcard was given! ")
        else:
            self.perform_mr_imaging_summary(
                                wildcards_scans_mr, 
                                path_data, 
                                path_save_checks, 
                                min_percentile, 
                                max_percentile)
        # CT imaging summary
        wildcards_scans_ct = [wildcard for wildcard in wildcards_scans if 'CTscan' in wildcard]
        if len(wildcards_scans_ct) == 0:
            print("Cannot perform imaging summary for CT, no CT scan wildcard was given! ")
        else:
            self.perform_ct_imaging_summary(
                                wildcards_scans_ct, 
                                path_data, 
                                path_save_checks, 
                                min_percentile, 
                                max_percentile)
