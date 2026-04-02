import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from numpyencoder import NumpyEncoder
from pycaret.classification import *

from MEDiml.learning.DataCleaner import DataCleaner
from MEDiml.learning.DesignExperiment import DesignExperiment
from MEDiml.learning.Estimator import Estimator
from MEDiml.learning.FSR import FSR
from MEDiml.learning.ml_utils import (average_results, combine_rad_tables,
                                      feature_importance_analysis,
                                      get_ml_test_table, get_radiomics_table,
                                      intersect)
from MEDiml.learning.Normalization import CombatNormalization
from MEDiml.learning.Results import Results

from ..utils.json_utils import load_json, save_json


class RadiomicsLearner:
    def __init__(self, path_study: Path, path_workspace: Path, path_settings: Path, experiment_label: str) -> None:
        """
        Constructor of the class DesignExperiment.

        Args:
            path_study (Path): Path to the main study folder where patients partition dictionaries are found.
            path_workspace (Path): Path to the workspace folder where features and outcomes tables are found.
            path_settings (Path): Path to the settings folder.
            experiment_label (str): String specifying the label to attach to a given learning experiment in 
                "path_experiments". This label will be attached to the ml__$experiments_label$.json file as well
                as the learn__$experiment_label$ folder. This label is used to keep track of different experiments 
                with different settings (e.g. radiomics, scans, machine learning algorithms, etc.).
        
        Returns:
            None
        """
        self.path_study = Path(path_study)
        self.path_workspace = Path(path_workspace)
        self.path_settings = Path(path_settings)
        self.experiment_label = experiment_label
    
    def __load_ml_info(self, ml_dict_paths: Dict) -> Dict:
        """
        Initializes the test dictionary information (training patients, test patients, ML dict, etc).

        Args:
            ml_dict_paths (Dict): Dictionary containing the paths to the different files needed 
                to run the machine learning experiment.
        
        Returns:
            dict: Dictionary containing the information of the machine learning test.
        """
        ml_dict = dict()

        # Training and test patients
        ml_dict['patientsTrain'] = load_json(ml_dict_paths['patientsTrain'])
        ml_dict['patientsTest'] = load_json(ml_dict_paths['patientsTest'])

        # Outcome table for training and test patients
        outcome_table = pd.read_csv(ml_dict_paths['outcomes'], index_col=0)
        ml_dict['outcome_table_binary'] = outcome_table.iloc[:, [0]]
        if outcome_table.shape[1] == 2:
            ml_dict['outcome_table_time'] = outcome_table.iloc[:, [1]]
        
        # Machine learning dictionary
        ml_dict['ml'] = load_json(ml_dict_paths['ml'])
        ml_dict['path_results'] = ml_dict_paths['results']

        return ml_dict
    
    def get_hold_out_set_table(self, ml: Dict, var_id: str, patients_id: List):
        """
        Loads and pre-processes different radiomics tables then combines them to be used for hold-out testing.

        Args:
            ml (Dict): The machine learning dictionary containing the information of the machine learning test.
            var_id (str): String specifying the ID of the radiomics variable in ml.
                --> Ex: var1
            patients_id (List): List of patients of the hold-out set.

        Returns:
            pd.DataFrame: Radiomics table for the hold-out set.
        """
        # Loading and pre-processing
        rad_var_struct = ml['variables'][var_id]
        rad_tables_holdout = list()
        for item in rad_var_struct['path'].values():
            # Reading the table
            path_radiomics_csv = item['csv']
            path_radiomics_txt = item['txt']
            image_type = item['type']
            rad_table_holdout = get_radiomics_table(path_radiomics_csv, path_radiomics_txt, image_type, patients_id)
            rad_tables_holdout.append(rad_table_holdout)
        
        # Combine the tables
        rad_tables_holdout = combine_rad_tables(rad_tables_holdout)
        rad_tables_holdout.Properties['userData']['flags_processing'] = {}

        return rad_tables_holdout
    
    def pre_process_variables(self, ml: Dict, outcome_table_binary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads and pre-processes different radiomics tables from different variable types
        found in the ml dict.
        
        Note: 
            only patients of the training/learning set should be found in this outcome table.

        Args:
            ml (Dict): The machine learning dictionary containing the information of the machine learning test.
            outcome_table_binary (pd.DataFrame): outcome table with binary labels. This table may be used to
                pre-process some variables with the "FDA" feature set reduction algorithm.

        Returns:
            Tuple: Two dict of processed radiomics tables, one dict for training and one for 
                testing (no feature set reduction). 
        """
        # Get a list of unique variables found in the ml variables combinations dict
        variables_id = [s.split('_') for s in ml['variables']['combinations']]
        variables_id = list(set([x for sublist in variables_id for x in sublist]))

        # For each variable, load the corresponding radiomics table and pre-process it
        processed_var_tables, processed_var_tables_test =  {var_id : self.pre_process_radiomics_table(
            ml, 
            var_id, 
            outcome_table_binary
        ) for var_id in variables_id}
        
        return processed_var_tables, processed_var_tables_test

    def pre_process_radiomics_table(
            self, 
            ml: Dict, 
            var_id: str, 
            outcome_table_binary: pd.DataFrame,
            patients_train: list
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        For the given variable, this function loads the corresponding radiomics tables and pre-processes them
        (cleaning, normalization and feature set reduction).

        Note: 
            Only patients of the training/learning set should be found in the given outcome table.
        
        Args:
            ml (Dict): The machine learning dictionary containing the information of the machine learning test 
                (parameters, options, etc.).
            var_id (str): String specifying the ID of the radiomics variable in ml. For example: 'var1'.
            outcome_table_binary (pd.DataFrame): outcome table with binary labels. This table may
                be used to pre-process some variables with the "FDA" feature set reduction algorithm.
            
            patients_train (list): List of patients to use for training.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Two dataframes of processed radiomics tables, one for training 
                and one for testing (no feature set reduction).
        """
        # Initialization
        patient_ids = list(outcome_table_binary.index)
        outcome_table_binary_training = outcome_table_binary.loc[patients_train]
        var_names = ['cleaning_profile', 'normalization', 'reduction_method']
        flags_preprocessing =  {key: key in ml['variables'][var_id].keys() for key in var_names}
        flags_preprocessing_test = flags_preprocessing.copy()
        flags_preprocessing_test['reduction_method'] = False

        # Pre-processing
        rad_var_struct = ml['variables'][var_id]
        rad_tables_learning = list()
        for item in rad_var_struct['path'].values():
            # Loading the table
            path_radiomics_csv = item['csv']
            path_radiomics_txt = item['txt']
            image_type = item['type']
            rad_table_learning = get_radiomics_table(path_radiomics_csv, path_radiomics_txt, image_type, patient_ids)

            # Data cleaning
            if flags_preprocessing['cleaning_profile']:
                cleaning_dict = ml['datacleaning'][ml['variables'][var_id]['cleaning_profile']]['continuous']
                data_cleaner = DataCleaner(**cleaning_dict)

                # Temp save of properties
                temp_properties = deepcopy(rad_table_learning.Properties)

                # Apply data cleaning
                rad_table_learning = data_cleaner.fit_transform(rad_table_learning)

                # Re-assign properties
                rad_table_learning.Properties = temp_properties

                if rad_table_learning is None:
                    continue

            # Normalization (ComBat)
            if flags_preprocessing['normalization']:
                normalization_method = ml['variables'][var_id]['normalization']
                # Some information must be stored to re-apply combat for testing data
                if 'combat' in normalization_method.lower():
                    # Training data
                    rad_table_learning.Properties['userData']['normalization'] = dict()
                    rad_table_learning.Properties['userData']['normalization']['original_data'] = dict()
                    rad_table_learning.Properties['userData']['normalization']['original_data']['path_radiomics_csv'] = path_radiomics_csv
                    rad_table_learning.Properties['userData']['normalization']['original_data']['path_radiomics_txt'] = path_radiomics_txt
                    rad_table_learning.Properties['userData']['normalization']['original_data']['image_type'] = image_type
                    rad_table_learning.Properties['userData']['normalization']['original_data']['patient_ids'] = patient_ids
                    if flags_preprocessing['cleaning_profile']:
                        data_cln_method = ml['variables'][var_id]['cleaning_profile']
                        rad_table_learning.Properties['userData']['normalization']['original_data']['datacleaning_method'] = data_cln_method

                    # Apply ComBat
                    normalization = CombatNormalization()
                    rad_table_learning = normalization.fit_transform(rad_table_learning)  # Training data
                else:
                    raise NotImplementedError(f'Normalization method: {normalization_method} not recognized.')

            # Save the table
            rad_tables_learning.append(rad_table_learning)

        # Seperate training and testing data before feature set reduction
        rad_tables_testing = deepcopy(rad_tables_learning)
        rad_tables_training = []
        for rad_tab in rad_tables_learning:
            patients_ids = intersect(patients_train, list(rad_tab.index))
            rad_tables_training.append(deepcopy(rad_tab.loc[patients_ids]))

        # Deepcopy properties
        temp_properties = list()
        for rad_tab in rad_tables_testing:
            temp_properties.append(deepcopy(rad_tab.Properties))

        # Feature set reduction (for training data only)
        if flags_preprocessing['reduction_method']:
            f_set_reduction_method = ml['variables'][var_id]['reduction_method']
            fsr = FSR(f_set_reduction_method)
            
            # Apply FDA
            rad_tables_training = fsr.apply_fsr(
                ml, 
                rad_tables_training, 
                outcome_table_binary_training, 
                path_save_logging=ml['path_results']
            )

        # Re-assign properties
        for i in range(len(rad_tables_testing)):
            rad_tables_testing[i].Properties = temp_properties[i]
        del temp_properties
        
        # Finalization steps
        rad_tables_training.Properties['userData']['flags_preprocessing'] = flags_preprocessing
        rad_tables_testing = combine_rad_tables(rad_tables_testing)
        rad_tables_testing.Properties['userData']['flags_processing'] = flags_preprocessing_test

        return rad_tables_training, rad_tables_testing

    def ml_run(self, path_ml: Path, holdout_test: bool = True, method: str = 'auto') -> None:
        """
        This function runs the machine learning test for the ceated experiment.

        Args:
            path_ml (Path): Path to the main dictionary containing info about the ml current experiment.
            holdout_test (bool, optional): Boolean specifying if the hold-out test should be performed.
        
        Returns:
            None.
        """
        # Set up logging file for the batch
        log_file = os.path.dirname(path_ml) + '/batch.log'
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s', filemode='w')

        # Start the timer
        batch_start = time.time()

        logging.info("\n\n********************MACHINE LEARNING RUN********************\n\n")

        # --> A. Initialization phase
        # Load the test dictionary and machine learning information
        ml_dict_paths = load_json(path_ml)      # Test information dictionary
        ml_info_dict = self.__load_ml_info(ml_dict_paths)       # Machine learning information dictionary

        # Machine learning assets
        patients_train = ml_info_dict['patientsTrain']
        patients_test = ml_info_dict['patientsTest']
        patients_holdout = load_json(self.path_study / 'patientsHoldOut.json') if holdout_test else None
        outcome_table_binary = ml_info_dict['outcome_table_binary']
        ml = ml_info_dict['ml']
        path_results = ml_info_dict['path_results']
        ml['path_results'] = path_results

        # --> B. Machine Learning phase 
        # B.1. Pre-processing features
        start = time.time()
        logging.info("\n\n--> PRE-PROCESSING TRAINING VARIABLES")

        # Not all variables will be used to train the model, only the user-selected variable
        var_id = str(ml['variables']['varStudy'])

        # Pre-processing of the radiomics tables/variables
        processed_training_table, processed_testing_table = self.pre_process_radiomics_table(
            ml, 
            var_id, 
            outcome_table_binary.copy(),
            patients_train
        )
        logging.info(f"...Done in {time.time()-start} s")

        # B.2. Pre-learning initialization
        # Patient definitions (training and test sets)
        patient_ids = list(outcome_table_binary.index)
        patients_train = intersect(intersect(patient_ids, patients_train), processed_training_table.index)
        patients_test = intersect(intersect(patient_ids, patients_test), processed_testing_table.index)
        patients_holdout = intersect(patient_ids, patients_holdout) if holdout_test else None

        # Initializing outcome tables for training and test sets
        outcome_table_binary_train = outcome_table_binary.loc[patients_train, :]
        outcome_table_binary_test = outcome_table_binary.loc[patients_test, :]
        outcome_table_binary_holdout = outcome_table_binary.loc[patients_holdout, :] if holdout_test else None

        # Serperate variable table for training sets (repetitive but double-checking)
        var_table_train = processed_training_table.loc[patients_train, :]

        # Initializing the model settings
        algorithm = ml['modeling']['method'] if 'method' in ml['modeling'].keys() else method
        var_importance_threshold = ml['modeling']['var_importance_threshold']
        optimize_threshold = ml['modeling']['optimize_threshold']
        optimization_metric = ml['modeling']['optimization_metric']
        method = ml['modeling']['method'] if 'method' in ml['modeling'].keys() else method
        use_gpu = ml['modeling']['useGPU'] if 'useGPU' in ml['modeling'].keys() else True
        seed = ml['modeling']['seed'] if 'seed' in ml['modeling'].keys() else None

        # B.2. Training the model
        tstart = time.time()
        logging.info(f"\n\n--> TRAINING {algorithm.upper()} MODEL FOR VARIABLE {var_id}")

        # Training the model
        estimator = Estimator(
            algorithm=algorithm,
            ml_config={
            'var_importance_threshold': var_importance_threshold,
            'optimize_threshold': optimize_threshold,
            'optimization_metric': optimization_metric,
            'use_gpu': use_gpu,
            'seed': seed
        })
        estimator.fit(var_table_train, outcome_table_binary_train)

        # Saving the trained model using pickle
        name_save_model = ml['modeling']['nameSave'] if 'nameSave' in ml['modeling'].keys() else None
        model_id = name_save_model + '_' + str(ml['variables']['varStudy'])
        path_model = os.path.dirname(path_results) + '/' + (model_id + '.pickle')
        estimator.save(path_model)

        logging.info("{}--> DONE. TOTAL TIME OF LEARNING PROCESS: {:.2f} min".format(" " * 4, (time.time()-tstart) / 60))

        # --> C. Testing phase        
        # C.1. Testing the model and computing model response
        tstart = time.time()
        logging.info(f"\n\n--> TESTING {algorithm.upper()} MODEL FOR VARIABLE {var_id}")

        # Preparing the variable table
        var_table_test = get_ml_test_table(estimator, processed_testing_table)

        # Getting the model response for training and test sets
        response_train = estimator.predict_proba(var_table_test.loc[patients_train, :])
        response_test = estimator.predict_proba(var_table_test.loc[patients_test, :])
        
        logging.info('{}--> DONE. TOTAL TIME OF LEARNING PROCESS: {:.2f}'.format(" " * 4, (time.time() - tstart)/60))
        
        if holdout_test:
            # --> D. Holdoutset testing phase
            # D.1. Prepare holdout test data
            var_table_all_holdout = self.get_hold_out_set_table(ml, var_id, patients_holdout)

            # D.2. Testing the model and computing model response on the holdout set
            tstart = time.time()
            logging.info(f"\n\n--> TESTING {algorithm.upper()} MODEL FOR VARIABLE {var_id} ON THE HOLDOUT SET")

            response_holdout = estimator.predict_proba(var_table_all_holdout.loc[patients_holdout, :])
        
        logging.info('{}--> DONE. TOTAL TIME OF LEARNING PROCESS: {:.2f}'.format(" " * 4, (time.time() - tstart)/60))
        
        # E. Computing performance metrics
        tstart = time.time()

        # Initialize the Results class
        result = Results(estimator.estimator_.model_info_, model_id)
        if holdout_test:
            run_results = result.to_json(
                response_train=response_train, 
                response_test=response_test,
                response_holdout=response_holdout, 
                patients_train=patients_train, 
                patients_test=patients_test, 
                patients_holdout=patients_holdout,
                outcome_table_binary_train=outcome_table_binary_train,
                outcome_table_binary_test=outcome_table_binary_test,
                outcome_table_binary_holdout=outcome_table_binary_holdout
            )
        else:
            run_results = result.to_json(
                response_train=response_train, 
                response_test=response_test,
                patients_train=patients_train, 
                patients_test=patients_test, 
                outcome_table_binary_train=outcome_table_binary_train,
                outcome_table_binary_test=outcome_table_binary_test,
            )

        logging.info('\n\n--> COMPUTING PERFORMANCE METRICS ... Done in {:.2f} sec'.format(time.time()-tstart))
        
        # F. Saving the results dictionary
        save_json(path_results, run_results, cls=NumpyEncoder)

        # Total computing time
        logging.info("\n\n*********************************************************************")
        logging.info('{} TOTAL COMPUTATION TIME: {:.2f} hours'.format(" " * 13, (time.time()-batch_start)/3600))
        logging.info("*********************************************************************")
        
    def run_experiment(self, holdout_test: bool = True, method: str = "pycaret") -> None:
        """
        Run the machine learning experiment for each split/run

        Args:
            holdout_test (bool, optional): Boolean specifying if the hold-out test should be performed.
            method (str, optional): String specifying the method to use to train the model.
                - "pycaret": Use PyCaret to train the model (automatic).
                - "grid_search": Grid search with cross-validation to find the best parameters.
                - "random_search": Random search with cross-validation to find the best parameters.
            
        Returns:
            None
        """
        # Initialize the DesignExperiment class
        experiment = DesignExperiment(self.path_study, self.path_workspace, self.path_settings, self.experiment_label)

        # Generate the machine learning experiment
        path_file_ml_paths = experiment.generate_experiment()

        # Run the different machine learning tests for the experiment
        tests_dict = load_json(path_file_ml_paths) # Tests dictionary
        for run in tests_dict.keys():
            self.ml_run(tests_dict[run], holdout_test, method)
        
        # Average results of the different splits/runs
        average_results(self.path_study / f'learn__{self.experiment_label}', save=True)

        # Analyze the features importance for all the runs
        feature_importance_analysis(self.path_study / f'learn__{self.experiment_label}')
        