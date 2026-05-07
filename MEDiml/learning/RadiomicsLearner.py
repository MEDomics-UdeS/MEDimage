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
                                      find_best_model, get_ml_test_table,
                                      get_radiomics_table, intersect)
from MEDiml.learning.Normalization import CombatNormalization
from MEDiml.learning.Results import Results

from ..utils.json_utils import load_json, save_json


class RadiomicsLearner:
    def __init__(self, path_study: Path, path_workspace: Path, path_settings: Path, experiment_label: str) -> None:
        """
        Constructor of the class DesignExperiment.

        Args:
            path_study (Path): Path to the main study folder where patients partition dictionaries are found.
            path_workspace (Path): Path to the folder where features and outcome files are found.
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
                path_save_logging=ml['path_results'] if 'path_results' in list(ml.keys()) else None
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

    def ml_run(self, path_ml: Path, holdout_test: bool = True, model: str = 'xgboost') -> None:
        """
        This function runs the machine learning test for the ceated experiment.

        Args:
            path_ml (Path): Path to the main dictionary containing info about the ml current experiment.
            holdout_test (bool, optional): Boolean specifying if the hold-out test should be performed.
            model (str, optional): Model for model training. Defaults to 'xgboost'.
                This parameter is used only if the ml dictionary does not contain the "model" key in the "modeling" section.
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
        algorithm = ml['modeling']['method'] if 'method' in ml['modeling'].keys() else model
        var_importance_threshold = ml['modeling']['var_importance_threshold']
        optimize_threshold = ml['modeling']['optimize_threshold']
        optimization_metric = ml['modeling']['optimization_metric']
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

    def train_final_model(self) -> None:
        """
        Train a final model on the full learning set and evaluate on holdout set.

        Returns:
            None
        """
        # Set up logging
        path_learn = self.path_study / f'learn__{self.experiment_label}'
        log_file = path_learn / 'final_model.log'
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s', filemode='w')
        
        batch_start = time.time()
        logging.info("\n\n********************FINAL MODEL TRAINING********************\n\n")
        
        try:
            # --> Phase 1: Find best model from splits
            logging.info("--> PHASE 1: FINDING BEST MODEL FROM SPLITS")
            tstart = time.time()
            best_model, best_results_dict = find_best_model(path_learn, metric='AUC')
            model_name = list(best_results_dict.keys())[0]
            logging.info(f"...Best model found: {model_name}")
            logging.info(f"...Test AUC: {best_results_dict[model_name]['test']['metrics']['AUC']:.4f}")
            logging.info(f"...Done in {time.time()-tstart:.2f} sec")
            
            # --> Phase 2: Load data for full training
            logging.info("\n--> PHASE 2: LOADING DATA FOR FULL TRAINING SET")
            tstart = time.time()
            
            # Load patient lists
            if not (self.path_study / 'patientsLearn.json').exists():
                logging.error("patientsLearn.json not found in study folder")
                raise FileNotFoundError(f"patientsLearn.json not found at {self.path_study}")
            
            patients_final_train = load_json(self.path_study / 'patientsLearn.json')
            
            patients_holdout = None
            if not (self.path_study / 'patientsHoldOut.json').exists():
                logging.warning("patientsHoldOut.json not found. Skipping holdout testing.")
            else:
                patients_holdout = load_json(self.path_study / 'patientsHoldOut.json')
            
            # Load outcomes table
            outcome_table = pd.read_csv(self.path_workspace / 'outcomes.csv', index_col=0)
            outcome_table_binary = outcome_table.iloc[:, [0]]
            
            # Filter to patients in learning set
            all_patients = patients_final_train + (patients_holdout or [])
            all_patients = intersect(all_patients, list(outcome_table_binary.index))
            
            # Get ML configuration from one of the splits
            test_paths = list(path_learn.glob('test__*'))
            if not test_paths:
                logging.error("No test folders found in results folder")
                raise ValueError(f"No test folders found at {path_learn}")
            
            ml_dict_paths = load_json(test_paths[0] / 'paths_ml.json')
            ml = load_json(ml_dict_paths['ml'])
            var_id = str(ml['variables']['varStudy'])
            
            
            logging.info(f"...Variable ID: {var_id}")
            logging.info(f"...Done in {time.time()-tstart:.2f} sec")
            
            # --> Phase 3: Preprocess full training data
            logging.info("\n--> PHASE 3: PREPROCESSING FULL TRAINING DATA")
            tstart = time.time()
            
            # Pre-process the full learning set with FSR re-fitted on all data
            processed_full_train, processed_holdout = self.pre_process_radiomics_table(
                ml,
                var_id,
                outcome_table_binary.copy(),
                all_patients  # All patients
            )
            
            # Get intersection of patients that survived preprocessing
            patients_final_train = intersect(patients_final_train, list(processed_full_train.index))                

            # Filter outcome tables
            outcome_final_train = outcome_table_binary.loc[patients_final_train, :]

            # Apply the process to the holdout set if it exists
            if patients_holdout:
                patients_holdout = intersect(patients_holdout, list(processed_holdout.index))
                outcome_final_holdout = outcome_table_binary.loc[patients_holdout, :]

            logging.info(f"...Learning set size: {len(patients_final_train)} patients")
            logging.info(f"...Holdout set size: {len(patients_holdout)} patients")

            logging.info(f"...Preprocessed training set size: {len(patients_final_train)} patients")
            logging.info(f"...Preprocessed holdout set size: {len(patients_holdout)} patients")
            logging.info(f"...Final feature count: {processed_full_train.shape[1]}")
            logging.info(f"...Done in {time.time()-tstart:.2f} sec")

            # --> Phase 4: Train final model
            logging.info("\n--> PHASE 4: TRAINING FINAL MODEL ON FULL LEARNING SET")
            tstart = time.time()

            # Prepare training data
            var_table_final_train = processed_full_train.loc[patients_final_train, :]

            # Re-train the final model
            best_model.fit(var_table_final_train, outcome_final_train)

            # Save the final model
            name_save_model = ml['modeling']['nameSave'] if 'nameSave' in ml['modeling'].keys() else 'final_model'
            model_id = name_save_model + '_FINAL_' + str(ml['variables']['varStudy'])
            path_final_model = path_learn / f'{model_id}.pickle'
            best_model.save(str(path_final_model))

            logging.info(f"--> DONE. Model saved to {path_final_model}")
            logging.info(f"...Training time: {time.time()-tstart:.2f} sec")

            if patients_holdout:
                # --> Phase 5: Evaluate on holdout set
                logging.info("\n--> PHASE 5: EVALUATING FINAL MODEL ON HOLDOUT SET")
                tstart = time.time()

                # Prepare holdout data with aligned features
                var_table_final_holdout = get_ml_test_table(best_model, processed_holdout)
                var_table_final_holdout = var_table_final_holdout.loc[patients_holdout, :]

                # Generate predictions
                response_holdout = best_model.predict_proba(var_table_final_holdout)

                logging.info(f"...Holdout predictions generated")
                logging.info(f"...Done in {time.time()-tstart:.2f} sec")

                # --> Phase 6: Compute and save results
                logging.info("\n--> PHASE 6: COMPUTING PERFORMANCE METRICS")
                tstart = time.time()

                result = Results(best_model.estimator_.model_info_, model_id)
                final_results = result.to_json(
                    response_holdout=response_holdout,
                    patients_holdout=patients_holdout,
                    outcome_table_binary_holdout=outcome_final_holdout
                )
                
                # Save results
                path_final_results = path_learn / 'final_model_results.json'
                save_json(path_final_results, final_results, cls=NumpyEncoder)
            
                logging.info(f"...Results saved to {path_final_results}")
                logging.info(f"...Holdout AUC: {final_results[model_id]['holdout']['metrics']['AUC']:.4f}")
                logging.info(f"...Done in {time.time()-tstart:.2f} sec")

            # Total computing time
            logging.info("\n\n*********************************************************************")
            logging.info('{} FINAL MODEL TRAINING COMPLETE'.format(" " * 13))
            logging.info('{} TOTAL COMPUTATION TIME: {:.2f} hours'.format(" " * 13, (time.time()-batch_start)/3600))
            logging.info("*********************************************************************")
            
        except Exception as e:
            logging.error(f"\n\nERROR during final model training: {str(e)}", exc_info=True)
            raise ValueError(f"Final model training failed: {str(e)}")
 
    def run_experiment(self, holdout_test: bool = True, model: str = "xgboost", finalize: bool = False) -> None:
        """
        Run the machine learning experiment for each split/run

        Args:
            holdout_test (bool, optional): Boolean specifying if the hold-out test should be performed.
            model (str, optional): String specifying the model to use to train the model.
                - "xgboost": Use XGBoost to train the model.
                - "rf": Use Random Forest to train the model.
            finalize (bool, optional): Boolean specifying if a final model should be trained on the full learning set 
                and tested on the holdout set. The final model will be trained

        Returns:
            None
        """
        # Initialize the DesignExperiment class
        experiment = DesignExperiment(self.path_study, self.path_workspace, self.path_settings, self.experiment_label)

        # Generate the machine learning experiment
        tests_dict = experiment.generate_experiment()

        # Run the different machine learning tests for the experiment
        for run in tests_dict.keys():
            self.ml_run(tests_dict[run], holdout_test, model)
        
        # Average results of the different splits/runs
        average_results(self.path_study / f'learn__{self.experiment_label}', save=True)

        # Analyze the features importance for all the runs
        feature_importance_analysis(self.path_study / f'learn__{self.experiment_label}')

        # Train a final model
        if finalize:
            self.train_final_model()
