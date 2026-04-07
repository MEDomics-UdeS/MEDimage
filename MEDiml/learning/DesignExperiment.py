import platform
import re
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from ..utils.get_institutions_from_ids import get_institutions_from_ids
from ..utils.json_utils import load_json, posix_to_string, save_json
from .ml_utils import cross_validation_split, get_stratified_splits


class DesignExperiment:
    def __init__(self, path_study: Path, path_workspace: Path, path_settings: Path, experiment_label: str) -> None:
        """
        Constructor of the class DesignExperiment.

        Args:
            path_study (Path): Path to the main study folder where the outcomes, 
                learning patients and holdout patients dictionaries are found.
            path_settings (Path): Path to the settings file.
            experiment_label (str): String specifying the label to attach to a given learning experiment in 
                "path_experiments". This label will be attached to the ml__$experiments_label$.json file as well
                as the learn__$experiment_label$ folder. This label is used to keep track of different experiments 
                with different settings (e.g. radiomics, scans, machine learning algorithms, etc.).
        
        Returns:
            None
        """
        self.path_study = Path(path_study)
        self.path_settings = Path(path_settings)
        self.path_workspace = Path(path_workspace)
        self.experiment_label = str(experiment_label)
        self.path_ml_object = None

    def __create_folder_and_content(
            self, 
            path_learn: Path,
            run_name: str, 
            patients_train: List,
            patients_test: List, 
            ml_path: Path
        ) -> List:
        """
        Creates json files needed for a given run
        
        Args:
            path_learn (Path): path to the main learning folder containing information about the training and test set.
            run_name (str): name for a given run.
            patients_train (List): list of patients in the training set.
            patients_test (List): list of patients in the test set.
            ml_path (Path): path to the given run.
        
        Returns:
            List: list of paths to the given run.
        """
        paths_ml = dict()
        path_run = path_learn / run_name
        Path.mkdir(path_run, exist_ok=True)
        path_train = path_run / 'patientsTrain.json'
        path_test = path_run / 'patientsTest.json'
        save_json(path_train, sorted(patients_train))
        save_json(path_test, sorted(patients_test))
        paths_ml['patientsTrain'] = path_train
        paths_ml['patientsTest'] = path_test
        paths_ml['outcomes'] = self.path_workspace / 'outcomes.csv'
        paths_ml['ml'] = self.path_ml_object
        paths_ml['results'] = path_run / 'run_results.json'
        path_file = path_run / 'paths_ml.json'
        paths_ml = posix_to_string(paths_ml)
        ml_path.append(path_file)
        save_json(path_file, paths_ml)

        return ml_path
    
    def __load_config(self) -> Dict:
        """Loads the YAML master configuration file."""
        with open(self.path_settings, 'r') as file:
            return yaml.safe_load(file)

    def __get_learning_dict(self) -> Path:
        """
        Generates a dictionary containing all settings for the learning experiment
        using a single YAML master configuration.
        
        Returns:
            Path: Path to the saved experiment-specific YAML options.
        """
        # Safety check: Verify master config exists
        if not self.path_settings.exists():
            raise FileNotFoundError(
                f"Master configuration file not found at: {self.path_settings}. "
                "Please ensure the consolidated YAML is in the settings folder."
            )

        # Load the Master Config
        config = self.__load_config()

        # Assemble Experiment Metadata
        # We maintain the structure while adding run-specific info
        ml_options = {
            'os': platform.system(),
            'experiment_label': self.experiment_label,
            'config_source': str(self.path_settings),
            # Directly map the sections from the YAML for downstream use
            'design': config.get('design'),
            'variables': config.get('variables'),
            'datacleaning': config.get('data_cleaning'),
            'fSetReduction': config.get('feature_reduction'),
            'normalization': config.get('normalization'),
            'modeling': config.get('modeling'),
            'study_metadata': config.get('study_metadata')
        }

        # Experiment Label Safety Check
        if not self.experiment_label:
            raise ValueError("Experiment label is empty. Class was not initialized properly.")

        return ml_options

    def __fill_learner_dict(self) -> Path:
        """
        Fills the main expirement dictionary from the settings in the different json files. 
        This main dictionary will hold all the settings for the data processing and learning experiment.
        
        Returns:
            Path: Path to the learner object.
        """
        # Initialization
        all_datacleaning = list()
        all_normalization = list()
        all_fset_reduction = list()
        ml = self.__get_learning_dict()

        # Machine learning variables
        if 'variables' in list(ml.keys()):
            var_options = ml['variables']
            fields = list(var_options.keys())
            vars = [(idx, s) for idx, s in enumerate(fields) if re.match(r"^var[0-9]{1,}$", s)]
            var_names = [var[1] for var in vars]  # list of var names

            # For each variable, organize the option in the ML dictionary
            for (idx, var) in vars:
                vars_dict = ml['variables']
                var_struct = vars_dict[var]
                
                # Radiomics variables
                if 'radiomics' in var_struct['nameType'].lower():
                    # Get radiomics features in workspace
                    if 'settofeatures' in var_struct['path'].lower():
                        name_folder = re.match(r"setTo(.*)inWorkspace", var_struct['path']).group(1)
                        path_features = self.path_workspace / name_folder
                    # Get radiomics features in path provided in the dictionary by the user 
                    else:
                        path_features = var_struct['path']
                    scans = var_struct['scans'] # list of imaging sequences
                    rois = var_struct['rois'] # list of roi labels
                    im_spaces = var_struct['imSpaces'] # list of image spaces (filterd and original)
                    use_combinations = var_struct['use_combinations'] if 'use_combinations' in list(var_struct.keys()) else False # boolean to use combinations of scans and im_spaces
                    if use_combinations:
                        all_combinations = []
                        scans = list(var_struct['combinations'].keys())
                        for scan in scans:
                            im_spaces = list(var_struct['combinations'][scan])
                            all_combinations += list(product([scan], rois, im_spaces))
                    else:
                        all_combinations = list(product(scans, rois, im_spaces))
                    
                    # Initialize dict to hold all paths to radiomics features (csv and txt files)
                    path = dict() 
                    for idx, (scan, roi, im_space) in enumerate(all_combinations):
                        rad_tab_x = {}
                        name_tab = 'radTab' + str(idx+1)
                        radiomics_table_name = 'radiomics__' + scan + '(' + roi + ')__' + im_space
                        rad_tab_x['csv'] = path_features / (radiomics_table_name + '.csv')
                        rad_tab_x['txt'] = path_features / (radiomics_table_name + '.txt')
                        rad_tab_x['type'] = path_features / (scan + '(' + roi + ')__' + im_space)
                        
                        # check if file exist
                        if not rad_tab_x['csv'].exists():
                            raise FileNotFoundError(f"File {rad_tab_x['csv']} does not exist.")
                        if not rad_tab_x['txt'].exists():
                            raise FileNotFoundError(f"File {rad_tab_x['txt']} does not exist.")
                        
                        path[name_tab] = rad_tab_x
                    
                    # Add path to ml dict for the current variable
                    vars_dict[var]['path'] = path
                    
                    # Add to ml dict for the current variable
                    ml['variables'].update(vars_dict)
                
                # Clinical or other variables (For ex: Volume)
                else:
                    # get path to csv file of features
                    if not var_struct['path']:
                        if var_options['pathCSV'] == 'setToCSVinWorkspace':
                            path_csv = self.path_workspace / 'CSV'
                        else:
                            path_csv = var_options['pathCSV']
                        var_struct['path'] = path_csv / var_struct['nameFile']
                
                # Add to ml dict for the current variable
                ml['variables'].update(vars_dict)

                # Initialize data processing methods
                if 'cleaning_profile' in var_struct.keys():
                    all_datacleaning.append(var_struct['cleaning_profile'])
                if 'normalization' in var_struct.keys():
                    all_normalization.append((var_struct['normalization']))
                if 'reduction_method' in var_struct.keys():
                    all_fset_reduction.append(var_struct['reduction_method'])

            # Combinations of variables
            if 'combinations' in var_options.keys():
                if var_options['combinations'] == ['all']:  # Combine all variables
                    combs = [comb for i in range(len(vars)) for comb in combinations(var_names, i+1)]
                    combstrings = ['_'.join(elt) for elt in combs]
                    ml['variables']['combinations'] = combstrings

        # Save the ML dictionary
        if self.experiment_label == "":
            raise ValueError("Experiment label is empty. Class was not initialized properly.")
        path_ml_object = self.path_study / f'ml_test__{self.experiment_label}.json'
        ml = posix_to_string(ml)    # Convert all paths to string
        save_json(path_ml_object, ml)
        
        # return ml
        return path_ml_object

    def create_experiment(self, ml: dict = None) -> Dict:
        """
        Create the machine learning experiment dictionary, organizes each test/split information in a seperate folder.

        Args:
            ml (dict, optional): Dictionary containing all the machine learning settings. Defaults to None.
        
        Returns:
            Dict: Dictionary containing all the organized machine learning settings.
        """
        # Initialization
        ml_path = list()
        ml = load_json(self.path_ml_object) if ml is None else ml

        # Learning set
        patients_learn = load_json(self.path_study / 'patientsLearn.json')
        
        # Outcomes table
        outcomes_table = pd.read_csv(self.path_workspace / 'outcomes.csv', index_col=0)

        # keep only patients in learn set and outcomes table
        patients_to_keep = list(filter(lambda x: x in patients_learn, outcomes_table.index.values.tolist()))
        outcomes_table = outcomes_table.loc[patients_to_keep]

        # Get the "experiment label" from ml__$experiment_label$.json
        if self.experiment_label:
            experiment_label = self.experiment_label
        else:
            experiment_label = Path(self.path_ml_object).name[4:-5]

        # Create the folder for the training and testing sets (machine learning) information
        name_learn = 'learn__' + experiment_label
        path_learn = Path(self.path_study) / name_learn
        Path.mkdir(path_learn, exist_ok=True)

        # Getting the type of test_sets
        test_sets_types = ml['design']['active_method']

        # Creating the sets for the different machine learning runs
        for type_set in test_sets_types:
            # Random splits
            if type_set.lower() == 'random':
                # Get the experiment options for the sets
                random_info = ml['design'][type_set]
                method = random_info['method']
                n_splits = random_info['nSplits']
                stratify_institutions = random_info['stratifyInstitutions']
                test_proportion = random_info['testProportion']
                seed = random_info['seed']
                if method == 'SubSampling':
                    # Get the training and testing sets
                    patients_train, patients_test = get_stratified_splits(
                        outcomes_table, n_splits, 
                        test_proportion, seed, 
                        stratify_institutions
                    )

                    # If patients are not in a list
                    if type(patients_train) != list and not hasattr((patients_train), "__len__"):
                        patients_train = [patients_train]
                        patients_test = [patients_test]
                    
                    for i in range(n_splits):
                        # Create a folder for each split/run
                        run_name = "test__{0:03}".format(i+1)
                        ml_path = self.__create_folder_and_content(
                            path_learn, 
                            run_name, 
                            patients_train[i], 
                            patients_test[i], 
                            ml_path
                        )
            # Institutions-based splits
            elif type_set.lower() == 'institutions':
                # Get institutions run info
                patient_ids = pd.Series(outcomes_table.index)
                institution_cat_vector = get_institutions_from_ids(patient_ids)
                institution_cats = list(set(institution_cat_vector))
                n_institution = len(institution_cats)
                # The 'Institutions' argument only make sense if n_institutions > 1
                if n_institution > 1:
                    for i in range(n_institution):
                        cat = institution_cats[i]
                        patients_train = [elt for elt in patient_ids if cat not in elt]
                        patients_test = [elt for elt in patient_ids if cat in elt]
                        run_name = f"test__{cat}"
                        # Create a folder for each split/run
                        ml_path = self.__create_folder_and_content(
                            path_learn,
                            run_name,
                            patients_train,
                            patients_test,
                            ml_path
                        )
                    if n_institution > 2:
                        size_inst = list()
                        for i in range(n_institution):
                            cat = institution_cats[i]
                            size_inst.append(sum([1 if cat in elt else 0 for elt in institution_cat_vector]))
                        ind_max = size_inst.index(max(size_inst))
                        str_test = list()
                        for i in range(n_institution):
                            if i != ind_max:
                                cat = institution_cats[i]
                                str_test.append(cat)
                        cat = institution_cats[ind_max]
                        patients_train = [elt for elt in patient_ids if cat in elt]
                        patients_test = [elt for elt in patient_ids if cat not in elt]
                        run_name = f"test__{'_'.join(str_test)}"
                        # Create a folder for each split/run
                        ml_path = self.__create_folder_and_content(
                            path_learn,
                            run_name,
                            patients_train,
                            patients_test,
                            ml_path
                        )
            elif type_set.lower() == 'cv':
                # Get the experiment options for the sets
                cv_info = ml['design'][type_set]
                n_folds = cv_info['nFolds']
                seed = cv_info['seed']

                # Get the training and testing sets
                patients_train, patients_test = cross_validation_split(
                    outcomes_table,
                    n_folds,
                    seed=seed
                )

                # If patients are not in a list
                if type(patients_train) != list and not hasattr((patients_train), "__len__"):
                    patients_train = [patients_train]
                    patients_test = [patients_test]

                for i in range(n_folds):
                    # Create a folder for each split/run
                    run_name = "test__{0:03}".format(i+1)
                    ml_path = self.__create_folder_and_content(
                        path_learn,
                        run_name,
                        patients_train[i],
                        patients_test[i],
                        ml_path
                    )
            else:
                raise ValueError("The type of test set is not recognized. Must be 'random' or 'institutions'.")

        # Make ml_path a dictionary to easily save it in json
        return {f"run{idx+1}": value for idx, value in enumerate(ml_path)}

    def generate_experiment(self):
        """
        Generate the json files containing all the options the experiment.
        The json files will then be used in machine learning.
        """
        # Fill the ml options dictionary
        self.path_ml_object = self.__fill_learner_dict()
        
        # Generate the experiment dictionary
        experiment_dict = self.create_experiment()
        
        # Saving the final experiment dictionary
        path_file = self.path_study / f'path_file_ml_paths__{self.experiment_label}.json'
        experiment_dict = posix_to_string(experiment_dict)  # Convert all paths to string
        save_json(path_file, experiment_dict)
        
        return path_file
