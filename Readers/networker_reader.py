import numpy as np
import yaml
import sys
import os
from os.path import dirname, abspath
import logging

class Reader:
    def __init__(self,path):
        self.path = path
        
        self.network_type = None
        self.batch_size = None
        self.learning_rate = None
        self.layers_dict = None

        self.network_parser()

    def set_full_path(self):
        """ Add agent library to the path """
        # Add main directory to the path
        main_dirname = dirname(dirname(abspath(__file__)))
        sys.path.append(main_dirname)
        return main_dirname


    def read_yaml(self,yaml_path):
        """Read yaml file and initate network arch. with hyperparameters

        Arguments:
            yaml_path {string} -- relative path to yaml file
            algorithm_name {string} -- type of the algorithm

        Returns:
            yaml_data {dict} -- Yaml data as dictionary

        """

        yaml_data = {}
        
        # Yaml Full Path
        full_path_dir = self.set_full_path()
        yaml_path =  full_path_dir + '/' + yaml_path
        
        try:
            with open(yaml_path) as yaml_f:
                yaml_data = yaml.load(yaml_f)
        except yaml.parser.ParserError:
            logging.error("Error while reading yaml, check if yaml structure is correct!")

        yaml_data = yaml_data['network_architecture']
        return yaml_data

    def network_parser(self):
        yaml_data = self.read_yaml(self.path)
        self.network_type = yaml_data['network_type']
        self.batch_size = yaml_data['batch_size']
        self.learning_rate = yaml_data['learning_rate']
        self.layers_dict = yaml_data['layers']