import xarray as xr
import numpy as np
from scipy.stats import linregress as linreg
from typing import Optional
from loguru import logger

import os

import random

from scipy.optimize import curve_fit


from collections import defaultdict

from datetime import datetime

import json


## TODO: decide whether to store the results in the class or return them
## need to decide how to store the results, e.g. in a dictionary or as attributes
## test how big the results would be -- could make it an option to store them or return them
## if returning, change class to also not store the input data, but only contain parameters and methods
## if storing results, store region and time bounds as attributes as well


class Multifractals:
    def __init__(self,
                 input_dataset: xr.Dataset,
                 input_field: str,
                 reso: float,
                 radii: np.ndarray,
                 fitting_range: tuple[int, int],
                 Q: np.ndarray = np.arange(1,11),
                 time_coord_name: str = 'time',
                ):
        
        self.input_dataset = input_dataset # input dataset containing the field to be analyzed
        self.input_field = input_field # name of the field to be analyzed
        self.reso = reso # resolution of the input fields in degrees
        self.radii = radii # list of radii (pixel distances) to be used in the analysis
        self.fitting_range = fitting_range
        self.Q = Q # moments of orders Q to be calculated
        self.time_coord_name = time_coord_name # coordinate name of time coordinate in dataset
        

    def _calculate_moments(self, input_snapshot: np.ndarray):
        """
        Method to calculate the moments of the input snapshot for the given radii and Q values.

        Parameters
        ----------

        input_snapshot : np.ndarray
            2D array containing the input snapshot to be analyzed

        Returns
        -------
        moments : np.ndarray
            2D array containing the moments of the input snapshot for the given radii and Q values
        """
        moments = np.zeros([len(self.radii), len(self.Q)])
        input_snapshot = np.squeeze(input_snapshot)
        assert len(input_snapshot.shape) == 2, "Squeezed input snapshot must be a 2-dimensional field."
        for r, r_idx in zip(self.radii, range(len(self.radii))):
            # ignore edge as it contains pixels from the other side of the array
            shifted = np.roll(input_snapshot, -r, axis=1)[:,:-r] # [:,-r:] are the pixels that wrapped around
            deltas = np.abs(input_snapshot[:,:-r] - shifted) # take same size cloudfield containing pixels we want to compare to shifted 
            moments[r_idx, :] = [np.nanmean(deltas**q) for q in self.Q]

        return moments
        
    def _calculate_zetas(self, moments: np.ndarray):
        """
        Method to calculate the zeta values for the given moments.

        Parameters
        ----------
        moments : np.ndarray
            2D array containing the moments of the input snapshot for the given radii and Q values

        Returns
        -------
        zetas : np.ndarray
            1D array containing the zeta values for the given moments
        """

        zetas = np.zeros(len(self.Q)+1)
        logR = np.log(self.radii)

        radii_in_fitting_range = np.where((self.radii >= self.fitting_range[0]) & (self.radii <= self.fitting_range[1]))[0]

        # TODO: move warning code to separate function and make it more robust
        warned=False
        for q in self.Q:
            normalised_moment_q = moments[:, q-1] / moments[0, q-1]
            logNormMoment = np.log(normalised_moment_q)

            a = linreg(logR[radii_in_fitting_range], logNormMoment[radii_in_fitting_range])[0]
            zetas[q] = a
            
            # can use below to check if fitting range is still good (it is approx. linear in fitting range)
            end = logNormMoment[radii_in_fitting_range[-1]]
            start = logNormMoment[radii_in_fitting_range[0]]
            predend = start + zetas[q] * (logR[radii_in_fitting_range[-1]] - logR[radii_in_fitting_range[0]])
            if not np.isclose(end, predend, rtol=0.01) and not warned:
                logger.warning(f'log-log moments are not linear in fitting range: log(moment) ends at {end}, the line ends at {predend}')
                warned=True
        return zetas

    def _calculate_multifractals(self, input_snapshot: np.ndarray):
        """
        Method to calculate the moments and zetas for the input snapshot.

        Parameters
        ----------
        input_snapshot : np.ndarray
            2D array containing the input snapshot to be analyzed

        Returns
        -------
        moments : np.ndarray
            2D array containing the moments of the input snapshot for the given radii and Q values

        zetas : np.ndarray
            1D array containing the zeta values for the given moments        
        """
        moments = self._calculate_moments(input_snapshot)
        zetas = self._calculate_zetas(moments)
        return moments, zetas


    def multifractal_analysis(self,
                              latitude_bounds: Optional[tuple[int, int]] = None,
                              longitude_bounds: Optional[tuple[int, int]] = None,
                              time_bounds: Optional[tuple[int, int]] = None):

        """
        Method to perform multifractal analysis on the input field

        Parameters
        ----------
        latitude_bounds : tuple[int, int], optional
            Tuple containing the minimum and maximum latitude values to be analyzed, by default None

        longitude_bounds : tuple[int, int], optional
            Tuple containing the minimum and maximum longitude values to be analyzed, by default None

        time_bounds : tuple[int, int], optional
            Tuple containing the minimum and maximum time values to be analyzed, by default None
        """

        # Method to perform multifractal analysis on the input field
        moments = {}
        zetas = {}
        # subset the input dataset to the region of interest
        input_snapshots = self.input_dataset[self.input_field]
        
        # TODO: make independent of variables named lon or longitude!
        if latitude_bounds is not None: 
            input_snapshots = input_snapshots.sel(lat=slice(latitude_bounds[0], latitude_bounds[1]))
        if longitude_bounds is not None: 
            input_snapshots = input_snapshots.sel(lon=slice(longitude_bounds[0], longitude_bounds[1]))
        if time_bounds is not None:
            # TODO: make independent of time bounds (using where x >= b[0] and t <= b[1])
            input_snapshots = input_snapshots.sel(time=slice(time_bounds[0], time_bounds[1]))
        
        # loop over times to calculate the multifractal parameters
        # for t in input_snapshots.time:
        for t in input_snapshots[self.time_coord_name].values:
            # shallow_conv_4x4deg.where(shallow_conv_4x4deg[time_coord_name] == t, drop=True))
            input_snapshot = input_snapshots.where(input_snapshots[self.time_coord_name]==t, drop=True).values
            # t_key = np.datetime_as_string(t.values, unit='s')
            if isinstance(t, np.datetime64):
                t_key = np.datetime_as_string(t, unit='s')
            elif type(t) == str:
                t_key = t
            else:
                logger.error(f"time coordinate type is {type(t)} but must be 'str' or 'np.datetime'")
            moments[t_key], zetas[t_key] = self._calculate_multifractals(input_snapshot)
        
        # TODO: decide whether to store the results in the class or return them
        # TODO: how to handle multiple regions?? need to create new class every time if storing results in class
        self.moments = moments
        self.zetas = zetas

    def save_multifractal_analysis_results(self,
                                           save_dir: str,
                                           save_as: str):
        """
        Method to save the moments and zetas as .json or .txt files

        Parameters
        ----------
        save_dir : str
            Directory where the results should be saved

        save_as : str
            Format in which the results should be saved, either 'json' or 'txt'
        """

        assert hasattr(self, 'moments') and hasattr(self, 'zetas'), "call multifractal_analysis before calling this function"

        if not os.path.isdir(save_dir):
            logger.error(f'the specified save directory "{save_dir}" does not exist')
    
        if save_as == 'json':
            logger.info('saving moments and zetas as .json files')
            with open(os.path.join(save_dir, "moments.json"), "w") as moments_file:
                json.dump(self.moments, moments_file)
            with open(os.path.join(save_dir, "zetas.json"), "w") as zetas_file:
                json.dump(self.zetas, zetas_file)
        elif save_as == 'txt':
            logger.info('saving moments and zetas as separate .txt files for each time step')
            for t_key, moments in self.moments:
                np.savetxt(os.path.join(save_dir, t_key + 'moments.txt'), moments)
            for t_key, zetas in self.zetas:
                np.savetxt(os.path.join(save_dir, t_key + 'zetas.txt'), zetas)
        else: 
            logger.error(f'save_as is {save_as} but must be one of [json, txts], others not yet implemented.')
        

    def _zeta_func(self, q, a, zeta_inf):
        zeta_q = a * q / (1 + a * q / zeta_inf)
        return zeta_q

    def _compute_multifractal_parameters(self,
                                         zetas: dict):
        parameters = {}
        Q_for_param_computation = np.insert(self.Q, 0, 0)
        for t_key, zeta_values in zetas.items():
            t_par, _ = curve_fit(self._zeta_func, Q_for_param_computation, zeta_values)
            parameters[t_key] = {'a': t_par[0], 'zeta_infinity': t_par[1]}

        # np.insert(IFSMultifractals.Q, 0, 0)
        return parameters
    
    def get_multifractal_parameters(self):
        assert hasattr(self, 'zetas'), "Zetas not yet computed, call multifractal_analysis before calling this function"

        if not hasattr(self, 'multifractal_parameters'):
            self.multifractal_parameters = self._compute_multifractal_parameters(self.zetas)
        return self.multifractal_parameters

    def compute_average(self):
        # Method to compute the average multifractal spectrum
        assert hasattr(self, 'moments') and hasattr(self, 'zetas'), "call multifractal_analysis before calling this function"
        
        self.mean_zetas = np.array(list(self.zetas.values())).mean(axis=0)
        self.mean_moments = np.array(list(self.moments.values())).mean(axis=0)


    def compute_diurnal_parameters(self, 
                                   bootstrap: bool = False,
                                   n_bootstrap: int = 1000,
                                   bootstrap_confidence_percentage: float = 95):
        # Method to compute diurnal cycle of multifractal parameters
        assert hasattr(self, 'moments'), "call multifractal_analysis before calling this function"
        
        hourly_moments = defaultdict(list)
        
        for key, moments in self.moments.items():
            hour = datetime.strptime(key, '%Y-%m-%dT%H:%M:%S').hour
            hourly_moments[hour].append(moments)
        
        self.diurnal_moments = {hour: np.mean(moments_list, axis=0) for hour, moments_list in hourly_moments.items()}
        self.diurnal_zetas = {hour: self._calculate_zetas(moments) for hour, moments in self.diurnal_moments.items()}
        
        self.diurnal_parameters = self._compute_multifractal_parameters(self.diurnal_zetas)
        
        if bootstrap:
            # computing bootstrapping confidence intervals around diurnal parameters
            bounds_percentage = (100 - bootstrap_confidence_percentage) / 2
            left_bound_idx = int(n_bootstrap * bounds_percentage / 100)
            right_bound_idx = int(n_bootstrap * (100 - bounds_percentage) / 100)

            diurnal_a_bounds = {}
            diurnal_zi_bounds = {}

            # generate n_bootstrap different samples of length n from moments in hourly_moments
            # store parameters a and zeta_infinity for the mean moment of each sample in sample_a and sample_zi
            for hour, moments in hourly_moments.items():
                n = len(moments)
                sample_a = []
                sample_zi = []
                for i in range(n_bootstrap):
                    sample_moments = random.choices(moments, k=n)
                    avg_moments = np.mean(sample_moments, axis=0)
                    zetas_dict = {'avg': self._calculate_zetas(avg_moments)}
                    parameters = self._compute_multifractal_parameters(zetas_dict)
                    sample_a.append(parameters['avg']['a'])
                    sample_zi.append(parameters['avg']['zeta_infinity'])

                sample_a.sort()
                sample_zi.sort()
                
                diurnal_a_bounds[hour] = (sample_a[left_bound_idx], sample_a[right_bound_idx])
                diurnal_zi_bounds[hour] = (sample_zi[left_bound_idx], sample_zi[right_bound_idx])
                
            self.diurnal_a_bounds = diurnal_a_bounds
            self.diurnal_zi_bounds = diurnal_zi_bounds


    def save_diurnal_parameters(self,
                                save_dir: str):
        assert hasattr(self, 'diurnal_parameters'), "call compute_diurnal_parameters before calling this function"

        if not os.path.isdir(save_dir):
            logger.error(f'the specified save directory "{save_dir}" does not exist')
        
        logger.info('saving diurnal parameters as .json file')
        with open(os.path.join(save_dir, "diurnal_parameters.json"), "w") as dp_file:
            json.dump(self.diurnal_parameters, dp_file)

        if hasattr(self, 'diurnal_a_bounds'):
            with open(os.path.join(save_dir, "diurnal_a_bounds.json"), "w") as da_file:
                json.dump(self.diurnal_a_bounds, da_file)

        if hasattr(self, 'diurnal_zi_bounds'):
            with open(os.path.join(save_dir, "diurnal_zi_bounds.json"), "w") as dzi_file:
                json.dump(self.diurnal_zi_bounds, dzi_file)