#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate spectra from QM data.

Created on Thu Sep  8 18:54:21 2022

Complete rewrite on Apr 2024 By Raul Losantos

Added multifile support on Aug 2024 By Raul Losantos

Added individual support on May 2025 By Raul Losantos

@author: Raul Losantos
@author: Alejandro Jodra
"""

import sys
import os
import argparse
import math
import time
import cclib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class ConfigParse():
    """Setting up all needed parameters.

    Configure all needed variables and parse and get QM data from files
    """

    def __init__(self):
        self.initial_wave = 100
        self.final_wave = 700
        self.all_energy = None
        self.all_fosc = None
        self.all_valuesy = None
        self.all_sumvaluesy = None
        self.data = None
        self.debug = None
        self.energ = None
        self.fc = 0
        self.file_count = 1
        self.FILEdata = None
        self.files = None
        self.fos = None
        self.metadata = None
        self.multi = None
        self.n_data_sets = None
        self.n_files = None
        self.n_states = 10
        self.NAMES = None
        self.normalized_spectra = None
        self.plot_single_f = None
        self.plot_single_fn = None
        self.qm_decimals = None
        self.sigma = 0.3
        self.software_qm = None
        self.TPA = None
        self.software_qm = None
        self.qm_decimals = None
        self.wavelengths = None
        self.weight_fn = None

    def parsecl(self):
        """Parse command line arguments."""
        desc = '''This tool Simulates UV-Vis spectra from QM data
        using pseudo Voigt function convolution'''
        parser = argparse.ArgumentParser(
            description=desc, formatter_class=argparse.RawTextHelpFormatter)
        group = parser.add_mutually_exclusive_group()
        parser.add_argument("-n", "--name",
                            dest="NAME",
                            required=False,
                            type=str,
                            help="Name for final output")
        parser.add_argument("-c", "--conf",
                            dest="dconf",
                            required=False,
                            type=str,
                            help="""Path of the configuration config file.
    Contains: Half-width in eV                       0.3
              Number of states to consider Nroots    15
              Starting and Final wavelenght to plot  200 600""")
        parser.add_argument("-d", "--data",
                            dest="FILEdata",
                            required=False,
                            type=str,
                            help="Raw file containing UV-Vis data En(eV) f")
        parser.add_argument(dest="QMdata",
                            nargs='*',
                            type=str,
                            help="Path of the QM data file/files")
        parser.add_argument("-m", "--multi",
                            dest="MULTI",
                            required=False,
                            action='store_false',
                            help="Use Multiple Files combined, e.g. Wigner Distribution")
        group.add_argument("-G", "--GAU",
                           dest="GAU",
                           required=False,
                           action='store_true',
                           help="Use Gaussian functions")
        group.add_argument("-P", "--PSVoigt",
                           dest="weight_fn",
                           required=False,
                           default=None,
                           type=float,
                           help="Use Gau.& Loren.weighted, Pseudo-Voigt model")
        group.add_argument("-L", "--LOREN",
                           dest="LOREN",
                           required=False,
                           action='store_true',
                           help="Use Lorentzian functions")
        parser.add_argument("-p", "--plot",
                            choices=('all', 'func', 'fosc', 'none'),
                            default='none',
                            dest="plotlevel",
                            required=False,
                            help="""Easy selection of plotting level
    all  = all functions and fosc bars
    func = only all functions, for a desired number use --funct
    fosc = only all oscillator strength bars ploted, use --fosc
    none = plot only convoluted spectra, no bars or functions""")
        parser.add_argument("--funct",
                            dest="plot_single_fn",
                            required=False,
                            type=int,
                            help="Number of individual functions ploted")
        parser.add_argument("--fosc",
                            dest="plot_single_f",
                            required=False,
                            type=int,
                            help="Number of oscillator strength bars ploted")
        parser.add_argument("--TPA",
                            dest="TPA",
                            required=False,
                            action='store_true',
                            help="Read Dalton TPA input")
        parser.add_argument("-v", "--verbose",
                            dest="debug",
                            required=False,
                            action='store_true',
                            help="More verbose output")
        return parser.parse_args()

    def ass_vars(self, args):
        """Define all needed variables in a dictionary."""
        # Define input and QMdata files
        if args.dconf:
            try:
                with open(args.dconf, encoding="utf-8") as dconf:
                    lines = list(dconf.read().split())
                    self.sigma = float(lines[0])
                    self.n_states = int(lines[1])
                    self.initial_wave = float(lines[2])
                    self.final_wave = float(lines[3])
            except FileNotFoundError:
                pass
        self.wavelengths = np.arange(self.initial_wave, self.final_wave+1., 1.)

        if args.QMdata:
            self.files = args.QMdata
            self.n_files = len(self.files)

        # Using QM data from text file
        if args.FILEdata:
            self.FILEdata = args.FILEdata
            self.n_files = ConfigParse.process_excited_state_file(
                self, self.FILEdata)
            self.files = [f"TMP_{i}.dat" for i in range(self.n_files)]
            self.qm_decimals = int(4)
        else:
            self.FILEdata = None

        # Set TPA
        if args.TPA:
            self.TPA = args.TPA

        # Set debug
        if args.debug:
            self.debug = args.debug
            sys.stdout = sys.__stdout__
        else:
            sys.stdout = open(os.devnull, 'w')

        # Use multiple independent files
        if args.MULTI:
            self.multi = args.MULTI
        else:
            self.multi = ""

        # Define NAMES as output name
        if args.NAME and not self.multi:
            self.NAMES = [args.NAME]
        else:
            self.NAMES = [os.path.splitext(os.path.basename(
                self.files[i]))[0] for i in range(self.n_files)]
            if not self.multi:
                self.NAMES = [os.path.splitext(
                    os.path.basename(self.files[0]))[0]]

        # Parse qm software
        ConfigParse.parse_qm_software(self, self.files[0], )

        # Define weights for Gaussian and Lorentzian functions,
        # using a Pseudo Voigt description
        for _ in range(1):
            if args.weight_fn is None:
                self.weight_fn = args.weight_fn
                if args.GAU:
                    # Gaussian Function
                    self.weight_fn = float(1.0)
                elif args.LOREN:
                    # Lorentzian Function
                    self.weight_fn = float(0.0)
                    break
                if not self.weight_fn:
                    self.weight_fn = float(1.0)
            else:
                if args.weight_fn >= 0 and args.weight_fn <= 1:
                    self.weight_fn = args.weight_fn
                else:
                    self.weight_fn = float(1.0)

        # Define plotting parameters.
        if args.plotlevel:
            self.plotlevel = args.plotlevel

            if self.plotlevel == "all":
                self.plot_single_fn = self.n_states
                self.plot_single_f = self.n_states

            if self.plotlevel == "func":
                self.plot_single_fn = self.n_states
                if args.plot_single_f:
                    self.plot_single_f = args.plot_single_f
                else:
                    self.plot_single_f = int(0)

            if self.plotlevel == "fosc":
                self.plot_single_f = self.n_states
                if args.plot_single_fn:
                    self.plot_single_fn = args.plot_single_fn
                else:
                    self.plot_single_fn = int(0)

            if self.plotlevel == "none":
                if args.plot_single_fn:
                    self.plot_single_fn = args.plot_single_fn
                else:
                    self.plot_single_fn = int(0)
                if args.plot_single_f:
                    self.plot_single_f = args.plot_single_f
                else:
                    self.plot_single_f = int(0)

    def parse_qm_software(self, file):
        """Parse type of QM files and it decimals."""
        if self.FILEdata is not None:
            self.energ, self.fos = ConfigParse.read_file_exc(self, file)
            self.software_qm = None
        else:
            with open(file, 'r', encoding="utf-8") as data:
                for line in data:  # Iterate over 'data' not 'file'
                    if 'Entering Gaussian System, Link 0' in line:
                        self.software_qm = "GAUSSIAN"
                        self.qm_decimals = int(4)
                        self.n_data_sets = 1
                        return
                    if 'OpenMolcas is free software' in line:
                        self.software_qm = "MOLCAS"
                        self.qm_decimals = int(8)
                        self.n_data_sets = 1
                        return
                    if 'ORCA' in line:
                        self.software_qm = "ORCA"
                        self.qm_decimals = int(9)
                        self.n_data_sets = 1
                        return
                    if 'TeraChem' in line:
                        self.software_qm = "TERACHEM"
                        self.qm_decimals = int(4)
                        self.n_data_sets = 1
                        return
                    if 'DALTON' in line:
                        self.software_qm = "DALTON"
                        self.qm_decimals = int(2)
                        self.n_data_sets = 1
                        return
                print("""
The QM software that you would like to use is not supported""")
                sys.exit(1)

    def get_qm_data(self, file):
        """Parse type of QM files and extract Excited State data."""
        parsers = {
            "GAUSSIAN": ConfigParse.read_gaussian_exc,
            "ORCA": ConfigParse.read_orca_exc,
            "MOLCAS": ConfigParse.read_molcas_exc,
            "TERACHEM": ConfigParse.read_terachem_exc,
            "DALTON": ConfigParse.read_dalton_tpa_exc
        }

        if self.FILEdata is not None:
            energy, f = ConfigParse.read_file_exc(self, file)
        else:
            for software, parser in parsers.items():
                if software == self.software_qm:
                    energy, f = parser(self, file)
        try:
            self.metadata = self.data.metadata['functional'] \
                + "_" + self.data.metadata.get('basis_set', 'ECP')
        except KeyError:
            self.metadata = self.data.metadata.get('excited_states_method', '')\
                + "_" + self.data.metadata.get('basis_set', self.software_qm)
        except AttributeError:
            self.metadata = self.software_qm
        return energy, f

    def process_excited_state_file(self, file_data):
        """
        Split an "energy(eV)""oscillator strength" file depending on _n_states.

        Parameters
        ----------
        file : STR
            Raw UV-Vis data file.

        Returns
        -------
        n_files: Splitted temporary files.

        """
        count = 0
        with open(file_data, "r", encoding="utf-8") as data:
            for count, line in enumerate(data):
                pass
        nlines = count + 1
        n_files = int(nlines/self.n_states)
        splitfile = None
        with open(file_data, encoding="utf-8") as data:
            for lineno, line in enumerate(data):
                if lineno % self.n_states == 0:
                    if splitfile:
                        splitfile.close()
                    split_filename = f'TMP_{int(lineno/self.n_states)}.dat'
                    splitfile = open(split_filename, "w", encoding="utf-8")
                splitfile.write(line)
            if splitfile:
                splitfile.close()
        return n_files

    def read_file_exc(self, file):
        """
        Read "energy (eV)" "oscillator strength" file/s.

        Parameters
        ----------
        file : STR
            Splitted temporary file.

        Returns
        -------
        energy Excitation energies.
        f      Oscillator strength.
        """
        with open(file, encoding="utf-8") as data:
            lines = list(data.read().split())
            ev_en = [float(lines[i*2]) for i in range(self.n_states)]
            energy = [float(1239.841/ev_en[i]) for i in range(self.n_states)]
            f = [float(lines[1+i*2]) for i in range(self.n_states)]
        self.n_data_sets = 1
        return [energy], [f]

    @staticmethod
    def split_gaussian_files(input_file):
        """Split gaussian log files."""
        # Read the input file
        with open(input_file, 'r', encoding="utf-8") as f:
            lines = f.readlines()

        # Init
        start_line = None
        end_line = None
        file_count = 0
        prepend = "Entering Gaussian System, Link 0 \n"
        # Iterate through the lines
        for i, line in enumerate(lines):
            if "Initial command:" in line:
                start_line = i
            elif "Normal termination of Gaussian" in line:
                end_line = i
                # Extract the relevant portion
                gaussian_content = "".join(lines[start_line:end_line + 1])
                if file_count > 0:
                    gaussian_content = prepend + gaussian_content
                # Save to a separate file
                output_filename = f"Split_{file_count}.log"
                with open(output_filename, 'w', encoding="utf-8"
                          ) as output_file:
                    output_file.write(gaussian_content)

                file_count += 1

        # print(f"Extracted {file_count} Gaussian files.")
        return file_count

    def read_gaussian_exc(self, file):
        """
        Read excited state Gaussian calculations.

        Parameters
        ----------
        file : STR
            Splitted temporary file.

        Returns
        -------
        energy Excitation energies.
        f      Oscillator strength.
        """
        data = cclib.io.ccread(file)
        self.data = data
        try:
            if data.optdone:
                return "", ""
        except AttributeError:
            try:
                if len(data.etoscs) > 0:
                    self.n_states = len(data.etoscs)
                    self.n_data_sets = 1
                    return [1E7/data.etenergies], [data.etoscs]
                return "", ""
            except AttributeError:
                return "", ""

    def read_orca_exc(self, file):
        """
        Read excited state ORCA calculations.

        Parameters
        ----------
        file : STR
            Splitted temporary file.

        Returns
        -------
        energy Excitation energies.
        f      Oscillator strength.
        """
        try:
            data = cclib.io.ccread(file)
            self.data = data
            energy = []
            f = []
            for i, sym in enumerate(data.etsyms):
                if sym == "Singlet":
                    energy.append(float(1E7/data.etenergies[i]))
                    f.append(float(data.etoscs[i]))
                else:
                    pass
            self.n_states = len(energy)
            self.n_data_sets = 1
            return [np.array(energy)], [np.array(f)]
        except AttributeError:
            with open(file, encoding="utf-8") as dato:
                lines = dato.read().split('\n')
            n1_line = lines.index('         ABSORPTION SPECTRUM VIA' +
                                  ' TRANSITION ELECTRIC DIPOLE MOMENTS')
            energy = [float(lines[n1_line+5+i].split()[2])
                      for i in range(self.n_states)]
            f = [float(lines[n1_line+5+i].split()[3])
                 for i in range(self.n_states)]
            try:
                n2_line = lines.index('         ABSORPTION SPECTRUM VIA' +
                                      ' TRANSITION VELOCITY DIPOLE MOMENTS')
                energy2 = [float(lines[n2_line+5+i].split()[2])
                           for i in range(self.n_states)]
                f2 = [float(lines[n2_line+5+i].split()[3])
                      for i in range(self.n_states)]
                self.n_data_sets = 2
                return [energy, energy2], [f, f2]
            except ValueError:
                self.n_data_sets = 1
                return [energy], [f]

    def read_molcas_exc(self, file):
        """
        Read excited state Molcas/OpenMolcas calculations.

        Parameters
        ----------
        file : STR
            Splitted temporary file.

        Returns
        -------
        energy Excitation energies.
        f      Oscillator strength.
        """
        with open(file, encoding="utf-8") as data:
            lines = list(data.read().split())
        n0_line = 0
        # get Total CASPT2 and (X)MS-CASPT2 energies
        n1_line = lines.index('energies:', n0_line+1)
        if lines[n1_line-2] == 'Total':
            energy = [float(lines[n1_line+7+i*7])
                      for i in range(self.n_states+1)]
            if self.debug:
                print(lines[n1_line-1], 'energies =', energy)
        n1_line = lines.index('energies:', n1_line+1)
        n1_line = lines.index('energies:', n1_line+1)
        if lines[n1_line-2] == 'Total':
            energy = [float(lines[n1_line+7+i*7])
                      for i in range(self.n_states+1)]
            if self.debug:
                print(lines[n1_line-1], 'energies =', energy)
        if self.debug:
            print(' ')

        n1_line = lines.index('D:o,', n0_line+1)
        energy = [(1/float(lines[n1_line+9+i*4]))*10**7
                  for i in range(self.n_states)]
        n1_line = lines.index('(sec-1)', n1_line+1)
        n1_line = lines.index('(sec-1)', n1_line+1)
        exc_num = [lines[n1_line+2+i*7] for i in range(self.n_states)]
        _n_states_rel = self.n_states
        for i in range(self.n_states):
            if exc_num[i] != '1':
                _n_states_rel -= 1
        f = [float(lines[n1_line+4+i*7]) for i in range(_n_states_rel)]
        for i in range(self.n_states-_n_states_rel):
            f[self.n_states-i] = 0.
        try:
            n2_line = lines.index('(sec-1)', n1_line+1)
            n2_line = lines.index('(sec-1)', n2_line+1)
            exc_num = [lines[n2_line+2+i*7] for i in range(self.n_states)]
            _n_states_rel = self.n_states
            for i in range(self.n_states):
                if exc_num[i] != '1':
                    _n_states_rel -= 1
            f2 = [float(lines[n2_line+4+i*7]) for i in range(_n_states_rel)]
            for i in range(self.n_states-_n_states_rel):
                f2[self.n_states-i] = 0.
            self.n_data_sets = 2
            return [energy, energy], [f, f2]
        except ValueError:
            self.n_data_sets = 1
            return [energy], [f]

    def read_terachem_exc(self, file):
        """
        Read excited state TeraCHEM calculations.

        Parameters
        ----------
        file : STR
            Splitted temporary file.

        Returns
        -------
        energy Excitation energies.
        f      Oscillator strength.
        """
        with open(file, encoding="utf-8") as data:
            lines = list(data.read().split())
        n1_line = lines.index('Results:')
        ev_en = [float(lines[n1_line+20+i*13]) for i in range(self.n_states)]
        energy = [float(1239.841/ev_en[i]) for i in range(self.n_states)]
        f = [float(lines[n1_line+21+i*13]) for i in range(self.n_states)]
        self.n_data_sets = 1
        return [energy], [f]

    def read_dalton_tpa_exc(self, file):
        """
        Read 1 or 2 photon absorption (OPA & TPA) from DALTON calculations.

        Parameters
        ----------
        file : STR
            Splitted temporary file.

        Returns
        -------
        energy Excitation energies.
        f      Oscillator strength.
        """
        try:
            data = cclib.io.ccread(file)
            self.data = data
            self.n_states = len(data.etoscs)
            self.n_data_sets = 1
            return [1E7/data.etenergies], [data.etoscs]
        except UnboundLocalError:
            with open(file, encoding="utf-8") as data:
                lines = list(data.read().split())
            n1_line = lines.index('summary')
            ev_en = [float(lines[n1_line+16+i*18])
                     for i in range(self.n_states)]
            energy = [float(2*1239.841/ev_en[i])
                      for i in range(self.n_states)]
            f = [float(lines[n1_line+21+i*18])
                 for i in range(self.n_states)]
            self.n_data_sets = 1
            return [energy], [f]


class Spectra(ConfigParse):
    """
    Workflow to Simulate UV-Vis spectrum from QM data.

    This class sets the method and functions to properly do it
    """

    def simulate(self):
        """Simulate UV-Vis spectra."""
        # Timing start
        start = time.time()
        # Configure dictionary
        ConfigParse.ass_vars(self, ConfigParse.parsecl(self))
        # Data extraction and convolution
        Spectra.generate_spectra(self)

        # Display configuration details
        print(f'Using {self.software_qm} data')
        print(f'Using {self.weight_fn} Gaussian & \
{1-self.weight_fn} Lorentzian functions')
        print(f'Plotting {self.plot_single_fn} individual functions and \
{self.plot_single_f} oscillator strength bars')
        print("")

        # Print energies and oscillator strengths
        if self.debug:
            print(_file)
            print('Energies:', self.energ)
            print('Oscillator strengths:', self.fos)
        if not self.multi:
            # Save transitions to file
            Spectra.output_ener_f(self)
            # Normalize spectra
            Spectra.normalize(self)
            # Plot spectra
            Spectra.plot_spectra(self)
        # Remove TMP files in case of File/None data
        if self.FILEdata:
            for count3 in range(self.n_files):
                os.remove(f"TMP_{count3}.dat")
            del count3
        end = time.time()
        print(self.__dict__)
        print("Execution time :", round(end-start, 3), "s")

    def function_gen(self, x_x, landa, f_osc):
        """
        Generate a pseudo-Voigt function for each electronic transition.

        Using Gaussian and Lorentzian functions.

        Parameters
        ----------
        x_x : ARR
            Wavelength range.
        landa : ARR
            Excitation energy.
        fo : ARR
            Oscillator strength.

        Returns
        -------
        spectra : ARR  Function values, spectrum.

        """

        if self.TPA:
            # Cross section value as maximum value
            dev = (float(self.sigma)*8.065544e3)
            sigma1 = ((1/3099.6)*self.sigma)/(0.4*math.pi)
            sigma2 = ((1E7/3099.6)*self.sigma)/(0.4*math.pi)
            const1 = sigma2
            gau_fn = (const1*(f_osc/sigma2)*math.exp(-((((1/x_x)-(1/landa))
                                                        / sigma1)**2)))

            spectrum = self.weight_fn * gau_fn
        else:
            # epsilon values
            dev = (float(self.sigma)*8.065544e3)
            sigma1 = ((1/3099.6)*self.sigma) / 0.4
            sigma2 = ((1E7/3099.6)*self.sigma)/0.4
            const1 = 1.3062974E8

            gau_fn = const1*(f_osc/dev)*math.exp(-1*(((1/x_x)-(1/landa)) /
                                                     (dev/1e7))**2)

            loren_fn = ((const1*(2/math.pi)**0.5*f_osc*(dev/1e14)) /
                        (4*((1/x_x)-(1/landa))**2+(dev/1e7)**2))

            spectrum = (self.weight_fn * gau_fn) + ((1 - self.weight_fn)
                                                    * loren_fn)
        return spectrum

    def generate_spectra(self):
        """
        Convolute and store all generated functions and convolute sum spectra.

        Returns
        -------
        _all_energy : ARR  All energies.
        _all_fosc : ARR  All oscillator strengths.
        _all_sumvaluesy : ARR  All convoluted spectra.
        _all_valuesy : ARR  All functions.
        """

        def convolute(self, _file):
            # Get energies and oscillator strengths for current data set
            self.energ, self.fos = ConfigParse.get_qm_data(self, _file)
            if not self.energ and not self.fos:
                return "", "", "", ""
            # Convolute function for each transition
            # Correct extra line with debug
            if self.debug:
                n_states_corr = self.n_states-1
            else:
                n_states_corr = self.n_states
            # n_states_corr = self.n_states
            valuesy = [[[Spectra.function_gen(self, i, self.energ[k][j],
                                              self.fos[k][j])
                        for i in self.wavelengths]
                        for j in range(n_states_corr)]
                       for k in range(self.n_data_sets)]

            # Sum all functions to get the total spectra
            sumvaluesy = np.sum(valuesy, axis=1)

            # output individual data sets
            if self.debug:
                to_output = valuesy
                total = sumvaluesy.tolist()
                for k in range(self.n_data_sets):
                    to_output[k].append(total[k])
                    Spectra.output(self, self.wavelengths,
                                   np.array(to_output[k]), k)

            return self.energ, self.fos, valuesy, sumvaluesy

        # Initialize data sets
        _all_energy = []
        _all_fosc = []
        _all_sumvaluesy = []
        _all_valuesy = []
        for _file in self.files:
            # Set workflow for independent files
            if self.multi:
                # if self.n_files == 1:
                self.file_count = ConfigParse.split_gaussian_files(_file)
                # else:
                # self.file_count = 1
                for i in range(self.file_count):
                    _all_energy = []
                    _all_fosc = []
                    _all_sumvaluesy = []
                    _all_valuesy = []
                    if self.file_count > 1:
                        self.energ, self.fos, valuesy, sumvaluesy = convolute(
                            self, f"Split_{i}.log")
                    else:
                        self.energ, self.fos, valuesy, sumvaluesy = convolute(
                            self, _file)
                    if self.energ:
                        # Append energy, fosc, sumvaluesy and valuesy
                        _all_energy.append(self.energ)
                        _all_fosc.append(self.fos)
                        _all_sumvaluesy.append(sumvaluesy)
                        _all_valuesy.append(valuesy)
                        self.all_energy = np.array(_all_energy)
                        self.all_fosc = np.array(_all_fosc)
                        self.all_sumvaluesy = np.array(_all_sumvaluesy)
                        self.all_valuesy = np.array(_all_valuesy)

                        # Save transitions to file
                        Spectra.output_ener_f(self)
                        # Normalize spectra
                        Spectra.normalize(self)
                        # Plot spectra
                        Spectra.plot_spectra(self)
                        if self.n_files == 1:
                            _all_energy = []
                            _all_fosc = []
                            _all_sumvaluesy = []
                            _all_valuesy = []
                    else:
                        pass

                # Increase file count
                if self.n_files > 1:
                    self.fc += 1
                for count3 in range(self.file_count):
                    os.remove(f"Split_{count3}.log")
                del count3
            else:
                self.energ, self.fos, valuesy, sumvaluesy = convolute(self,
                                                                      _file)
                # Append energies, oscillator strengths, sumvaluesy and valuesy
                _all_energy.append(self.energ)
                _all_fosc.append(self.fos)
                _all_sumvaluesy.append(sumvaluesy)
                _all_valuesy.append(valuesy)

        if not self.multi:
            self.all_energy = np.array(_all_energy)
            self.all_fosc = np.array(_all_fosc)
            self.all_sumvaluesy = np.array(_all_sumvaluesy)
            self.all_valuesy = np.array(_all_valuesy)

    def normalize(self):
        """Normalize the convoluted spectra depending on number of files."""
        if self.multi:
            if self.n_files > 1:
                factor = 1
            else:
                factor = self.n_files
        else:
            factor = self.n_files
        norm_spectra = np.sum(self.all_sumvaluesy, axis=0)/factor
        Spectra.output_norm(self, self.wavelengths, norm_spectra)
        self.normalized_spectra = np.array(norm_spectra)

    def output(self, x_val, y_val, k):
        """
        Save individual Functions to file.

        Parameters
        ----------
        x_val : LIST
            Wavelength range.
        y_val : LIST
            Spectra data.
        filename : STR
            File name for output.dat.
        k : INT
            Iteration
        """
        if k == 1:
            prop = "_velo"
        else:
            prop = ""
        # Write output
        with open(f'Data_{self.NAMES[self.fc]}_{self.sigma}eV_' +
                  f'{self.weight_fn}{prop}_{self.metadata}.dat', 'w',
                  encoding="utf-8") as data:
            data.write('  x Values   \t')
            for i in range(y_val.shape[0]):
                data.write(f'S{i+1} Values   \t')
            data.write('Total Values   \t')
            data.write('\n')
            data.write('----------- \t')
            for i in range(y_val.shape[0]):
                data.write('-------------\t')
            data.write('\n')
            for i, _ in enumerate(x_val):
                data.write(f'{_:10.3f}   \t')
                for j in range(y_val.shape[0]):
                    if j == y_val.shape[0]:
                        data.write(f'{y_val[j, i]:10.7E}')
                    else:
                        data.write(f'{y_val[j, i]:10.7E}\t')
                data.write('\n')

        # Assuming x_val and y_val are numpy arrays
        data = {'x Values': x_val}
        for i in range(y_val.shape[0]):
            data[f'S{i+1} Values'] = y_val[i]

        df = pd.DataFrame(data)
        df['Total Values'] = df.iloc[:, 1:].sum(axis=1)

        # Save to a .dat file
        df.to_excel(f'Data_{self.NAMES[self.fc]}_{self.sigma}eV_' +
                    f'{self.weight_fn}{prop}_{self.metadata}.xlsx',
                    index=False, float_format='%.7E')

    def output_norm(self, x_val, y_val):
        """
        Save Normalized convoluted spectra to file.

        Parameters
        ----------
        x_val : LIST
            Wavelength range.
        y_val : LIST
            Spectra data.
        """
        for k in range(self.n_data_sets):
            if k == 1:
                with open(f'{self.NAMES[self.fc]}_Normalized_{self.sigma}eV_' +
                          f'{self.weight_fn}_velo_{self.metadata}.dat',
                          'w', encoding="utf-8") as data:
                    data.write('  x Values   \t')
                    data.write('y Norml Values\n')
                    data.write('-----------  \t')
                    data.write('-------------\n')
                    for i, _ in enumerate(x_val):
                        data.write(f'{_:10.3f}\t')
                        data.write(f'{y_val[k][i]:10.7E}\t')
                        data.write('\n')

                # Excel export
                data = {'x Values': x_val, 'y Norml Values': y_val[k]}
                df = pd.DataFrame(data)
                # Save to a .xlsx file
                df.to_excel(f'{self.NAMES[self.fc]}_Normalized_{self.sigma}' +
                            f'eV_{self.weight_fn}_velo_{self.metadata}.xlsx',
                            index=False, float_format='%.7E')
            else:
                with open(f'{self.NAMES[self.fc]}_Normalized_{self.sigma}eV_' +
                          f'{self.weight_fn}_{self.metadata}.dat', 'w',
                          encoding="utf-8") as data:
                    data.write('  x Values   \t')
                    data.write('y Norml Values\n')
                    data.write('-----------  \t')
                    data.write('-------------\n')
                    for i, _ in enumerate(x_val):
                        data.write(f'{_:10.3f}\t')
                        data.write(f'{y_val[k][i]:10.7E}\t')
                        data.write('\n')

                # Excel export
                data = {'x Values': x_val, 'y Norml Values': y_val[k]}
                df = pd.DataFrame(data)
                # Save to a .xlsx file
                df.to_excel(f'{self.NAMES[self.fc]}_Normalized_{self.sigma}' +
                            f'eV_{self.weight_fn}_{self.metadata}.xlsx',
                            index=False, float_format='%.7E')

    def output_ener_f(self):
        """Save all transitions to file."""
        x_val = self.all_energy
        y_val = self.all_fosc
        for k in range(self.n_data_sets):
            if k == 1:
                with open(f'{self.NAMES[self.fc]}_Transitions_velocity_' +
                          f'{self.metadata}.dat', 'w',
                          encoding="utf-8") as data:
                    data.write('Energies  Osc_str_(f)')
                    data.write('\n')
                    for h_h in range(self.n_files):
                        for j in range(self.n_states):
                            data.write(f'{x_val[h_h][k][j]:7.3f}   ')
                            data.write(
                                f'{round(y_val[h_h][k][j], self.qm_decimals):10.{self.qm_decimals}f}\n')
            else:
                with open(f'{self.NAMES[self.fc]}_Transitions_' +
                          f'{self.metadata}.dat', 'w',
                          encoding="utf-8") as data:
                    data.write('Energies  Osc_str_(f)')
                    data.write('\n')
                    if self.multi:
                        if self.n_files > 1:
                            lista = range(1)
                        else:
                            lista = range(self.n_files)
                    else:
                        lista = range(self.n_files)
                    for h_h in lista:
                        for j in range(self.n_states):
                            data.write(f'{x_val[h_h][k][j]:7.3f}   ')
                            data.write(
                                f'{round(y_val[h_h][k][j], self.qm_decimals):10.{self.qm_decimals}f}\n')

    def save_figures(self, _n_data_sets_iter):
        """
        Save spectra png.

        Parameters
        ----------
        _n_data_sets_iter : INT
            _n_data_sets iteration.
        """
        # Save png spectrum
        if _n_data_sets_iter == 1:
            plt.savefig(f'{self.NAMES[self.fc]}_Spectrum_velocity_' +
                        f'{self.metadata}.png', dpi=450)
        else:
            if self.TPA:
                plt.savefig(f'{self.NAMES[self.fc]}_Spectrum_' +
                            f'{self.metadata}_TPA.png', dpi=450)
            else:
                plt.savefig(f'{self.NAMES[self.fc]}_Spectrum_' +
                            f'{self.metadata}.png', dpi=450)

    def print_individuals(self, k, ax1, _all_energy, _all_fosc, _all_valuesy):
        """
        Plot individual functions or bars.

        Parameters
        ----------
        k : TYPE
            DESCRIPTION.
        ax1 : TYPE
            DESCRIPTION.
        all_energy : ARR
            Excitation energies.
        all_fosc : ARR
            Oscillator strength.
        all_valuesy : ARR
            Single function values.
        """
        if self.plot_single_f > 0:
            if not self.TPA:
                ax2 = ax1.twinx()
                ax2.set_ylabel('Oscillator Strength (esu$^{2}$ cm$^{2}$)')

        def parallel_plotting(h_h, colors):
            if self.plot_single_f > 0:
                if self.TPA:
                    for j in range(self.plot_single_f):
                        ax1.bar(_all_energy[h_h, k, j], _all_fosc[h_h, k, j],
                                color=colors[j], edgecolor='black')
                else:
                    for j in range(self.plot_single_f):
                        if _all_energy[h_h, k, j] > self.initial_wave:
                            ax2.bar(_all_energy[h_h, k, j],
                                    _all_fosc[h_h, k, j],
                                    color=colors[j])
                        else:
                            pass
            if self.plot_single_fn > 0:
                for j in range(self.plot_single_fn):
                    ax1.plot(self.wavelengths, _all_valuesy[h_h, k, j],
                             color=colors[j])
        # Set colors to have always the same for each pair of data
        if self.plot_single_fn >= self.plot_single_f:
            colors = plt.cm.inferno(np.linspace(0, 1, self.plot_single_fn))
        else:
            colors = plt.cm.inferno(np.linspace(0, 1, self.plot_single_f))

        if self.multi:
            if self.n_files > 1:
                lista = range(1)
            else:
                lista = range(self.n_files)
        else:
            lista = range(self.n_files)

        for h_h in lista:
            parallel_plotting(h_h, colors)

    def plot_spectra(self):
        """Plot normalized UV-Vis spectra."""
        # Plot the Normalized convoluted spectra
        for k in range(self.n_data_sets):
            fig, ax1 = plt.subplots()
            del fig
            plt.plot(self.wavelengths, self.normalized_spectra[k], linewidth=3)
            if self.TPA:
                plt.title('Simulated TPA Spectrum')
                ax1.set_xlabel(r'$\lambda$ (nm)')
                ax1.set_ylabel(r'$\sigma$ (GM)')
            else:
                plt.title('Simulated UV/Visible Spectrum')
                ax1.set_xlabel(r'$\lambda$ (nm)')
                ax1.set_ylabel(r'$\epsilon$ (L mol$^{-1}$ cm$^{-1}$)')
            ax1.set_ylim(0.)

            Spectra.print_individuals(self, k, ax1, self.all_energy,
                                      self.all_fosc, self.all_valuesy)

            ax1.plot(self.wavelengths, self.normalized_spectra[k], linewidth=3,
                     color='steelblue')
            plt.tight_layout()
            # Saving file to png
            Spectra.save_figures(self, k)
            if self.debug:
                plt.show()


if __name__ == "__main__":
    # Append location to the path
    sys.path.append(os.path.dirname(__file__))
    # Run launcher
    spectra = Spectra()
    spectra.simulate()
