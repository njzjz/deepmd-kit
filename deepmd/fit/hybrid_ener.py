import warnings
import numpy as np
from typing import Tuple, List

from deepmd.env import tf
from .fitting import Fitting
from .get_fitting import get_fitting


class HybridEnerFitting(Fitting):
    r"""Hybrid energy fitting model.

    Parameters
    ----------
    list : list
        List of fittings
    suffixes : list, optional
        The suffixes of the scope. If None, use _index as suffixes.
    """
    def __init__ (self, 
                  list: list,
                  suffixes: list = None,
    ) -> None:
        # warning: list is conflict with built-in list
        fitting_list = list
        if fitting_list == [] or fitting_list is None:
            raise RuntimeError('cannot build fitting from an empty list of fittings.')
        formatted_fitting_list = []
        for ii in fitting_list:
            if isinstance(ii, Fitting):
                formatted_fitting_list.append(ii)
            elif isinstance(ii, dict):
                # descrpt does not affect energy fitting
                formatted_fitting_list.append(get_fitting(ii['type'], None)(**ii))
            else:
                raise NotImplementedError
        self.fitting_list = formatted_fitting_list
        self.numb_fitting = len(self.fitting_list)
        if suffixes is None:
            suffixes = ["_" + str(ii) for ii in range(self.numb_fitting)]
        assert len(suffixes) == self.numb_fitting, "suffixes must have the same length as fitting_list"
        self.suffixes = suffixes


    def get_numb_fparam(self) -> int:
        """
        Get the number of frame parameters
        """
        return self.fitting_list[0].numb_fparam

    def get_numb_aparam(self) -> int:
        """
        Get the number of atomic parameters
        """
        return self.fitting_list[0].numb_fparam

    def compute_output_stats(self, 
                             all_stat: dict
    ) -> None:
        """
        Compute the ouput statistics

        Parameters
        ----------
        all_stat
                must have the following components:
                all_stat['energy'] of shape n_sys x n_batch x n_frame
                can be prepared by model.make_stat_input
        """
        for ff in self.fitting_list:
            ff.compute_output_stats(all_stat)

    def compute_input_stats(self, 
                            all_stat : dict,
                            protection : float = 1e-2) -> None:
        """
        Compute the input statistics

        Parameters
        ----------
        all_stat
                if numb_fparam > 0 must have all_stat['fparam']
                if numb_aparam > 0 must have all_stat['aparam']
                can be prepared by model.make_stat_input
        protection
                Divided-by-zero protection
        """
        for ff in self.fitting_list:
            ff.compute_input_stats(all_stat)
            
    @cast_precision
    def build (self, 
               inputs : tf.Tensor,
               natoms : tf.Tensor,
               input_dict : dict = None,
               reuse : bool = None,
               suffix : str = '', 
    ) -> tf.Tensor:
        """
        Build the computational graph for fitting net

        Parameters
        ----------
        inputs
                The input descriptor
        input_dict
                Additional dict for inputs. 
                if numb_fparam > 0, should have input_dict['fparam']
                if numb_aparam > 0, should have input_dict['aparam']
        natoms
                The number of atoms. This tensor has the length of Ntypes + 2
                natoms[0]: number of local atoms
                natoms[1]: total number of atoms held by this processor
                natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        reuse
                The weights in the networks should be reused when get the variable.
        suffix
                Name suffix to identify this descriptor

        Returns
        -------
        ener
                The atomic energy
        """
        outs = []
        for ii, ss in zip(self.fitting_list, self.suffixes):
            out = ii.build(inputs, natoms, input_dict, reuse, ss + suffix)
            outs.append(tf.reshape(out), [-1])
        # sum the first axis
        return tf.reduce_sum(outs, 0)

    def init_variables(self,
                       graph: tf.Graph,
                       graph_def: tf.GraphDef,
                       suffix : str = "",
    ) -> None:
        """
        Init the fitting net variables with the given dict

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix : str
            suffix to name scope
        """
        for ii, ss in zip(self.fitting_list, self.suffixes):
            ii.init_variables(graph, graph_def, ss + suffix)

    def enable_compression(self,
                           model_file: str,
                           suffix: str = "",
    ) -> None:
        """
        Set the fitting net attributes from the frozen model_file when fparam or aparam is not zero

        Parameters
        ----------
        model_file : str
            The input frozen model file
        suffix : str, optional
                The suffix of the scope
        """
        for ii, ss in zip(self.fitting_list, self.suffixes):
            ii.enable_compression(model_file, ss + suffix)
 

    def enable_mixed_precision(self, mixed_prec: dict = None) -> None:
        """
        Reveive the mixed precision setting.

        Parameters
        ----------
        mixed_prec
                The mixed precision setting used in the embedding net
        """
        for ii in self.fitting_list:
            ii.enable_mixed_precision(mixed_prec)
