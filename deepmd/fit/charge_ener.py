from deepmd.env import (
    tf,
    GLOBAL_TF_FLOAT_PRECISION,
)
from deepmd.common import add_data_requirement

from .ener import EnerFitting


class ChargeEnerFitting(EnerFitting):
    def __init__ (self, *args, **kwargs) -> None:
        EnerFitting.__init__(self, *args, **kwargs)
        add_data_requirement('charge', 1, atomic=True,  must=False, high_prec=False)
        self.dim_descrpt += 1

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
                The system energy
        """
        # charge
        charge = input_dict['charge']
        charge = tf.reshape(charge, [-1, natoms[0], 1])
        # zero charge, zero energy
        atomic_zero = tf.zeros_like(charge, dtype=GLOBAL_TF_FLOAT_PRECISION)

        # inputs
        inputs = tf.reshape(inputs, [-1, natoms[0], self.dim_descrpt - 1])
        inputs_charged = tf.concat([inputs, charge], axis = 2)
        inputs_neutral = tf.concat([inputs, atomic_zero], axis = 2)
        # fitting
        fitting_charged = EnerFitting.build(self, inputs_charged, natoms, input_dict, False, suffix)
        fitting_neutral = EnerFitting.build(self, inputs_neutral, natoms, input_dict, True, suffix)
        fitting_diff = fitting_charged - fitting_neutral
        # skip if charge is zero
        result = tf.cond(tf.count_nonzero(charge) > 0, lambda: fitting_diff, lambda: atomic_zero)
        return result

    def _init_variables(self,
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
        return