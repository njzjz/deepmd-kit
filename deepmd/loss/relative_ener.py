from typing import Dict, Tuple

from .loss import Loss
from deepmd.common import add_data_requirement
from deepmd.env import tf
from deepmd.env import global_cvt_2_tf_float, global_cvt_2_ener_float


class RelativeEnergyLoss(Loss):
    """Relative energy loss.

    Considering a reaction is: A + B -> C + D, and we
    only want to train the exact reaction energy:

    .. math::
        \Delta E = E(C) + E(D) - E(A) - E(B)

    The loss of the relative energy is defined by the MSE:

    .. math::
        L = \frac{1}{N} \sum_k^N (\Delta E_k - \Delta \hat{E}_k)^2

    In the DeePMD-kit package, we do not want to break the
    current interface, so we put aperiodic structures
    A, B, C, D into the same frame, but far away from each other.
    They will have no interation between each other.
    Then add `atom_pref` data to mark stoichiometric number
    :math:`\nu`:

    .. math::
        \Delta E = \sum_i \nu_i E_i

    :math:`\Delta E` could be labeled by `rel_energy` data.

    Forces can be fitted if available, considering the optimized
    geometry should be accurate.

    Parameters
    ----------
    starter_learning_rate : float
        start learning rate
    start_pref_re : float
        start prefactor of relative energy
    limit_pref_re : float
        limit prefactor of relative energy
    start_pref_f : float
        start prefactor of force
    limit_pref_f : float
        limit prefactor of force
    """

    def __init__(self,
                 starter_learning_rate: float,
                 start_pref_re: float = 0.02,
                 limit_pref_re: float = 1.00,
                 start_pref_f: float = 1000,
                 limit_pref_f: float = 1.00,
                 ) -> None:
        self.starter_learning_rate = starter_learning_rate
        self.start_pref_re = start_pref_re
        self.limit_pref_re = limit_pref_re
        self.start_pref_f = start_pref_f
        self.limit_pref_f = limit_pref_f
        # self.has_re = True
        if self.start_pref_re != 0.0 or self.limit_pref_re != 0.0:
            raise RuntimeError(
                "Relative energy loss prefactor must be not zero. "
                "Otherwise, use energy loss instead."
            )
        self.has_f = (self.start_pref_f != 0.0 or self.limit_pref_f != 0.0)
        add_data_requirement('rel_energy', 1, atomic=False,
                             must=True, high_prec=True)
        add_data_requirement('force', 3, atomic=True,
                             must=False, high_prec=False)
        add_data_requirement('atom_pref', 1, atomic=True,
                             must=True, high_prec=False)

    def build(self,
              learning_rate: tf.Tensor,
              natoms: tf.Tensor,
              model_dict: Dict[str, tf.Tensor],
              label_dict: Dict[str, tf.Tensor],
              suffix: str) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Build the relative energy loss function graph.

        Parameters
        ----------
        learning_rate : tf.Tensor
            learning rate
        natoms : tf.Tensor
            number of atoms
        model_dict : dict[str, tf.Tensor]
            A dictionary that maps model keys to tensors
        label_dict : dict[str, tf.Tensor]
            A dictionary that maps label keys to tensors
        suffix : str
            suffix

        Returns
        -------
        tf.Tensor
            the total squared loss
        dict[str, tf.Tensor]
            A dictionary that maps loss keys to more loss tensors
        """
        # loss of relative energy
        # nframes * natoms
        atom_ener = model_dict['atom_ener']
        # nframes * natoms
        atom_pref = label_dict['atom_pref']
        relative_atom_ener = atom_ener * atom_pref
        # make the sum -> nframes
        relative_ener = tf.sum(relative_atom_ener, axis=1)
        relative_ener_hat = label_dict['rel_energy']
        l2_rel_ener_loss = tf.reduce_mean(
            tf.square(relative_ener_hat - relative_ener), name="l2_rel_ener_" + suffix)

        # loss of forces
        force = model_dict['force']
        force_hat = label_dict['force']
        force_reshape = tf.reshape(force, [-1])
        force_hat_reshape = tf.reshape(force_hat, [-1])
        atom_pref_reshape = tf.reshape(atom_pref, [-1])
        diff_f = force_hat_reshape - force_reshape
        l2_force_loss = tf.reduce_mean(
            tf.square(diff_f), name="l2_force_" + suffix)

        # prefactor
        atom_norm = 1. / global_cvt_2_tf_float(natoms[0])
        atom_norm_ener = 1. / global_cvt_2_ener_float(natoms[0])
        find_force = label_dict['find_force']
        pref_re = global_cvt_2_ener_float(self.limit_pref_re + (
            self.start_pref_re - self.limit_pref_re) * learning_rate / self.starter_learning_rate)
        pref_f = global_cvt_2_tf_float(find_force * (self.limit_pref_f + (
            self.start_pref_f - self.limit_pref_f) * learning_rate / self.starter_learning_rate))

        more_loss = {}
        # relative energy loss
        l2_loss = atom_norm_ener * (pref_re * l2_rel_ener_loss)
        more_loss['l2_rel_ener_loss'] = l2_rel_ener_loss
        # force loss
        if self.has_f:
            l2_loss += global_cvt_2_ener_float(pref_f * l2_force_loss)
        more_loss['l2_force_loss'] = l2_force_loss

        # only used when tensorboard was set as true
        self.l2_loss_summary = tf.summary.scalar('l2_loss', tf.sqrt(l2_loss))
        self.l2_loss_rel_ener_summary = tf.summary.scalar('l2_rel_ener_loss', global_cvt_2_tf_float(tf.sqrt(l2_rel_ener_loss)) / global_cvt_2_tf_float(natoms[0]))
        self.l2_loss_force_summary = tf.summary.scalar('l2_force_loss', tf.sqrt(l2_force_loss))

        self.l2_l = l2_loss
        self.l2_more = more_loss
        return l2_loss, more_loss

    def eval(self,
             sess: tf.Session,
             feed_dict: Dict[tf.placeholder, tf.Tensor],
             natoms: tf.Tensor) -> dict:
        """Eval the loss function.

        Parameters
        ----------
        sess : tf.Session
            TensorFlow session
        feed_dict : dict[tf.placeholder, tf.Tensor]
            A dictionary that maps graph elements to values
        natoms : tf.Tensor
            number of atoms

        Returns
        -------
        dict
            A dictionary that maps keys to values. It
            should contain key `natoms`
        """
        placeholder = self.l2_l
        run_data = [
            self.l2_l,
            self.l2_more['l2_rel_ener_loss'],
            self.l2_more['l2_force_loss'] if self.has_f else placeholder,
        ]
        error, error_re, error_f = run_sess(sess, run_data, feed_dict=feed_dict)
        results = {"natoms": natoms[0], "rmse": np.sqrt(error)}
        results["rmse_re"] = np.sqrt(error_re) / natoms[0]
        if self.has_f:
            results["rmse_f"] = np.sqrt(error_f)
        return results
