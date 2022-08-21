from .fitting import Fitting
from . import EnerFitting, WFCFitting, PolarFittingLocFrame, PolarFittingSeA, GlobalPolarFittingSeA, DipoleFittingSeA
from .hybrid_ener import HybridEnerFitting
from .charge_ener import ChargeEnerFitting


def get_fitting_network(fitting_type: str, descrpt_type: str, fitting_param: dict) -> Fitting:
    """Get the fitting network.
    
    Parameters
    ----------
    fitting_type : str
        The fitting type.
    descrpt_type : str
        The descriptor type.

    Returns
    -------
    Fitting
        The fitting network.
    """
    # fitting net
    fitting_type = fitting_param.get('type', 'ener')
    fitting_param.pop('type', None)
    if fitting_type == 'ener':
        return EnerFitting(**fitting_param)
    elif fitting_type == 'hybrid_ener':
        return HybridEnerFitting(**fitting_param)
    elif fitting_type == 'charge_ener':
        return ChargeEnerFitting(**fitting_param)
    # elif fitting_type == 'wfc':            
    #     return WFCFitting(fitting_param, self.descrpt)
    elif fitting_type == 'dipole':
        if descrpt_type == 'se_e2_a':
            return DipoleFittingSeA(**fitting_param)
        else :
            raise RuntimeError('fitting dipole only supports descrptors: se_e2_a')
    elif fitting_type == 'polar':
        # if descrpt_type == 'loc_frame':
        #     return PolarFittingLocFrame(fitting_param, self.descrpt)
        if descrpt_type == 'se_e2_a':
            return PolarFittingSeA(**fitting_param)
        else :
            raise RuntimeError('fitting polar only supports descrptors: loc_frame and se_e2_a')
    elif fitting_type == 'global_polar':
        if descrpt_type == 'se_e2_a':
            return GlobalPolarFittingSeA(**fitting_param)
        else :
            raise RuntimeError('fitting global_polar only supports descrptors: loc_frame and se_e2_a')
    else :
        raise RuntimeError('unknow fitting type ' + fitting_type)
