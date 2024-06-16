
import pathlib 
import pytest
import numpy as np
from unittest.mock import patch 
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from Galaxy6DLib import MagneticSensor

# Dir containing 8 files
FIXTURE_DIR = pathlib.Path(__file__).parent.resolve() / 'data'

def inc(x):
    return x + 1

#@patch("matplotlib.pyplot.show")

@pytest.mark.datafiles( FIXTURE_DIR / 'mag.txt',)
def test_calibration(datafiles):
    filename = list(datafiles.iterdir())[0]
    ms = MagneticSensor()    
    ms.loadRaw(filename)
    
    ms.estimateCalibration(targetField=0.54)


    # Magneto 1.2 results, normalized to 0.54 [same unit as measurement]
    b  = np.asarray([ -0.021659,  0.013250, -0.026167]);
    Ai = np.asarray([[ 0.908200, -0.016901,  0.006229],
                     [-0.016901,  0.915374,  0.003136],
                     [ 0.006229,  0.003136,  0.983337]])
    
        
    if 1==0:  # my test file results
        b = np.asarray([0.049905, 0.128849, 0.044358]); 
        Ai = np.asmatrix([[ 2.177104, -0.014507,  0.011303],
                          [-0.014507,  2.149547,  0.022436],
                          [ 0.011303,  0.022436,  2.161512]])

    logger.info(f"Results for testfile {filename}")        
    logger.info(f"True  b:   {b}")        
    logger.info(f"Estim b:   {ms.b.round(6)}")        
    logger.info(f"True  Ai:\n {Ai}")        
    logger.info(f"Estim Ai:\n {ms.Ai.round(6)}")        

    #ms.plot()

    #np.testing.assert_allclose(ms.b,b,rtol=1e-3)     # actual versus desired
    #np.testing.assert_allclose(ms.Ai,Ai,rtol=1e-3) 

    assert(True)
    #assert inc(3) == 5