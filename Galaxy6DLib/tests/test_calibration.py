
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
#print("FixDirXXX:",FIXTURE_DIR )


#@patch("matplotlib.pyplot.show")
@pytest.mark.datafiles( FIXTURE_DIR / 'mag.txt',)
@pytest.mark.parametrize("method", ['minimize','literature'])
#def test_calibration(datafiles,method='literature'):
def test_calibration(datafiles,method):
    filename = list(datafiles.iterdir())[0]
    ms = MagneticSensor()    
    ms.loadRaw(filename)
    


    # Magneto 1.2 results, normalized to 0.54 [same unit as measurement]
    b  = np.asarray([ -0.021659,  0.013250, -0.026167]);
    Ai = np.asarray([[ 0.908200, -0.016901,  0.006229],
                     [-0.016901,  0.915374,  0.003136],
                     [ 0.006229,  0.003136,  0.983337]])
    
    ms.estimateCalibration(targetField=0.54,method=method)
    logger.info(f"Results for testfile {filename}")        
    logger.info(f" method: {method}")        
    logger.info(f" True  b:   {b.round(3)}")        
    logger.info(f" Estim b:   {ms.b.round(3)}")        
    logger.info(f" True  Ai:\n {Ai.round(3)}")        
    logger.info(f" Estim Ai:\n {ms.Ai.round(3)}")        

    #ms.plot()

    np.testing.assert_allclose(ms.b,b,rtol=1e0,atol=0.1)   # actual versus desired
    np.testing.assert_allclose(ms.Ai,Ai,rtol=1e0,atol=0.1) # actual versus desired
