# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 12:39:44 2023

@author:  BastelBaus
@Content: Script to plot and evaluate different magnet measurements 
          from a Galaxy device

Link: Magnetic Sensor Calibrataion Algorithms: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8401862/

"""

import logging 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import pickle
import csv
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def sqrtSumSqr(x):
    return np.sqrt(np.sum(np.square(x),axis=1))
def sqrtMeanSqr(x):
    return np.sqrt(np.mean(np.square(x),axis=1))


def meanLength(vec):
    ''' manLeangth returns the length of an 3x1 arrary or an list of nx3 arrays
    '''
    if vec.shape[0] == 3: return np.mean( np.sqrt( np.square(vec[0,:]) + np.square(vec[1,:]) + np.square(vec[2,:])  ))
    else: return np.mean( np.sqrt( np.square(vec[:,0]) + np.square(vec[:,1]) + np.square(vec[:,2])  ))
    

from platform import system
def plt_maximize():
    # See discussion: https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
    backend = plt.get_backend()
    cfm = plt.get_current_fig_manager()
    if backend == "wxAgg":
        cfm.frame.Maximize(True)
    elif backend == "TkAgg":
        if system() == "Windows":
            cfm.window.state("zoomed")  # This is windows only
        else:
            cfm.resize(*cfm.window.maxsize())
    elif backend == "QT4Agg":
        cfm.window.showMaximized()
    elif callable(getattr(cfm, "full_screen_toggle", None)):
        if not getattr(cfm, "flag_is_max", None):
            cfm.full_screen_toggle()
            cfm.flag_is_max = True
    else:
        raise RuntimeError("plt_maximize() is not implemented for current backend:", backend)
    

class MagneticSensor:
    ''' This class implements a single axis magnetic sensor
    '''


    col  = ['.r','.b','.g']  # just a color vector
    
    Blim = 1.5 # [Gauss] he limits when plotting the magnetic fields

    
    
    
    def __init__(self, data=None,sensitivity=1.0):
        ''' Init Function Docu
            :param data:  desdcription of data
            :type data: list, optional

        '''
        #TODO: setting the senbsitivity does not work
        self.n = 0                    # number of samples
        self.s = 1.0                  # sensitivity in [LSB/G]
        self.method = None
        self.doplot = False

        self.tm              = None   # timestamps
        self.dataRaw         = None   # the raw measurement data    
        self.fieldVec        = None   # absolut length of vector, non calibrated
        self.rotVec          = None   # vector holding the 3 rotation angles  
        self.meanFieldVec    = None   # mean magnetic field over the full measurement
        self.stdFieldVec     = None   # mean magnetic field over the full measurement
     
        self.targetField = 1          #
        self.dataCal         = None   # the calibrated measurement data
        self.fieldVecCal     = None   # absolut length of vector, non calibrated
        self.rotVecCal       = None   # vector holding the 3 rotation angles
        self.meanFieldVecCal = None   # mean magnetic field over the full measurement
        self.stdFieldVecCal  = None   # mean magnetic field over the full measurement

        if  data is not None: self.__setMeasurements(data,sensitivity)

    def __setMeasurements(self,data,sensitivity=1.0):       

        logger.debug(f'set measurement with shape {data.shape} (sensitivity:{sensitivity})')
        self.dataRaw = data
        self.n       = self.dataRaw.shape[1]
        self.s       = sensitivity

    def setTime(self,timestamps):
        """ setTime sets the time axis for the measurement
        
        :param timestamps:  an iterable list of timestamps
        :type timestamps: list
        :return: Nothing
        :rtype: None
        """        
        self.tm = np.asarray(timestamps)

    def calculateRawKPIs(self):
        self.fieldVec  = np.sqrt( np.square(self.dataRaw[0]) + np.square(self.dataRaw[1]) + np.square(self.dataRaw[2]) )
        self.meanFieldVec = np.mean(self.fieldVec)
        self.stdFieldVec  = np.std(self.fieldVec)
        self.rotVec       = np.asarray( [ np.arctan2(self.dataRaw[0], self.dataRaw[1])*180/np.pi,
                                           np.arctan2(self.dataRaw[0], self.dataRaw[2])*180/np.pi,
                                           np.arctan2(self.dataRaw[1], self.dataRaw[2])*180/np.pi ] )
    def calculateCalKPIs(self):
        self.fieldVecCal  = np.sqrt( np.square(self.dataCal[0]) + np.square(self.dataCal[1]) + np.square(self.dataCal[2]) )
        self.meanFieldVecCal = np.mean( self.fieldVecCal )
        self.stdFieldVecCal  = np.std(self.fieldVecCal)
        self.rotVecCal = np.asarray( [ np.arctan2(self.dataCal[0], self.dataCal[1])*180/np.pi,
                                        np.arctan2(self.dataCal[0], self.dataCal[2])*180/np.pi,
                                        np.arctan2(self.dataCal[1], self.dataCal[2])*180/np.pi ] )

    def setAib(self,Ai,b, targetField=1.0, printStats=False):
        """ setAib sets the multiplication matrix Ai and ofset vector b.
        
        :param targetField:  The target magnitude of the magnetic field vector, defaults to 1.0  
        :type targetField: float
        :param printStats:  If true, some statistics are printed to the standard io, defaults to False
        :type printStats: Boolean
        :return: Nothing
        
        """        

        self.method = "extern"
        self.Ai = Ai
        self.b  = b        
        self.targetField = targetField
        self.__applyCalibration()
        self.calculateRawKPIs()
        self.calculateCalKPIs()                                                                                                       
        if printStats: self.printStats() #TODO: does not yetr work


    def calculateAll(self, targetField=1, method='minimize', printStats = True):
        ''' Estimates the calibration parameters, applies them and print statistics.
        '''

        self.calculateRawKPIs()

        self.estimateCalibration(targetField=targetField,method=method)
        self.__applyCalibration()
        self.calculateCalKPIs()                                                                                                       
        if printStats: self.printStats()
    
    def loadRaw(self,filename):
        ''' Load magnetic field data from a file. Three tab seperate values 
            for Bx,By and Bz.
        '''
        logger.info(f'load data from {filename}')
        result = [] 
        with open(filename, newline='') as csvfile:    
            spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            for i,row in enumerate(spamreader):
                if len(row) != 3: continue
                dat = [float(row[0]),float(row[1]),float(row[2])];
                result.append(dat)

        n = len(result)
        logger.info(f'loaded {n} datalines')
        if n>0: self.__setMeasurements( np.asarray(result,dtype=np.float32).T )
            
    def saveRaw(self,filename="output.txt"):
        ''' Save magnetic field data to a file. Three tab seperate values 
            for Bx,By and Bz.
        '''
        with open(filename, "w") as text_file:
            for k in range(self.n):
                row = f"{self.dataRaw[0][k]/self.s}\t{self.dataRaw[1][k]/self.s}\t{self.dataRaw[2][k]/self.s}"
                print(row, file=text_file)

    def getStatisticError(self):
        ampCal = np.sqrt(np.sum(np.square(self.dataCal.T),axis=1))
        err = np.sqrt(np.mean(np.square(ampCal-self.targetField )))
        return err

    def getMean(self):
            return np.mean(self.dataRaw,axis=1)

        
    def printStats(self):
        print('------- Statistics -------')
        print('sensitivity       :',self.s,' [LSB/GAUSS]')
        print('samples           :',self.n)
        print('field vector (raw):',np.round(self.meanFieldVec/self.s*1000)/1000,'GAUSS +/-',np.round(self.stdFieldVec/self.s*1000),'mGAUSS')
        print('field vector (cal):',np.round(self.meanFieldVecCal/self.s*1000)/1000,'GAUSS +/-',np.round(self.stdFieldVecCal/self.s*1000),'mGAUSS')
        print('stat              :', self.getStatisticError())

    
    def __applyCalibration(self):
        logger.info(f'Apply the calibration to the raw data. Sensitivity: {self.s}')
        self.dataCal = ((self.dataRaw.T - self.b) @ self.Ai).T
        logger.debug(f'done, new cal data shape: {self.dataCal.shape} ')


        #self.dataCal[0:3,k] = np.matmul(self.Ai, self.dataRaw[0:3,k] - self.b) / (self.meanFieldVec / self.s )
        #self.dataCal = ((self.dataRaw.T - self.b) @ self.Ai).T / (self.meanFieldVec / self.s )
        #logger.info(f'Normalize {self.meanFieldVec} with sensitivity {self.s}')
 

    def estimateCalibration(self,targetField=1,method='minimize'):
        ''' estimateCalibration estiamtes offset (b) as well as 
            scale and cross talk factors (Ai).

            parmeters:
                targetField: defines the taret amplitude of the 
                        magnetic field (normalization)
                        Default: 1
                method: 'mimnimized' does a scipy minimization search 
                        and any other parmeter does a elliptic optimization as given
                        in literature
                        Default ('minimize')
        '''
        self.method = method
        if method=='minimize':    self.estimateCalibrationMinimizeAbs(targetField=targetField)
        if method=='minimizeVec': self.estimateCalibrationMinimizeVec(targetField=targetField)
        if method=='literature':  self.estimateCalibrationLiterature(targetField=targetField)
        

    def estimateCalibrationMinimizeVec(self,targetField=1):

        self.Ai = np.eye(3)
        self.b = np.zeros(3)
        self.targetField = targetField

        mm = np.mean(self.dataRaw,axis=1)
        mr = (np.max(self.dataRaw,axis=1)-np.min(self.dataRaw,axis=1))/2

        def optfun(x,val):
            
            #logger.debug(f'get:{x.shape} val:{x} ' )
            #mm = np.mean(val,axis=1)
            Ai = x[0:9].reshape(3,3)
            b  = x[9:12]
            dataCal = (val.T - (b + mm) ) @ Ai

            dataRef = ( dataCal / np.linalg.norm(dataCal,axis=1,keepdims=True) ) * self.targetField             
            ampDelta =  np.sqrt(np.sum(np.square( dataCal-dataRef ),axis=1))
            err = np.sqrt(np.mean(np.square(ampDelta )))
            return err

        x0 = np.concatenate( ( np.eye(3).reshape( (9,) ), np.zeros((3,))) )
        logger.info(f'Optimize with start value size:{x0.shape} val:{x0} to target {self.targetField}' )
        relTol = 0.4; 
        tol = [ ( mm[i] - relTol*mr[i]  , mm[i] + relTol*mr[i] ) for i in range(3) ]  
        bnds = ( (None,None), (None,None), (None,None), \
                 (None,None), (None,None), (None,None), \
                 (None,None), (None,None), (None,None), \
                  tol[0], tol[1], tol[2] )
        #res = scipy.optimize.minimize(optfun,x0, self.dataRaw, bounds =bnds) #TODO: does not work, don'T know why
        res = scipy.optimize.minimize(optfun,x0, self.dataRaw)

        if res.success:
            logger.info(f'Optimize successfully: {res.fun} @ {res.x.shape}' )
            self.Ai = res.x[0:9].reshape(3,3)
            self.b  = res.x[9:12] + mm
        else:
            logger.warning(f'Optimize failed: {res.message} \n >>> {res.x.shape} val:{res.x}' )

        logger.info(f'New Calib Params:\nAi:\n{self.Ai.round(5)}\nb:{self.b.round(5)} ' )

    def estimateCalibrationMinimizeAbs(self,targetField=1):

        self.Ai = np.eye(3)
        self.b = np.zeros(3)
        self.targetField = targetField

        mm = np.mean(self.dataRaw,axis=1)
        mr = (np.max(self.dataRaw,axis=1)-np.min(self.dataRaw,axis=1))/2

        def optfun(x,val):
            
            #logger.debug(f'get:{x.shape} val:{x} ' )
            #mm = np.mean(val,axis=1)
            Ai = x[0:9].reshape(3,3)
            b  = x[9:12]
            dataCal = (val.T - (b + mm) ) @ Ai
            ampCal = np.sqrt(np.sum(np.square(dataCal),axis=1))
            err = np.sqrt(np.mean(np.square(ampCal-self.targetField )))
            return err

        x0 = np.concatenate( ( np.eye(3).reshape( (9,) ), np.zeros((3,))) )
        logger.info(f'Optimize with start value size:{x0.shape} val:{x0} to target {self.targetField}' )
        relTol = 0.4; 
        tol = [ ( mm[i] - relTol*mr[i]  , mm[i] + relTol*mr[i] ) for i in range(3) ]  
        bnds = ( (None,None), (None,None), (None,None), \
                 (None,None), (None,None), (None,None), \
                 (None,None), (None,None), (None,None), \
                  tol[0], tol[1], tol[2] )
        #res = scipy.optimize.minimize(optfun,x0, self.dataRaw, bounds =bnds) #TODO: does not work, don'T know why
        res = scipy.optimize.minimize(optfun,x0, self.dataRaw)

        if res.success:
            logger.info(f'Optimize successfully: {res.fun} @ {res.x.shape}' )
            self.Ai = res.x[0:9].reshape(3,3)
            self.b  = res.x[9:12] + mm
        else:
            logger.warning(f'Optimize failed: {res.message} \n >>> {res.x.shape} val:{res.x}' )

        logger.info(f'New Calib Params:\nAi:\n{self.Ai.round(5)}\nb:{self.b.round(5)} ' )



    def estimateCalibrationLiterature(self,targetField=1):
        ''' Implementation of calibration algo 
         
            References:
            * https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8401862/
            * https://sites.google.com/view/sailboatinstruments1/d-implementation
            * https://de.mathworks.com/matlabcentral/fileexchange/23377-ellipsoid-fitting
            * https://github.com/millerlp/Rmagneto/blob/master/Rmagneto.c
            * https://github.com/beattiea/TiltyIMU/blob/master/Tilty%20Software/Processing/Experiments/Testing%20Magnetometer%20calibration/ellipsoid_fit/ellipsoid_fit.m
            * https://github.com/hightower70/MagCal            
        '''


        logger.info('Estimate the calibration parameter to optimaly solve the elyptic fit')

        self.targetField = targetField

        x = self.dataRaw[0]#/self.s
        y = self.dataRaw[1]#/self.s
        z = self.dataRaw[2]#/self.s
                    
        logger.debug(f"data x: {x}")
        logger.debug(f"data y: {y}")
        logger.debug(f"data z: {z}")
        
        D = np.zeros( (9,self.n))
        for i in range(self.n): # todo, replace by matrix operations
            D[0][i] = x[i] * x[i];
            D[1][i] = y[i] * y[i];
            D[2][i] = z[i] * z[i];
            D[3][i] = 2.0 * y[i] * z[i];
            D[4][i] = 2.0 * x[i] * z[i];
            D[5][i] = 2.0 * x[i] * y[i];
            D[6][i] = 2.0 * x[i];
            D[7][i] = 2.0 * y[i];
            D[8][i] = 2.0 * z[i];
        logger.debug(f"D:\n{D}")
        
        # https://github.com/beattiea/TiltyIMU/blob/master/Tilty%20Software/Processing/Experiments/Testing%20Magnetometer%20calibration/ellipsoid_fit/ellipsoid_fit.m
        # v = ( D' * D ) \ ( D' * ones( size( x, 1 ), 1 ) );
            # Octave: B \ b
            # Python: x,resid,rank,s = np.linalg.lstsq(B,b)
        B = np.matmul(D,D.transpose())
        b = np.matmul(D,np.ones(self.n))        
        v,resid,rank,s  = np.linalg.lstsq(B,b, rcond=None)
        logger.debug(f"v:\n{v}")
       
        # https://github.com/beattiea/TiltyIMU/blob/master/Tilty%20Software/Processing/Experiments/Testing%20Magnetometer%20calibration/ellipsoid_fit/ellipsoid_fit.m
        # A = [ v(1) v(4) v(5) v(7); ...
        #       v(4) v(2) v(6) v(8); ...
        #       v(5) v(6) v(3) v(9); ...
        #       v(7) v(8) v(9) -1 ];
        Q = np.zeros(16)
        Q[ 0] = v[0];
        Q[ 1] = v[3]; 
        Q[ 2] = v[4];
        Q[ 3] = v[6]; 
        Q[ 4] = v[3];
        Q[ 5] = v[1]; 
        Q[ 6] = v[5];
        Q[ 7] = v[7]; 
        Q[ 8] = v[4];
        Q[ 9] = v[5];
        Q[10] = v[2];
        Q[11] = v[8];
        Q[12] = v[6];
        Q[13] = v[7];
        Q[14] = v[8];
        Q[15] = -1;
        Q = Q.reshape((4,4))
        logger.debug(f"Q:\n{Q}")
    
        center,resid,rank,s  = np.linalg.lstsq(- Q[0:3,0:3], [ v[6] , v[7] , v[8] ] ,rcond=None)
        self.b = center
        logger.debug(f"b:\n{self.b}")

        #% form the corresponding translation matrix
        T = np.eye(4,4)
        T[3,0:3] = center
        logger.debug(f"T:\n{T}")
        #% translate to the center
        R = np.matmul(np.matmul(T,Q),T.transpose())
        logger.debug(f"R:\n{R.round(3)}")

        # solve the eigenproblem
        [ evals, evecs]= np.linalg.eig( -R[0:3,0:3]/R[3,3]) 
        logger.debug(f"evals:{evals.round(3)}")
        logger.debug(f"evecs:\n{evecs.round(3)}")
        radii = np.sqrt( np.reciprocal(evals) )
        logger.debug(f"radii:      {radii.round(3)}")
        logger.debug(f"Norm radii: {np.linalg.norm(radii).round(3)}")
        

        # do ellipsoid fitting
        #scale = np.linalg.inv( np.diag(radii) ) * np.min(radii) 
        scale = np.linalg.inv( np.diag(radii) ) 
        comp = np.matmul(np.matmul(evecs,scale),evecs.transpose())        
        
        nDat = np.mean( np.sqrt( np.square(x) + np.square(y) + np.square(z) ))
        logger.debug(f"Norm Dat:   {nDat.round(3)}")
        
        A = np.asarray([x - self.b[0],
                         y - self.b[1],
                         z - self.b[2]])
        
        # normalize
        G = np.matmul(comp,A)
        nG = np.mean( np.sqrt( np.square(G[0]) + np.square(G[1]) + np.square(G[2]) ))
        logger.debug(f"G:\n{G.round(3)}")
        logger.debug(f"Norm G:\n{nG.round(3)}")
        
        self.Ai = comp  * ( targetField / nG * nDat )

        # normalize results
        mm = np.mean(np.sqrt(np.sum(np.square(self.dataRaw),axis=0)))
        self.b  /= mm
        self.Ai /= mm

        # final results
        logger.info(f"b  = {self.b}")
        logger.info(f"Ai =\n {self.Ai.round(6)}")

        return



    def plot(self):
        from matplotlib.gridspec import GridSpec
 
        # axis scaling
        Bl = MagneticSensor.Blim

        # create objects
        #fig = plt.figure(figsize=plt.figaspect(1.))
        fig = plt.figure(constrained_layout=True)
        plt_maximize()
        gs = GridSpec(2, 4, figure=fig)
 
        # create sub plots as grid
        ax = [None,None,None,None,None,None]
        ax[0] = fig.add_subplot(gs[0,0:2])
        ax[1] = fig.add_subplot(gs[1,0:2])
        
        ax[2] = fig.add_subplot(gs[0,2],projection='3d')
        ax[3] = fig.add_subplot(gs[1,2])
        ax[4] = fig.add_subplot(gs[0,3])
        ax[5] = fig.add_subplot(gs[1,3])
 


        # 3D figure 
        ax[2].plot(self.dataRaw[0,:]/self.s,self.dataRaw[1,:]/self.s,self.dataRaw[2,:]/self.s,'r',label='raw')
        ax[2].plot(self.dataCal[0,:]/self.s,self.dataCal[1,:]/self.s,self.dataCal[2,:]/self.s,'g',label='cal')
        ax[2].set_xlim(-Bl,Bl)
        ax[2].set_ylim(-Bl,Bl)
        ax[2].set_zlim(-Bl,Bl)
        ax[2].set_box_aspect([1.0, 1.0, 1.0])
        ax[2].set_aspect('equal')
        ax[2].set_proj_type('ortho')
        ax[2].legend()

        # x/y/z projections
        k1 = [0,0,1]; k2 = [1,2,2]
        k3 = ['Bx [G]','Bx [G]','By [G]']
        k4 = ['By [G]','Bz [G]','Bz [G]']               
        for k in range(3):
            a = ax[3+k]
            mra = np.mean( self.dataRaw[k1[k],:]/self.s )
            mrb = np.mean( self.dataRaw[k2[k],:]/self.s ) 
            mca = np.mean( self.dataCal[k1[k],:]/self.s )
            mcb = np.mean( self.dataCal[k2[k],:]/self.s ) 
            a.plot(self.dataRaw[k1[k],:]/self.s,self.dataRaw[k2[k],:]/self.s,'r',label='raw')
            a.plot(self.dataCal[k1[k],:]/self.s,self.dataCal[k2[k],:]/self.s,'g',label='cal')
            a.plot(mra,mrb,'ok',label='center_r_a_w')
            a.plot(mca,mcb,'ok',label='center_c_a_l')
            t = np.deg2rad( np.asarray(range(3601))/10)
            a.plot(self.targetField*np.sin(t),self.targetField*np.cos(t),'k')
            a.set_xlim(-Bl,Bl)
            a.set_ylim(-Bl,Bl)             
            a.set_aspect('equal')
            a.set_xlabel(k3[k])
            a.set_ylabel(k4[k])
            a.grid('on')
            

        # time signals raw
        s = self.s
        print("S::::",s)
        ax[0].plot(self.dataRaw[0]/s,'-r');
        ax[0].plot(self.dataRaw[1]/s,'-g');
        ax[0].plot(self.dataRaw[2]/s,'-b');
        ax[0].plot(self.fieldVec/s,'-k');
        ax[0].set_title('raw')

        # time signals after calibration
        ax[1].plot(self.dataCal[0]/s,'-r');
        ax[1].plot(self.dataCal[1]/s,'-g');
        ax[1].plot(self.dataCal[2]/s,'-b');
        ax[1].plot(self.fieldVecCal/s,'-k');
        ax[1].set_title(f'calibrated ({self.method})')

        for i in range(2):
            ax[i].legend(['Bx','By','Bz','abs(B)'],loc="center right")
            ax[i].set_xlabel('samples')
            ax[i].set_ylabel('signal [GAUSS]')        
            ax[i].grid('on')
            ax[i].set_ylim(-Bl,Bl)

        return


class Galaxy6D:
    # properties of magnetic sensor
    MAG_SENS_LOW   =  3000 # [LSB/G] Sensitivity of magnetometer [8G Range] // mgPerDigit = 4.35f
    MAG_SENSE_HIGH = 12000 # [LSB/G] Sensitivity of magnetometer [2G Range] // mgPerDigit = 1.22f
    # mag_sens_mT = mag_sens/10 # [LSB/mT] convert GAUSS to TEslas: 1G = 10-4T ==> 1G = 10mT
    
    # properties for plotting
    COL = ['.r','.b','.g']
    
    def __init__(self):
        self.ms = [None,None,None]  # Magnetic Sensors Class
        self.rate = None
        self.acc  = None
        self.doplot = False

    def __optfun(self,x,val):
        ''' 
            Parameters:
                x the current value array
                val: parameters,
        '''
        mm = [None,None,None]
        for i in range(3): mm[i] = self.ms[i].getMean()


        dataCal = [None,None,None]
        for i in range(3):
            Ai = x[( 0+i*12):( 9+i*12)].reshape(3,3)
            b  = x[( 9+i*12):(12+i*12)]
            r = R.from_euler('z', i*120, degrees=True)
            dataCal[i] = ( val[i].T - (b + mm[i]) ) @ Ai 
            dataCal[i] = r.apply( dataCal[i] )
            
        # dataCal is the array of corrected magentic vectors for all 
        # three sensors. The amplitude should be self.targetField for 
        # all three sensors and the direction should be equal after 
        # 120deg rotations. This leads to a multi target optimization
        # - getting the average direction of all three vectors
        # - scaling to the targetField
        # - calculating the geometrical difference to this vector
        #   for all three vectors
        # - calculating the mean squared error of all this differences
        #
        # Reasoning: the earth magnetic field amplitude does not change and the 
        # best estimate of the direction is the average of all three sensors.
        # Since we estimate the full amtrix Ai, it includes the rotations already
        #
    
        sumCal = dataCal[0] #+ dataCal[1]  + dataCal[2]
        normSumCal = np.linalg.norm(sumCal,axis=1,keepdims=True)
        #print("sum shape:",sumCal.shape)
        targetVec = (sumCal / normSumCal ) * self.targetField *2

        normTargetVec = np.linalg.norm(targetVec,axis=1)
        #print("minmax1: ",np.min(normTargetVec),np.max(normTargetVec))
        #print("minmax2: ",np.min(normSumCal),np.max(normSumCal))
        #print("Norm:", np.linalg.norm(targetVec , axis = 1))
        #print("sum shape:",targetVec.shape)
        #print("sum :",dataCal[i].shape)

        err = [0,0,0]
        for i in range(3):
            #ampCal = np.sqrt(np.sum(np.square(dataCal[i]),axis=1))
            #err[i] = np.sqrt(np.mean(np.square(ampCal-self.targetField )))
            errVec = (targetVec - dataCal[i])
            ampCal = np.sqrt(np.sum(np.square(errVec),axis=1))
            ofs =  np.where(ampCal<self.targetField/2,1,0)*self.targetField*5
            #print(" - ",np.min(ampCal),np.max(ampCal))
            #print(" - ",np.min(ofs),np.max(ofs))
            #ampCal = ampCal + ofs
            #print(" - ",np.min(ampCal),np.max(ampCal))
            err[i] = np.sqrt(np.mean(np.square(ampCal)))
            if self.doplot:
                plt.figure()
                plt.plot(dataCal[i],'-')
                plt.plot(targetVec,':')
                plt.show(block=False)
                plt.figure()
                plt.plot(normSumCal,'-')
                plt.show(block=False)
                plt.figure()
                plt.plot(errVec,'-')
                plt.show(block=False)
                plt.figure()
                plt.plot(ampCal,'-')
                plt.show()
                raise SystemExit
        
        #return err[0]
        return np.sqrt(np.mean(np.square(err)))



    def calRemainder(self):

        mm = [None,None,None]
        for i in range(3): mm[i] = np.mean(self.ms[i].dataRaw,axis=1)
        val = [self.ms[0].dataRaw,self.ms[1].dataRaw,self.ms[2].dataRaw]
        x = np.zeros(12*3)
        for i in range(3):
            x[( 0+i*12):( 9+i*12)] = self.ms[i].Ai.reshape(9,)
            x[( 9+i*12):(12+i*12)] = self.ms[i].b - mm[i]
        return self.__optfun(x,val)
        
    def estimateCalibration(self,targetField=1):

        self.Ai = np.eye(3)
        self.b = np.zeros(3)
        self.targetField = targetField

        mm = [None,None,None]
        for i in range(3): 
            mm[i] = self.ms[i].getMean()
            print(f"mm[{i}]:{mm[i]}")
            
        val = [self.ms[0].dataRaw,self.ms[1].dataRaw,self.ms[2].dataRaw]
            

        relTol = 0.2; 
        mr = [0,0,0]; tol = [0,0,0]
        for i in range(3):
            mr[i] = (np.max(self.ms[i].dataRaw,axis=1)-np.min(self.ms[i].dataRaw,axis=1))/2
            print(f"param {i}")
            tol[i] = [ ( mm[i][j] - relTol*mr[i][j]  , mm[i][j] + relTol*mr[i][j] ) for j in range(3) ]  
            print("mr ",mr[i])
            print("mm ",mm[i])
            print("tol",tol[i])
        
        bnds = ( (None,None), (None,None), (None,None), \
                 (None,None), (None,None), (None,None), \
                 (None,None), (None,None), (None,None), \
                  tol[0][0], tol[0][1], tol[0][2], \
                (None,None), (None,None), (None,None), \
                 (None,None), (None,None), (None,None), \
                 (None,None), (None,None), (None,None), \
                  tol[1][0], tol[1][1], tol[1][2], \
                (None,None), (None,None), (None,None), \
                 (None,None), (None,None), (None,None), \
                 (None,None), (None,None), (None,None), \
                  tol[2][0], tol[2][1], tol[2][2] \
                ) 


        x0 = np.concatenate( ( np.eye(3).reshape( (9,) ), np.zeros((3,))) )
        x0 = np.concatenate( ( x0,x0,x0) )        
        
        x0 = [None,None,None]
        for i in range(3):
            print(np.mean(tol[i],axis=0))
            x0[i] = np.concatenate( ( np.eye(3).reshape( (9,) ), np.mean(tol[i],axis=1) ) ) 
        x0 = np.concatenate( ( x0[0],x0[1],x0[2]) )        


        logger.info(f'Optimize with start value size:{x0.shape} val:{x0} to target {self.targetField}' )
        logger.info(f'Bounds:\n{bnds}' )

        res = scipy.optimize.minimize(self.__optfun,x0, args=val, bounds=bnds,tol=1e-5,method='Powell')
        #res = scipy.optimize.minimize(optfun,x0, args=val,tol=1e-3)

        if res.success:
            logger.info(f'Optimize successfully: {res.fun} @ {res.x.shape}' )
        else:
            logger.warning(f'Optimize failed: {res.message} \n >>> {res.x.shape} val:{res.x}' )

        logger.info(f'New Calib Params:' )
        for i in range(3):
            Ai = res.x[( 0+i*12):( 9+i*12)].reshape(3,3)
            b0  = res.x[( 9+i*12):(12+i*12)]
            b = b0 + 0*mm[i]
            print('b0:',b0)
            self.ms[i].setAib(Ai,b,self.targetField)
            logger.info(f' {i}: {b} \n Ai: {Ai}' )

        #TODO: Change to function, set Ai,b


        
    def loadFile(self,filename):
        ''' Load a file and stores '''
        result = [] 
        timestamp =  []
        with open(filename, newline='') as csvfile:    
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                if not row[1].isnumeric(): continue
                tm = row[1]
                sensors = row[2:-1]
                #print(tm,' ==> ',sensors)
                result.append(sensors)
                timestamp.append(tm)
                
        result = np.np.asarray(result, dtype=np.float32)
        result = np.transpose(result)
        self.ms[0] = MagneticSensor(result[:, 0: 3] / self.MAG_SENS_LOW)
        self.ms[1] = MagneticSensor(( result[:, 3: 6])/ self.MAG_SENS_LOW)
        self.ms[2] = MagneticSensor(( result[:, 6: 9])/ self.MAG_SENS_LOW)
        self.acc   = MagneticSensor(( result[:, 9:12]) )
        self.rate  = MagneticSensor(( result[:,12:15]))
                
        timestamp = np.np.asarray(timestamp, dtype=np.float32)
        result = np.transpose(result)
        timestamp = timestamp- timestamp[0]
        self.ms[0].setTime(timestamp)
        self.ms[1].setTime(timestamp)
        self.ms[2].setTime(timestamp)
        self.acc.setTime(timestamp) 
        self.rate.setTime(timestamp) 

    def loadFilePickle(self,filename):
        logger.info(f'load file {filename}')
        with open( filename, 'rb') as handle:
            data = pickle.load(handle)
            
        logger.info(f'loaded data with shape  {data.shape}')
            
        self.rate  = MagneticSensor(( data[:, 1: 4].T))
        self.acc   = MagneticSensor(( data[:, 4: 7].T) )
        self.ms[0] = MagneticSensor(( data[:, 7:10].T)/ self.MAG_SENS_LOW)
        self.ms[1] = MagneticSensor(( data[:,10:13].T)/ self.MAG_SENS_LOW)
        self.ms[2] = MagneticSensor(( data[:,13:16].T)/ self.MAG_SENS_LOW)
                
        timestamp = np.asarray(data[:, 0],dtype=np.float32) 
        timestamp = timestamp- timestamp[0]
        self.ms[0].setTime(timestamp)
        self.ms[1].setTime(timestamp)
        self.ms[2].setTime(timestamp)
        self.acc.setTime(timestamp) 
        self.rate.setTime(timestamp) 

if __name__ == '__main__':
    logger.info(f"Connot run the module {__name__} itself")