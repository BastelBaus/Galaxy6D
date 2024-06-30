System Function Overview
=====================================


Earth Magnetic Field Model
---------------------------------------------------

Let :math:`\vec{E}` be the earth magnetic field with comnponents in all 3 dimensions.

.. math::   \vec{E} = \begin{pmatrix} m_x \\ m_y \\ m_z \end{pmatrix}

Let :math:`\hat{\vec{M}}_i` be measured magnetic field of the :math:`i^{th}` 
magnetic field sensor with :math:`i=1..3`. Let :math:`\vec{O}_i` be the offset 
and :math:`S_i` be the scale error of the magnetic sensors.

.. math::   \vec{O}_i = \begin{pmatrix} o_{x,i} \\ o_{y,i} \\ o_{z,i} \end{pmatrix} 
            \space \space \space  and \space \space \space 
            S_i = \begin{pmatrix} s_{x,i} & 0 & 0 \\ 0 & s_{y,i} & 0 \\ 0 & 0 & s_{z,i} \end{pmatrix} 

Let $\vec{K}_i$ be the signal at each sensor induced by the permanent magnets. With the rotation matrixes $D_i$ for each magnetic sensor we can get the measured magnetic field $\hat{\vec{M}}_i$ for each sensor by;

.. math::   \hat{\vec{M}}_i = D_i \cdot S_i \cdot \vec{E} + \vec{O}_i + \vec{K}_i

For ease of formulation we define the $1^{st}$ as reference of the Galaxy6D. Then we can define the three 
rotation matrixes quite easily by 3 angles (:math:`\alpha_1=\beta_1=\gamma_1=0?`) defining the overall knob
position as well as 2 angles (:math:`alpha_2=120°` and :math:`alpha_3=-120°`) from the design of the position 
of the magnetic sensors (see also `Wikipedia <https://en.wikipedia.org/wiki/Rotation_matrix>`__).

.. math::   D_1 = \begin{pmatrix} cos(\alpha_1) & -sin(\alpha_1) & 0 \\ sin(\alpha_1) & cos(\alpha_1) & 0 \\ 0 & 0 & 1 \end{pmatrix} *
            \begin{pmatrix} cos(\beta_1) & 0 & sin(\beta_1) & 0 \\ 0 & 1 & 0 \\ -sin(\beta_1) & 0 & cos(\beta_1)  \end{pmatrix} *          
            \begin{pmatrix} 1 & 0 & 0 \\ 0 & cos(\gamma_1) & -sin(\gamma_1) \\ 0 & sin(\gamma_1) & cos(\gamma_1) \end{pmatrix} =
            \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}          

            D_2 = D_1 \cdot \begin{pmatrix} cos(\alpha_2) & -sin(\alpha_2) & 0 \\ sin(\alpha_2) & cos(\alpha_2) & 0 \\ 0 & 0 & 0 \end{pmatrix} = 
            \begin{pmatrix} -0.5 & -\sqrt{3}/2 & 0 \\ \sqrt{3}/2 & -0.5 & 0 \\ 0 & 0 & 0 \end{pmatrix}     
            
            \space \space \space and \space \space \space 
            D_3 = D_1 \cdot \begin{pmatrix} cos(\alpha_3) & -sin(\alpha_3) & 0 \\ sin(\alpha_3) & cos(\alpha_3) & 0 \\ 0 & 0 & 0 \end{pmatrix} = 
            \begin{pmatrix} -0.5 & \sqrt{3}/2 & 0 \\ -\sqrt{3}/2 & -0.5 & 0 \\ 0 & 0 & 0 \end{pmatrix} 
    
As first test, we will only rotate the whole knob, this considering $\vec{K}_i$ just be included in $\vec{O}_i$ as additional (temporal constant) offset. We will design a Kalman filter to fuse the 3 sensors and estimate the earth magnetic field as well as the scale factor errors and offset. Basically we are only interested in the offset (and there only the part induced by the permanent magents) but this will be done on the second part of the model.


We start with the state vector $\vec{S}$ holding the 3D earth magnetic field ($\vec{M}$), the 3 offsets  ($\vec{O}_i$) and scale factor errors  (diagonal elements of $S_i$ namely $diag{S}_i$) of each magnetic sensors. In total this sum's up to 21 states.
$$ \vec{S} = \begin{pmatrix} \vec{M} & \vec{O}_1 & \vec{O}_2 & \vec{O}_3 & diag(S_1) & diag(S_2) & diag(S_3) \end{pmatrix}^T $$

The Observation matrix $H$ can be calculated as;

.. math::   \hat{\vec{M}} = \begin{pmatrix}  \hat{\vec{M}}_1 & \hat{\vec{M}}_2 & \hat{\vec{M}}_3 \end{pmatrix} ^T = 
            \begin{pmatrix}  \hat{\vec{M}}_1 & \hat{\vec{M}}_2 & \hat{\vec{M}}_3 \end{pmatrix} * \vec{S}          
        


We will use a Python toolbox to implement the Kalman filter (see `nonlinest <https://github.com/boschresearch/nonlinest>`__). 


Adding angular rates to the Kalman filter 
---------------------------------------------------



<br><hr> 
\>> Back to  **[main page](index.md)** <br>
\>> Go to **[main repository](https://github.com/BastelBaus/Galaxy6D)**


