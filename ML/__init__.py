"""
__init__.py for ML/
"""

# All the imports we need

import theano
import numpy as np
import theano.tensor as T
from theano import sandbox


# for linreg_gradDes.py in ML/

from linreg_gradDes import LinearReg, LinearReg_loaded

# for gradDes.py in ML/

from gradDes import LogReg

# for NN.py in ML/
import NN
#from NN import Layer, cost_functional, cost_functional_noreg, gradientDescent_step, predicted_logreg, MLP
#from NN import Thetab, build_cost_functional, build_gradDescent_step, MLP, Universal_Approx
#from NN import build_gradDescent_step, MLP, Universal_Approx
from NN import MLP, Universal_Approx


# for LSTM.py in ML/
import LSTM
#from LSTM import gates, Thetab, Thetabtheta, Feedforward


# temporary for Herta's LSTM implementation
import LSTM_Herta
#from LSTM_Herta import Thetab_right, Thetabtheta_right, ThetabthetaW_right, Feedforward_g_right, Feedforward_ifo_right, LSTM_Model_right, MemoryBlock_right

# for LSTM_Right.py in ML/
import LSTM_Right

# for GRUs_Right.py in ML/
import GRUs_Right

# for SVM.py in ML/
import SVM
