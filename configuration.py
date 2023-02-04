import argparse

parser = argparse.ArgumentParser(prog = 'AAPNN',
                                 description = 'Aging-Aware Training for Printed Neural Networks')

# printing-related hyperparameters for pNNs
parser.add_argument('--gmin', type=float, default=0.01, help='minimal printable conductance value')
parser.add_argument('--gmax', type=float, default=10.,  help='maximal printable conductance value')
parser.add_argument('--T', type=float, default=0.1,  help='measuring threshold')
parser.add_argument('--m', type=float, default=0.3,  help='measuring margin')

parser.add_argument('--ACT_eta1', type=float, default=0.134,  help='eta_1 for activation function')
parser.add_argument('--ACT_eta2', type=float, default=0.962,  help='eta_2 for activation function')
parser.add_argument('--ACT_eta3', type=float, default=0.183,  help='eta_3 for activation function')
parser.add_argument('--ACT_eta4', type=float, default=24.10,  help='eta_4 for activation function')
parser.add_argument('--NEG_eta1', type=float, default=-0.104,  help='eta_1 for negative weight function')
parser.add_argument('--NEG_eta2', type=float, default=0.899,  help='eta_2 for negative weight function')
parser.add_argument('--NEG_eta3', type=float, default=-0.056,  help='eta_3 for negative weight function')
parser.add_argument('--NEG_eta4', type=float, default=3.858,  help='eta_4 for negative weight function')
# parser.add_argument('--ACT_eta1', type=float, default=2.1493e-03,  help='eta_1 for activation function')
# parser.add_argument('--ACT_eta2', type=float, default=1.0022e+00,  help='eta_2 for activation function')
# parser.add_argument('--ACT_eta3', type=float, default=2.3549e-03,  help='eta_3 for activation function')
# parser.add_argument('--ACT_eta4', type=float, default=1.2297e+01,  help='eta_4 for activation function')
# parser.add_argument('--NEG_eta1', type=float, default=2.1493e-03,  help='eta_1 for negative weight function')
# parser.add_argument('--NEG_eta2', type=float, default=1.0022e+00,  help='eta_2 for negative weight function')
# parser.add_argument('--NEG_eta3', type=float, default=2.3549e-03,  help='eta_3 for negative weight function')
# parser.add_argument('--NEG_eta4', type=float, default=1.2297e+01,  help='eta_4 for negative weight function')

# machine-learning-related hyperparameters
parser.add_argument('--hidden', type=list, default=[3],   help='topology of the hidden layers')
parser.add_argument('--SEED', type=int, default=0,   help='random seed')
parser.add_argument('--DEVICE', metavar='device', type=str, default='gpu', help='device for training')
parser.add_argument('--TIMELIMITATION', type=float, default=45, help='maximal running time (in hour)')
parser.add_argument('--PATIENCE', type=int, default=500, help='patience for early-stopping')
parser.add_argument('--EPOCH', type=int, default=10**10, help='maximal epochs')
parser.add_argument('--DATASET', type=int, default=0, help='training dataset')
parser.add_argument('--LR', type=float, default=0.1,   help='learning rate')
parser.add_argument('--PROGRESSIVE', type=bool, default=False,   help='whether the learning rate will be adjusted')
parser.add_argument('--LR_PATIENCE', type=int, default=100, help='patience for updating learning rate')
parser.add_argument('--LR_DECAY', type=float, default=0.5, help='decay of learning rate for progressive lr')
parser.add_argument('--LR_MIN', type=float, default=1e-4, help='minimal learning rate for stop training')

# aging-aware-related hyperparameters
parser.add_argument('--MODE', type=str, default='nominal', help='training mode: aging, nominal')
parser.add_argument('--M_train', type=int, default=50,  help='number of stochastic aging models during training')
parser.add_argument('--K_train', type=int, default=10,  help='number of temporal sampling during training')
parser.add_argument('--M_test',  type=int, default=500, help='number of stochastic aging models for testing')
parser.add_argument('--K_test',  type=int, default=500,  help='number of temporal sampling for testing')
parser.add_argument('--t_test_max',  type=int, default=10,  help='test time interval')
parser.add_argument('--integration', type=str, default='MC', help='method for integration: Monte-Carlo, Gaussian Quadrature')

# variation-related hyperparameters
parser.add_argument('--VARIATION', type=bool, default=False,  help='whether the variation will be considered during training')
parser.add_argument('--N_train', type=int, default=20,  help='number of sampling for variation during training')
parser.add_argument('--e_train', type=int, default=0.1, help='variation during training')
parser.add_argument('--N_test',  type=int, default=100, help='number of sampling for variation for testing')
parser.add_argument('--e_test', type=int, default=0.1, help='variation for testing')

# log-file-related information
parser.add_argument('--projectname', type=str,   help='name of the project')
parser.add_argument('--temppath', type=str, default='/temp',   help='path to temp files')
parser.add_argument('--logfilepath', type=str, default='/log',   help='path to log files')
parser.add_argument('--recording', type=bool, default=False,   help='save information in each epoch')
parser.add_argument('--recordpath', type=str, default='/record',   help='save information in each epoch')
parser.add_argument('--savepath', type=str, default='/experiment',   help='save information in each epoch')
parser.add_argument('--loglevel', type=str, default='info',   help='level of message logger')
