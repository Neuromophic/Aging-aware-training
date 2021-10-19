import numpy as np

class Linear_aging_model_sampler:
    def __init__(self, dists_linear, param_data_linear_approx):
        self.dist = dists_linear
        self.param_data = param_data_linear_approx
        
    def get_models(self, number_of_models=1):
        return [PiecewiseLinear(params=self.transform_sample(self._get_param_sample())) 
                                for i in range(number_of_models)]
    
    @staticmethod
    def transform_sample(x):
        ''' negates first entries '''
        return np.r_[-x[:-1], x[-1]]
        
        
    def _get_param_sample(self):
        ''' samples from the param distributions '''
        return np.array([self.dist[c].rvs(1)*self.param_data[c].std() 
                         for c in self.param_data.columns]).ravel()

    
class PiecewiseLinear:
    def __init__(self, params):
        self.params = params 
        
    def __call__(self, x):
        m0, m1, b1 = self.params
        b0 = 1

        mod = lambda z : max(z*m0 + b0, z*m1 + b1)

        return np.array([mod(xi) for xi in x])