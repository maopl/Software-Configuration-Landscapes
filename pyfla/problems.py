import random
import math
import itertools
import pandas as pd
from functools import lru_cache

class NK:
    """
    Class for generating the Kauffman's NK landscape model.
    
    Parameters
    ----------
    n : int
        Number of elements (positions)
    k : int, default=1
        Degree of interaction among elements. Ranging from 1 to n - 1.
    """
    
    def __init__(self, n, k, exponent=1):
        self.n = n
        self.k = k
        self.exponent = exponent
        self.elements = range(n)
        self.dependence = [
            tuple(sorted([e] + random.sample(set(self.elements) - set([e]), k)))
            for e in self.elements
        ]
        self.values = {}
    
    @lru_cache(maxsize=None)
    def evaluate(self, config):
        total_value = 0.0
        config = tuple(config)  
        for e in self.elements:
            key = (e,) + tuple(config[i] for i in self.dependence[e])
            if key not in self.values:
                self.values[key] = random.random()
            total_value += self.values[key]
        
        total_value /= self.n
        if self.exponent != 1:
            total_value = math.pow(total_value, self.exponent)
        
        return total_value
    
    def get_data(self):
        all_configs = itertools.product((0, 1), repeat=self.n)
        config_values = {config: self.evaluate(config) for config in all_configs}
        
        data = pd.DataFrame(list(config_values.items()), columns=["config", "fitness"])
        data['config'] = data['config'].apply(list)
        return data