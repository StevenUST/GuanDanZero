from guandan_net_tensorflow import GuandanNetForTwo
import numpy as np

base_path = './saved_model/'
meta_path = 'guandan_model_v1_0.model.meta'

if __name__ == "__main__":
    np.random.seed(111)
    
    model = GuandanNetForTwo()
    model.restore_model(base_path, meta_path)
    
    a = np.random.rand(1, 70)
    b = np.random.rand(1, 16, 15, 1)
    c = np.random.rand(1, 70)
    d = np.random.rand(1, 16, 15, 1)
    e = np.random.rand(1, 17)
    f = np.random.rand(1, 15)
    g = np.random.rand(1, 13)
    
    t1, t2 = model.get_prob(a, b, c, d, e, f, g)
    
    print(t1)
    print(t2)
    
    model.restore_model(base_path, meta_path)
    
    t3, t4 = model.get_prob(a, b, c, d, e, f, g)
    
    print(t3)
    print(t4)
    