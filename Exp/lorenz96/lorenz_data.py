import numpy as np
import torch
from scipy.integrate import odeint

def lorenz96(x, t, F=8):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

def get_data(N=40, F=8, T=10000, dt=0.01, train_ratio=0.8):
    x0 = F * np.ones(N)
    x0[0] += 0.01
    t = np.arange(0.0, T, dt)
    
    x = odeint(lorenz96, x0, t, args=(F,))
    
    data = torch.tensor(x, dtype=torch.float32)
    X = data[:-1]
    Y = data[1:]
    
    split = int(train_ratio * len(X))
    train_data = (X[:split], Y[:split])
    test_data = (X[split:], Y[split:])
    
    return train_data, test_data, dt

if __name__ == "__main__":
    train, test, dt = get_data()
    print(f"Data generated. Train shape: {train[0].shape}")

