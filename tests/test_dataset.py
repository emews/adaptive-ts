import numpy as np
from adaptive_ts.dataset import Dataset
rng = np.random.default_rng(123)
X = rng.normal(size = (100,2))
# add seed
ints = rng.integers(low=1,high=7,size=100).reshape(-1,1)
X = np.concatenate([X,ints],axis=1)
Y = np.exp(10*X[:,0]) + X[:,-1]
Y = Y.squeeze()

# save manual
yraw = np.log(Y)
ym, ystd = yraw.mean(), yraw.std()
y_ref = (yraw - ym) / ystd



def test_dataset():
    data = Dataset(X=X,Y=Y)
    assert np.allclose(y_ref,data.data['Y'].values)

def test_append():
    data = Dataset(X=X,Y=Y)

    newX = 2 * X
    newY = np.exp(10*newX[:,1]) + newX[:,-1]
    data.append(X=newX,Y=newY)
    y2 = (np.log(newY) - ym) / ystd
    y_test = np.concat([y_ref,y2])
    assert np.allclose(y_test,data.data['Y'].values)
    



