# codes

## Environment
Tested on Python 3.9.16.

```
pip install -r requirements.txt
```

## Numerical experiment
```
zn=10 #number of latent variable
theta=0 #rotation angle of basis
Gdense=0.2 #graph density
Mdense=0.3  #rho
distance=0.0  #delta in translation vector
m=5  #number of MLP layer for mixing function 
python main_numerical.py --rotation "$theta" --z-n "$zn" --x-n "$zn" --nn "$zn" --DAG-dense "$Gdense" --mask-dense "$Mdense" --distance "$distance" --n-mixing-layer "$m"
```

## Image-based experiment
```
# multi-balls with stationary position
python main_balls.py
```
