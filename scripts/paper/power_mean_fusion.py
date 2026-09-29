import os as _os, sys as _sys
# run from anywhere: load datasets from cTSNE2026/TOTAL
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'data'))

# Generalized cell-driven fusion: weighted power-mean operator
#   D_fused^(p) = ( sum_m w_m * Dtilde_m^p )^(1/p),  p in [1,inf]
#   p=1 -> sum-fusion ; p->inf -> max-fusion.
# Validate: select p by the C-way eigengap (label-free) and compare NMI to fixed sum/max.
import numpy as np, scipy.io as sio, pandas as pd, scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import svd, eigh
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
def dense(X): return np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
def load(n):
    M={'Buettner':'Test_1_mECS.mat','Kolod':'Test_2_Kolod.mat','Pollen':'Test_3_Pollen.mat',
       'Usoskin':'Test_4_Usoskin.mat','Zeisel':'Test_5_Zeisel.mat'}
    if n in M: m=sio.loadmat(M[n]); return dense(m['in_X']).astype(float), m['true_labs'].ravel()
    if n=='Patel': d=pd.read_csv('patel_editted.csv'); return d.drop(columns=['Label']).values.astype(float), d['Label'].values
    if n=='Goolam': g=pd.read_csv('Goolam_editted.csv',index_col=0); return g.iloc[:,:-1].values.astype(float), g.iloc[:,-1].values
def Dy(X):
    B=(X>0).astype(float); nB=1-B; nTT=B@B.T; nFF=nB@nB.T; nTF=B@nB.T; nFT=nB@B.T
    D=2*nTF*nFT/(nTT*nFF+nTF*nFT+1e-12); np.fill_diagonal(D,0); return D
def Dl(X,v=.8):
    U,S,Vt=svd(X,full_matrices=False); r=max(int(np.searchsorted(np.cumsum(S)/S.sum(),v)+1),1)
    return squareform(pdist((U[:,:r]*S[:r])@Vt[:r,:],'chebyshev'))
def Df(X): return squareform(pdist(X,'minkowski',p=.25))
def nm(D):
    o=D[~np.eye(D.shape[0],dtype=bool)]; m=np.median(o[o>0]) if np.any(o>0) else 1; return D/(m+1e-12)
def powmean(views,w,p):
    if p>=64: return np.maximum.reduce([w_*0+V for w_,V in zip(w,views)]) if False else np.maximum.reduce(views)
    acc=sum(wm*np.power(V,p) for wm,V in zip(w,views))
    return np.power(acc,1.0/p)
def affinity(D,K=20,mu=.5):
    Ds=np.sort(D,1); mk=Ds[:,1:K+1].mean(1); e=(mk[:,None]+mk[None,:]+D)/3+1e-12
    W=np.exp(-(D**2)/(mu*e)); np.fill_diagonal(W,0); return (W+W.T)/2
def eigengap(W,C):
    d=W.sum(1); di=1/np.sqrt(d+1e-12); L=np.eye(W.shape[0])-(di[:,None]*W*di[None,:])
    ev=np.sort(eigh(L)[0]); return ev[C]-ev[C-1]   # gap between C-th and (C+1)-th smallest (0-idx C)
def spec(W,y,C):
    return nmi(y,SpectralClustering(C,affinity='precomputed',assign_labels='discretize',random_state=0).fit_predict(W),average_method='max')
SP={'Buettner':37.9,'Kolod':27.9,'Pollen':51.0,'Usoskin':78.1,'Patel':53.3,'Goolam':68.6,'Zeisel':46.0}
PS=[1,2,4,8,16,64]
print(f"{'Dataset':9s}{'p*=gap':>7s}{'NMI@p*':>8s}{'NMI sum(p1)':>12s}{'NMI max':>8s}{'NMI best-p':>11s}")
sumv=[];maxv=[];selv=[];bestv=[]
for n in ['Buettner','Kolod','Pollen','Usoskin','Patel','Goolam','Zeisel']:
    X,y=load(n); C=len(np.unique(y))
    V=[nm(Dy(X)),nm(Df(X)),nm(Dl(X))]
    b=SP[n]/100; w=(0.8,0.1,0.1) if b>0.5 else (1/3,1/3,1/3)
    gaps={};nmis={}
    for p in PS:
        Dp=powmean(V,w,p); W=affinity(Dp); gaps[p]=eigengap(W,C); nmis[p]=spec(W,y,C)
    pstar=max(gaps,key=gaps.get)            # eigengap-selected p (label-free)
    pbest=max(nmis,key=nmis.get)            # oracle best p
    sumv.append(nmis[1]); maxv.append(nmis[64]); selv.append(nmis[pstar]); bestv.append(nmis[pbest])
    print(f"{n:9s}{pstar:7d}{nmis[pstar]:8.3f}{nmis[1]:12.3f}{nmis[64]:8.3f}{nmis[pbest]:11.3f}",flush=True)
import numpy as np
sumv,maxv,selv,bestv=map(np.array,(sumv,maxv,selv,bestv))
print(f"\nMEAN  eigengap-sel p*={selv.mean():.3f}  sum(p=1)={sumv.mean():.3f}  max(p=inf)={maxv.mean():.3f}  oracle-best-p={bestv.mean():.3f}")
print(f"eigengap-p vs best fixed(max(sum,max)) per-dataset: sel>=fixed on {int((selv>=np.maximum(sumv,maxv)-1e-9).sum())}/7")
