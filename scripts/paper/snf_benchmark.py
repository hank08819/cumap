import os as _os, sys as _sys
# run from anywhere: load datasets from cTSNE2026/TOTAL
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'data'))

"""
SNF fusion baseline over the THREE cell-driven views (Yule, L-Chebyshev, fractional).
Same views as c-TSNE/CUMAP; only the fusion operator differs (SNF cross-diffusion
vs our max/sum). Native SNF downstream = spectral clustering at ground-truth k.
Reports max-normalized NMI to match the paper.
"""
import numpy as np, scipy.io as sio, pandas as pd
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import svd
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

rng = 0
np.random.seed(rng)

def load(name):
    if name in ('Buettner','Kolod','Pollen','Usoskin'):
        f={'Buettner':'Test_1_mECS.mat','Kolod':'Test_2_Kolod.mat',
           'Pollen':'Test_3_Pollen.mat','Usoskin':'Test_4_Usoskin.mat'}[name]
        m=sio.loadmat(f); X=m['in_X'].astype(float); y=m['true_labs'].ravel()
    elif name=='Patel':
        df=pd.read_csv('patel_editted.csv'); y=df['Label'].values
        X=df.drop(columns=['Label']).values.astype(float)
    elif name=='Goolam':
        dg=pd.read_csv('Goolam_editted.csv',index_col=0)
        y=dg.iloc[:,-1].values; X=dg.iloc[:,:-1].values.astype(float)
    return X, y

# ---- three cell-driven view distances (as defined in the paper) ----
def D_yule(X):
    B=(X>0).astype(float); nB=1.0-B
    nTT=B@B.T; nFF=nB@nB.T; nTF=B@nB.T; nFT=nB@B.T
    D=2*nTF*nFT/(nTT*nFF+nTF*nFT+1e-12)   # = 1 - Yule's Q, in [0,2]
    np.fill_diagonal(D,0.0); return D

def D_lcheb(X, var=0.80):
    U,S,Vt=svd(X,full_matrices=False)
    r=int(np.searchsorted(np.cumsum(S)/S.sum(), var)+1); r=max(r,1)
    Xr=(U[:,:r]*S[:r])@Vt[:r,:]
    return squareform(pdist(Xr,'chebyshev'))

def D_frac(X, f=0.25):
    # (sum |xi-xj|^f)^{1/f} = minkowski with p=f
    return squareform(pdist(X,'minkowski',p=f))

def norm_med(D):
    off=D[~np.eye(D.shape[0],dtype=bool)]
    m=np.median(off[off>0]) if np.any(off>0) else 1.0
    return D/(m+1e-12)

# ---- SNF (Wang et al. 2014) ----
def affinity(D, K=20, mu=0.5):
    n=D.shape[0]; Ds=np.sort(D,axis=1)
    meanK=Ds[:,1:K+1].mean(axis=1)
    eps=(meanK[:,None]+meanK[None,:]+D)/3+1e-12
    W=np.exp(-(D**2)/(mu*eps)); W=(W+W.T)/2
    return W

def status(W):
    d=W.sum(1)-np.diag(W)
    P=W/(2*(d[:,None]+1e-12)); np.fill_diagonal(P,0.5); return P

def local(W, K=20):
    n=W.shape[0]; S=np.zeros_like(W)
    idx=np.argsort(-W,axis=1)[:,:K]
    for i in range(n):
        nb=idx[i]; s=W[i,nb]; s=s/(s.sum()+1e-12); S[i,nb]=s
    return S

def snf(Ds, K=20, mu=0.5, t=20):
    Ws=[affinity(D,K,mu) for D in Ds]
    Ps=[status(W) for W in Ws]; Ss=[local(W,K) for W in Ws]
    M=len(Ws)
    for _ in range(t):
        newP=[]
        for m in range(M):
            other=sum(Ps[l] for l in range(M) if l!=m)/(M-1)
            P=Ss[m]@other@Ss[m].T; P=(P+P.T)/2
            newP.append(status(P))
        Ps=newP
    Pc=sum(Ps)/M; Pc=(Pc+Pc.T)/2
    Pc[Pc<0]=0; return Pc

def spec_nmi(W, y, k):
    sc=SpectralClustering(n_clusters=k, affinity='precomputed',
                          assign_labels='discretize', random_state=rng)
    pred=sc.fit_predict(W)
    return nmi_score(y, pred, average_method='max')

OURS={'Buettner':0.83,'Kolod':1.00,'Pollen':0.97,'Usoskin':0.87,'Patel':0.89,'Goolam':0.79}
rows=[]
for name in ['Buettner','Kolod','Pollen','Usoskin','Patel','Goolam']:
    X,y=load(name); k=len(np.unique(y))
    Dy=norm_med(D_yule(X)); Dl=norm_med(D_lcheb(X)); Df=norm_med(D_frac(X))
    # single-view spectral (sanity) + SNF fusion
    nmi_y=spec_nmi(affinity(Dy),y,k)
    nmi_l=spec_nmi(affinity(Dl),y,k)
    nmi_f=spec_nmi(affinity(Df),y,k)
    Pc=snf([Dy,Dl,Df]); nmi_snf=spec_nmi(Pc,y,k)
    rows.append((name,k,OURS[name],nmi_snf,nmi_y,nmi_l,nmi_f))
    print(f"{name:9s} k={k:2d}  ours={OURS[name]:.2f}  SNF={nmi_snf:.3f}  "
          f"[viewY={nmi_y:.3f} viewL={nmi_l:.3f} viewF={nmi_f:.3f}]", flush=True)

# paired Wilcoxon ours vs SNF
from scipy.stats import wilcoxon
ours=np.array([r[2] for r in rows]); snfv=np.array([r[3] for r in rows])
try:
    W,p=wilcoxon(ours,snfv,alternative='greater')
    print(f"\nPaired Wilcoxon (ours>SNF), n={len(rows)}: W+={W:.1f}, one-sided p={p:.4g}")
except Exception as e:
    print('wilcoxon:',e)
print("mean ours=%.3f  mean SNF=%.3f  wins=%d/%d"%(ours.mean(),snfv.mean(),(ours>snfv).sum(),len(rows)))
