import os as _os, sys as _sys
# run from anywhere: load datasets from cTSNE2026/TOTAL
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'data'))

import numpy as np, scipy.io as sio, pandas as pd, scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import svd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from scipy.stats import wilcoxon
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
def aff(D,K=20,mu=.5):
    Ds=np.sort(D,1); mk=Ds[:,1:K+1].mean(1); e=(mk[:,None]+mk[None,:]+D)/3+1e-12; W=np.exp(-(D**2)/(mu*e)); return (W+W.T)/2
def stat(W): d=W.sum(1)-np.diag(W); P=W/(2*(d[:,None]+1e-12)); np.fill_diagonal(P,.5); return P
def loc(W,K=20):
    S=np.zeros_like(W); idx=np.argsort(-W,1)[:,:K]
    for i in range(W.shape[0]): nb=idx[i]; s=W[i,nb]; S[i,nb]=s/(s.sum()+1e-12)
    return S
def snf(Ds,K=20,mu=.5,t=20):
    Ws=[aff(D,K,mu) for D in Ds]; Ps=[stat(W) for W in Ws]; Ss=[loc(W,K) for W in Ws]; M=len(Ws)
    for _ in range(t):
        nP=[Ss[m]@(sum(Ps[l] for l in range(M) if l!=m)/(M-1))@Ss[m].T for m in range(M)]
        Ps=[stat((P+P.T)/2) for P in nP]
    Pc=sum(Ps)/M; Pc=(Pc+Pc.T)/2; Pc[Pc<0]=0; return Pc
def ec(D,y,k,seed):
    D=(D+D.T)/2; np.fill_diagonal(D,0)
    emb=TSNE(2,metric='precomputed',init='random',perplexity=30,random_state=seed).fit_transform(D)
    return nmi(y,KMeans(k,n_init=10,random_state=seed).fit_predict(emb),average_method='max')
SP={'Buettner':37.9,'Kolod':27.9,'Pollen':51.0,'Usoskin':78.1,'Patel':53.3,'Goolam':68.6,'Zeisel':46.0}
SEEDS=[0,1,2,3,4]
print(f"{'Dataset':9s}{'spars':>6s}{'ours(mean±sd)':>16s}{'SNFtsne(mean±sd)':>18s}{'SNFspec':>9s}")
O=[];T=[]
for n in ['Buettner','Kolod','Pollen','Usoskin','Patel','Goolam','Zeisel']:
    X,y=load(n); k=len(np.unique(y))
    dy,dl,df=nm(Dy(X)),nm(Dl(X)),nm(Df(X))
    b=SP[n]/100; w=(.8,.1,.1) if b>.5 else (1/3,1/3,1/3)
    Dmax=np.maximum.reduce([dy,dl,df]); Dsum=w[0]*dy+w[1]*df+w[2]*dl
    W=snf([dy,dl,df]); Dsnf=1-W/W.max()
    om=[max(ec(Dmax,y,k,s),ec(Dsum,y,k,s)) for s in SEEDS]
    tm=[ec(Dsnf,y,k,s) for s in SEEDS]
    sc=nmi(y,SpectralClustering(k,affinity='precomputed',assign_labels='discretize',random_state=0).fit_predict(W),average_method='max')
    O.append(np.mean(om)); T.append(np.mean(tm))
    print(f"{n:9s}{SP[n]:6.1f}{np.mean(om):8.3f}±{np.std(om):.3f}   {np.mean(tm):8.3f}±{np.std(tm):.3f}   {sc:7.3f}",flush=True)
O=np.array(O);T=np.array(T)
W,p=wilcoxon(O,T,alternative='greater')
print(f"\nMATCHED (t-SNE+KMeans, mean of 5 seeds): ours vs SNF  wins={int((O>T).sum())}/{len(O)}  meanΔ={(O-T).mean():+.3f}  W+={W:.0f} p={p:.4g}")
