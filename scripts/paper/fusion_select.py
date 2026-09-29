import os as _os, sys as _sys
# run from anywhere: load datasets from cTSNE2026/TOTAL
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'data'))

# Generalized power-mean fusion with JOINT (p, weights) selection by a label-free
# internal criterion. Compare selectors (silhouette vs eigengap) against fixed
# sum/max and the oracle. Goal: a selector that reliably beats fixed sum on NMI.
import numpy as np, scipy.io as sio, pandas as pd, scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import svd, eigh
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
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
def powmean(V,w,p):
    if p>=64: return np.maximum.reduce(V)
    return np.power(sum(wm*np.power(Vi,p) for wm,Vi in zip(w,V)),1.0/p)
def affinity(D,K=20,mu=.5):
    Ds=np.sort(D,1); mk=Ds[:,1:K+1].mean(1); e=(mk[:,None]+mk[None,:]+D)/3+1e-12
    W=np.exp(-(D**2)/(mu*e)); np.fill_diagonal(W,0); return (W+W.T)/2
def gap(W,C):
    d=W.sum(1); di=1/np.sqrt(d+1e-12); L=np.eye(W.shape[0])-(di[:,None]*W*di[None,:])
    ev=np.sort(eigh(L)[0]); return ev[C]-ev[C-1]
def spec(W,C):
    return SpectralClustering(C,affinity='precomputed',assign_labels='discretize',random_state=0).fit_predict(W)
PS=[1,2,4,8,64]; WS=[(1/3,1/3,1/3),(0.6,0.2,0.2),(0.8,0.1,0.1),(0.9,0.05,0.05)]
SP={'Buettner':37.9,'Kolod':27.9,'Pollen':51.0,'Usoskin':78.1,'Patel':53.3,'Goolam':68.6,'Zeisel':46.0}
hdr=f"{'Dataset':9s}{'sum':>7s}{'max':>7s}{'sil-sel':>8s}{'gap-sel':>8s}{'oracle':>7s}"
print(hdr)
res={'sum':[],'max':[],'sil':[],'gap':[],'orc':[]}
for n in ['Buettner','Kolod','Pollen','Usoskin','Patel','Goolam','Zeisel']:
    X,y=load(n); C=len(np.unique(y)); V=[nm(Dy(X)),nm(Df(X)),nm(Dl(X))]
    grid=[]
    for p in PS:
        for w in WS:
            D=powmean(V,w,p); W=affinity(D); lab=spec(W,C)
            try: sil=silhouette_score(D,lab,metric='precomputed') if len(set(lab))>1 else -1
            except Exception: sil=-1
            grid.append(dict(p=p,w=w,nmi=nmi(y,lab,average_method='max'),sil=sil,gap=gap(W,C)))
    sumv=[g for g in grid if g['p']==1 and g['w']==WS[0]][0]['nmi']
    # 'sum' baseline = equal-weight p=1 ; 'max' = p=64 best weight
    maxv=max(g['nmi'] for g in grid if g['p']==64)
    sil_sel=max(grid,key=lambda g:g['sil'])['nmi']
    gap_sel=max(grid,key=lambda g:g['gap'])['nmi']
    orc=max(g['nmi'] for g in grid)
    for k,v in zip(['sum','max','sil','gap','orc'],[sumv,maxv,sil_sel,gap_sel,orc]): res[k].append(v)
    print(f"{n:9s}{sumv:7.3f}{maxv:7.3f}{sil_sel:8.3f}{gap_sel:8.3f}{orc:7.3f}",flush=True)
import numpy as np
m={k:np.mean(v) for k,v in res.items()}
print(f"\nMEAN  sum={m['sum']:.3f}  max={m['max']:.3f}  sil-sel={m['sil']:.3f}  gap-sel={m['gap']:.3f}  oracle={m['orc']:.3f}")
sil=np.array(res['sil']); sm=np.array(res['sum'])
from scipy.stats import wilcoxon
try:
    W,p=wilcoxon(sil,sm,alternative='greater'); print(f"silhouette-selected vs fixed-sum: wins {int((sil>sm).sum())}/7, meanD={ (sil-sm).mean():+.3f}, Wilcoxon p={p:.4g}")
except Exception as e: print(e)
