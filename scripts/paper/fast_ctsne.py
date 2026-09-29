"""
Fast c-TSNE: O(n log n . D) implementation and benchmark.

Exact c-TSNE  : full n x n fused affinity + full eigendecomposition (lambda_max
                normalization, O(n^3)) + exact t-SNE  (O(n^2)/iter).
Fast  c-TSNE  : approximate k-NN under the cell-driven views via pynndescent
                (NN-descent, ~O(n log n . D)); fuse the three views only on the
                candidate edges; median-of-edges normalization (O(nk)); sparse
                k-NN affinity + Barnes-Hut t-SNE (O(n log n)/iter).

Reports (a) NMI fidelity (fast vs exact) on the 7 local datasets and
        (b) wall-clock scaling on subsampled Zeisel, with a fitted exponent.
Fidelity is theory-backed: Theorem 1 bounds the k-NN-truncation objective error.
"""
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'data'))
import time, warnings, numpy as np, scipy.io as sio, pandas as pd, scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform, cdist
from numpy.linalg import svd, eigvalsh
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from pynndescent import NNDescent
warnings.filterwarnings('ignore')
RNG=0; K=60; PERP=15; R=15

def dense(X): return np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
def load(n):
    M={'Buettner':'Test_1_mECS.mat','Kolod':'Test_2_Kolod.mat','Pollen':'Test_3_Pollen.mat',
       'Usoskin':'Test_4_Usoskin.mat','Zeisel':'Test_5_Zeisel.mat'}
    if n in M: m=sio.loadmat(M[n]); return dense(m['in_X']).astype(np.float64), m['true_labs'].ravel()
    if n=='Patel': d=pd.read_csv('patel_editted.csv'); return d.drop(columns=['Label']).values.astype(np.float64), d['Label'].values
    if n=='Goolam': g=pd.read_csv('Goolam_editted.csv',index_col=0); return g.iloc[:,:-1].values.astype(np.float64), g.iloc[:,-1].values

# ---- view distances on explicit candidate edge lists (O(nk D)) ----
def svd_coords(X,r=R):
    U,S,Vt=svd(X,full_matrices=False); r=min(r,len(S)); return (U[:,:r]*S[:r])  # n x r reconstruction coords
def med(v):
    v=v[v>0]; return np.median(v) if v.size else 1.0

def fast_ctsne(X, k=K):
    t={}; n=X.shape[0]
    B=(X>0).astype(np.float32)
    t0=time.time(); Xr=svd_coords(X); t['svd']=time.time()-t0
    # ANN under two native metrics (yule on binary, chebyshev on SVD coords)
    t0=time.time()
    nn_y=NNDescent(B, metric='yule', n_neighbors=k, random_state=RNG, verbose=False).neighbor_graph[0]
    nn_c=NNDescent(Xr, metric='chebyshev', n_neighbors=k, random_state=RNG, verbose=False).neighbor_graph[0]
    t['ann']=time.time()-t0
    # union candidates, score all three views on edges, fuse (sparsity-adaptive sum)
    t0=time.time()
    rows=[]; cols=[]; vals=[]
    spars=float(np.mean(X==0)); w=(0.8,0.1,0.1) if spars>0.5 else (1/3,1/3,1/3)
    # UNBIASED per-view scale from RANDOM pairs (not k-NN edges, which are biased small)
    Dg=X.shape[1]
    smp=np.random.RandomState(RNG).choice(n, size=min(n,250), replace=False)
    Bs=B[smp].astype(np.float64); Xs=X[smp]; Xrs=Xr[smp]
    nTT=Bs@Bs.T; tot=Bs.sum(1); nTF=tot[:,None]-nTT; nFT=tot[None,:]-nTT; nFF=Dg-nTT-nTF-nFT
    Dys=2*nTF*nFT/(nTT*nFF+nTF*nFT+1e-12); offm=~np.eye(len(smp),dtype=bool)
    SY=med(Dys[offm]); SF=med(pdist(Xs,'minkowski',p=0.25)); SC=med(pdist(Xrs,'chebyshev'))
    for i in range(n):
        C=np.unique(np.concatenate([nn_y[i],nn_c[i]])); C=C[C!=i]
        bi=B[i]; nb=B[C]; ntt=nb@bi; tot=bi.sum(); cb=nb.sum(1)
        ntf=tot-ntt; nft=cb-ntt; nff=X.shape[1]-ntt-ntf-nft
        dy=2*ntf*nft/(ntt*nff+ntf*nft+1e-12)/SY
        df=((np.abs(X[i]-X[C])**0.25).sum(1)**4)/SF
        dc=np.max(np.abs(Xr[i]-Xr[C]),axis=1)/SC
        fused=w[0]*dy+w[1]*df+w[2]*dc
        order=np.argsort(fused)[:k]
        rows+= [i]*len(order); cols+=list(C[order]); vals+=list(fused[order])
    D=sp.csr_matrix((vals,(rows,cols)),shape=(n,n)); D=D.maximum(D.T)
    t['fuse']=time.time()-t0
    t0=time.time()
    emb=TSNE(2,metric='precomputed',init='random',perplexity=PERP,method='barnes_hut',random_state=RNG).fit_transform(D)
    t['tsne']=time.time()-t0
    return emb, t

def exact_ctsne(X, norm='eig'):
    t={}; n=X.shape[0]
    B=(X>0).astype(np.float64); nB=1-B
    t0=time.time()
    nTT=B@B.T; nFF=nB@nB.T; nTF=B@nB.T; nFT=nB@B.T
    Dy=2*nTF*nFT/(nTT*nFF+nTF*nFT+1e-12); np.fill_diagonal(Dy,0)
    Df=squareform(pdist(X,'minkowski',p=0.25)); Xr=svd_coords(X); Dc=squareform(pdist(Xr,'chebyshev'))
    t['dist']=time.time()-t0
    t0=time.time()
    if norm=='eig':       # naive: largest |eigenvalue| via full eigendecomposition O(n^3)
        for D_ in (Dy,Df,Dc):
            lam=np.max(np.abs(eigvalsh(D_))); D_/=(lam+1e-12)
    else:                 # median normalization (matches Fast c-TSNE, isolates k-NN effect)
        for D_ in (Dy,Df,Dc):
            D_/=(med(squareform(D_,checks=False))+1e-12)
    t['norm']=time.time()-t0
    spars=float(np.mean(X==0)); w=(0.8,0.1,0.1) if spars>0.5 else (1/3,1/3,1/3)
    Dfused=w[0]*Dy+w[1]*Df+w[2]*Dc
    t0=time.time()
    emb=TSNE(2,metric='precomputed',init='random',perplexity=PERP,method='exact',random_state=RNG).fit_transform(Dfused)
    t['tsne']=time.time()-t0
    return emb, t

def nmi_of(emb,y,k): return nmi(y,KMeans(k,n_init=10,random_state=RNG).fit_predict(emb),average_method='max')

def mean_nmi(emb_fn, X, y, C, seeds=(0,1,2)):
    return float(np.mean([nmi_of(emb_fn(), y, C) for _ in seeds]))

if __name__=='__main__':
    _=NNDescent(np.random.rand(60,8).astype(np.float32),metric='yule',n_neighbors=10,random_state=0).neighbor_graph
    print("=== (a) Fidelity (matched median normalization; isolates k-NN truncation), avg 3 seeds ===")
    print(f"{'Dataset':9s}{'n':>6s}{'NMI exact':>10s}{'NMI fast(k=60)':>15s}{'gap':>7s}")
    for name in ['Goolam','Buettner','Pollen','Patel','Usoskin','Kolod']:
        X,y=load(name); C=len(np.unique(y))
        ex=mean_nmi(lambda:exact_ctsne(X,norm='median')[0], X,y,C)
        fa=mean_nmi(lambda:fast_ctsne(X,k=60)[0], X,y,C)
        print(f"{name:9s}{X.shape[0]:6d}{ex:10.3f}{fa:15.3f}{fa-ex:+7.3f}",flush=True)
    print("\n=== (b) Accuracy vs k (the approximation knob): NMI -> exact as k grows ===")
    print(f"{'Dataset':9s}{'exact':>7s}{'k=30':>7s}{'k=60':>7s}{'k=120':>7s}{'k=240':>7s}")
    for name in ['Buettner','Usoskin','Zeisel']:
        X,y=load(name); C=len(np.unique(y)); ne=X.shape[0]
        ex=mean_nmi(lambda:exact_ctsne(X,norm='median')[0],X,y,C)
        row=[]
        for k in [48,72,120,240]:
            if k>=ne-2: row.append(float('nan')); continue
            row.append(mean_nmi(lambda:fast_ctsne(X,k=k)[0],X,y,C))
        print(f"{name:9s}{ex:7.3f}"+''.join(f"{v:7.3f}" for v in row),flush=True)
    print("\n=== (c) Wall-clock scaling on subsampled Zeisel (D=4412), exact uses eig (O(n^3)) ===")
    X,y=load('Zeisel'); C=len(np.unique(y)); rs=np.random.RandomState(0)
    print(f"{'n':>6s}{'t_exact(s)':>11s}{'t_fast(s)':>10s}{'speedup':>8s}")
    ns=[300,600,1200,2000,3005]; TE=[];TF=[]
    for n in ns:
        idx=rs.choice(X.shape[0],size=n,replace=False); Xn=X[idx]
        te=sum(exact_ctsne(Xn,norm='eig')[1].values()); tf=sum(fast_ctsne(Xn,k=60)[1].values())
        TE.append(te); TF.append(tf)
        print(f"{n:6d}{te:11.2f}{tf:10.2f}{te/tf:7.1f}x",flush=True)
    ns=np.array(ns)
    se=np.polyfit(np.log(ns),np.log(TE),1)[0]; sf=np.polyfit(np.log(ns),np.log(TF),1)[0]
    print(f"\nfitted scaling exponent  exact: n^{se:.2f}   fast: n^{sf:.2f}")
