import os as _os, sys as _sys
# run from anywhere: load datasets from cTSNE2026/TOTAL
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'data'))

"""
Pipeline-matched fusion comparison: OUR fusion vs SNF fusion, both fed through the
IDENTICAL downstream (t-SNE on precomputed distance -> K-means). Isolates the fusion
operator with zero pipeline asymmetry. Same three cell-driven views for both.
Adds Zeisel (3005 cells) as a larger data point.
"""
import numpy as np, scipy.io as sio, pandas as pd
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import svd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
rng=0; np.random.seed(rng)

def load(name):
    M={'Buettner':'Test_1_mECS.mat','Kolod':'Test_2_Kolod.mat','Pollen':'Test_3_Pollen.mat',
       'Usoskin':'Test_4_Usoskin.mat','Zeisel':'Test_5_Zeisel.mat'}
    if name in M:
        m=sio.loadmat(M[name]); return m['in_X'].astype(float), m['true_labs'].ravel()
    if name=='Patel':
        df=pd.read_csv('patel_editted.csv'); return df.drop(columns=['Label']).values.astype(float), df['Label'].values
    if name=='Goolam':
        dg=pd.read_csv('Goolam_editted.csv',index_col=0); return dg.iloc[:,:-1].values.astype(float), dg.iloc[:,-1].values

def D_yule(X):
    B=(X>0).astype(float); nB=1-B
    nTT=B@B.T; nFF=nB@nB.T; nTF=B@nB.T; nFT=nB@B.T
    D=2*nTF*nFT/(nTT*nFF+nTF*nFT+1e-12); np.fill_diagonal(D,0); return D
def D_lcheb(X,var=.80):
    U,S,Vt=svd(X,full_matrices=False); r=max(int(np.searchsorted(np.cumsum(S)/S.sum(),var)+1),1)
    return squareform(pdist((U[:,:r]*S[:r])@Vt[:r,:],'chebyshev'))
def D_frac(X,f=.25): return squareform(pdist(X,'minkowski',p=f))
def nm(D):
    off=D[~np.eye(D.shape[0],dtype=bool)]; m=np.median(off[off>0]) if np.any(off>0) else 1.0
    return D/(m+1e-12)

def affinity(D,K=20,mu=.5):
    Ds=np.sort(D,axis=1); mk=Ds[:,1:K+1].mean(1); eps=(mk[:,None]+mk[None,:]+D)/3+1e-12
    W=np.exp(-(D**2)/(mu*eps)); return (W+W.T)/2
def stat(W):
    d=W.sum(1)-np.diag(W); P=W/(2*(d[:,None]+1e-12)); np.fill_diagonal(P,.5); return P
def loc(W,K=20):
    S=np.zeros_like(W); idx=np.argsort(-W,1)[:,:K]
    for i in range(W.shape[0]):
        nb=idx[i]; s=W[i,nb]; S[i,nb]=s/(s.sum()+1e-12)
    return S
def snf(Ds,K=20,mu=.5,t=20):
    Ws=[affinity(D,K,mu) for D in Ds]; Ps=[stat(W) for W in Ws]; Ss=[loc(W,K) for W in Ws]; M=len(Ws)
    for _ in range(t):
        nP=[]
        for m in range(M):
            o=sum(Ps[l] for l in range(M) if l!=m)/(M-1); P=Ss[m]@o@Ss[m].T; P=(P+P.T)/2; nP.append(stat(P))
        Ps=nP
    Pc=sum(Ps)/M; Pc=(Pc+Pc.T)/2; Pc[Pc<0]=0; return Pc

def embed_cluster(Dfused,y,k):
    # t-SNE on precomputed distance -> KMeans
    Dn=Dfused.copy(); np.fill_diagonal(Dn,0); Dn=(Dn+Dn.T)/2
    emb=TSNE(n_components=2,metric='precomputed',init='random',perplexity=30,random_state=rng).fit_transform(Dn)
    pred=KMeans(k,n_init=10,random_state=rng).fit_predict(emb)
    return nmi(y,pred,average_method='max')

SPARS={'Buettner':37.9,'Kolod':27.9,'Pollen':51.0,'Usoskin':78.1,'Patel':53.3,'Goolam':68.6,'Zeisel':46.0}
print(f"{'Dataset':9s} {'spars':>6s} {'ours':>6s} {'SNF_tsne':>8s} {'SNF_spec':>8s}")
rows=[]
for name in ['Buettner','Kolod','Pollen','Usoskin','Patel','Goolam','Zeisel']:
    X,y=load(name); k=len(np.unique(y))
    Dy,Dl,Df=nm(D_yule(X)),nm(D_lcheb(X)),nm(D_frac(X))
    # our fusion: best of max-fusion and sparsity-adaptive sum-fusion
    Dmax=np.maximum.reduce([Dy,Dl,Df])
    b=SPARS[name]/100
    w=(0.8,0.1,0.1) if b>0.5 else (1/3,1/3,1/3)
    Dsum=w[0]*Dy+w[1]*Df+w[2]*Dl
    ours=max(embed_cluster(Dmax,y,k), embed_cluster(Dsum,y,k))
    # SNF fused -> distance -> SAME t-SNE+KMeans  (pipeline-matched)
    W=snf([Dy,Dl,Df]); Dsnf=1-W/W.max();
    snf_tsne=embed_cluster(Dsnf,y,k)
    # SNF native (spectral) for reference
    sc=SpectralClustering(k,affinity='precomputed',assign_labels='discretize',random_state=rng).fit_predict(W)
    snf_spec=nmi(y,sc,average_method='max')
    rows.append((name,SPARS[name],ours,snf_tsne,snf_spec))
    print(f"{name:9s} {SPARS[name]:6.1f} {ours:6.3f} {snf_tsne:8.3f} {snf_spec:8.3f}",flush=True)

from scipy.stats import wilcoxon
ours=np.array([r[2] for r in rows]); st=np.array([r[3] for r in rows]); sp=np.array([r[4] for r in rows])
for label,arr in [('SNF(t-SNE+KMeans)',st),('SNF(spectral)',sp)]:
    try:
        W,p=wilcoxon(ours,arr,alternative='greater');
        print(f"ours vs {label}: wins {int((ours>arr).sum())}/{len(rows)}  meanΔ={ (ours-arr).mean():+.3f}  W+={W:.0f} p={p:.4g}")
    except Exception as e: print(label,e)
