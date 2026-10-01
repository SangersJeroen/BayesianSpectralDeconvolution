import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
    "axes.labelsize":9,"legend.frameon":False,"pdf.fonttype":42,"figure.dpi":150})
C={"truth":"#d1495b","kern":"#2e4057","data":"#7a7a7a","fit":"#00798c","acc":"#edae49"}
rng=np.random.default_rng(3)
x=np.arange(0,12,0.1); G=0.5; SIG=0.01
def kern(x,mu): return 1/(1+((x-mu)/G)**2)          # Lorentzian, peak-normalised
def model(x,I,mu): return sum(i*kern(x,m) for i,m in zip(I,mu))
I0=np.array([0.5,1.0,1.0]); MU0=np.array([3.,6.,10.])
clean=model(x,I0,MU0); y=clean+rng.normal(0,SIG,x.size)

# 1 problem
fig,ax=plt.subplots(figsize=(6.2,2.6))
ax.vlines(MU0,0,I0,color=C["truth"],lw=2.5,label="true sources")
ax.plot(MU0,I0,"o",color=C["truth"])
ax.plot(x,kern(x,2)*0.6+0,color=C["kern"],lw=1.2,ls="--",label="blur kernel (width known)")
ax.plot(x,y,".",ms=3.5,color=C["data"],label="noisy data $y_i$")
ax.plot(x,clean,color=C["fit"],lw=1,label="noiseless response")
ax.set_xlabel("$x$ (e.g. energy)");ax.set_ylabel("intensity");ax.legend(loc="upper left",fontsize=8,ncol=2)
ax.set_ylim(0,1.55); fig.tight_layout(); fig.savefig("figs/fig1_problem.pdf")

# LS fits for K=1..5
def fit(K,ntry=60):
    best=None
    for _ in range(ntry):
        p0=np.r_[rng.uniform(.2,1.2,K),rng.uniform(1,11,K)]
        r=least_squares(lambda p:(model(x,p[:K],p[K:])-y)/SIG,p0,
             bounds=(np.r_[np.full(K,.2),np.full(K,1)],np.r_[np.full(K,1.2),np.full(K,11)]))
        if best is None or r.cost<best.cost: best=r
    return best
fits={K:fit(K) for K in range(1,6)}
chi2={K:2*f.cost for K,f in fits.items()}; print(chi2)

# 2 fit vs K
fig,axs=plt.subplots(1,3,figsize=(6.4,1.9),sharey=True)
for a,K in zip(axs,[1,2,3]):
    p=fits[K].x; a.plot(x,y,".",ms=2.5,color=C["data"]); a.plot(x,model(x,p[:K],p[K:]),color=C["fit"],lw=1.4)
    a.set_title(f"$K={K}$,  $\\chi^2={chi2[K]:.0f}$",fontsize=9); a.set_xlabel("$x$")
axs[0].set_ylabel("intensity"); fig.tight_layout(); fig.savefig("figs/fig2_kfits.pdf")

# 3 HMC toy
A=np.array([[1,.9],[.9,1]]); Ai=np.linalg.inv(A)
U=lambda q:0.5*q@Ai@q; gU=lambda q:Ai@q
def leap(q,p,eps,n):
    qs=[q.copy()];Hs=[U(q)+.5*p@p]; 
    p=p-.5*eps*gU(q)
    for i in range(n):
        q=q+eps*p; 
        p=p-(eps if i<n-1 else .5*eps)*gU(q)
        qs.append(q.copy());Hs.append(U(q)+.5*p@p)
    return np.array(qs),np.array(Hs)
def euler(q,p,eps,n):
    qs=[q.copy()];Hs=[U(q)+.5*p@p]
    for i in range(n):
        q,p=q+eps*p,p-eps*gU(q); qs.append(q.copy());Hs.append(U(q)+.5*p@p)
    return np.array(qs),np.array(Hs)
q0=np.array([-1.5,1.7]);p0=np.array([1.0,.6])
ql,_=leap(q0,p0,.12,20); _,Hl=leap(q0,p0,.12,60); _,He=euler(q0,p0,.12,60)
fig,(a,b)=plt.subplots(1,2,figsize=(6.4,2.5),gridspec_kw={"width_ratios":[1,1.1]})
g=np.linspace(-3,3,200);X,Y=np.meshgrid(g,g)
Z=0.5*(Ai[0,0]*X**2+2*Ai[0,1]*X*Y+Ai[1,1]*Y**2)
a.contour(X,Y,Z,levels=[.5,2,4.5,8],colors="#bbbbbb",linewidths=.8)
a.plot(*ql.T,"-o",ms=2.5,color=C["fit"],lw=1); a.plot(*q0,"s",color=C["truth"],ms=5)
a.set_xlabel("$\\theta_1$");a.set_ylabel("$\\theta_2$");a.set_title("leapfrog trajectory",fontsize=9)
b.semilogy(np.abs(He-He[0])+1e-4,color=C["truth"],label="naive Euler");b.semilogy(np.abs(Hl-Hl[0])+1e-4,color=C["fit"],label="leapfrog")
b.set_xlabel("step");b.set_ylabel("$|H-H_0|$");b.legend(fontsize=8);b.set_title("energy error",fontsize=9)
fig.tight_layout();fig.savefig("figs/fig3_hmc.pdf")

# 4 tempering: profile likelihood of one peak's centre
mus=np.linspace(1,11,1000)
def prof(mu):
    k=kern(x,mu); Ihat=np.clip(k@y/(k@k),.2,1.2); return -0.5*np.sum((y-Ihat*k)**2)/SIG**2
lL=np.array([prof(m) for m in mus]); lL-=lL.max(); print(lL.min())
fig,(a,b)=plt.subplots(1,2,figsize=(6.4,2.4))
a.plot(mus,lL/1e3,color=C["kern"]);a.set_xlabel("centre $\\mu$ of one peak");a.set_ylabel("$\\log L\\ (\\times10^3)$")
a.set_title("likelihood: many local optima",fontsize=9)
for beta,c,lab in [(1,C["truth"],"$\\beta=1$"),(1e-3,C["acc"],"$\\beta=10^{-3}$"),(1e-4,C["fit"],"$\\beta=10^{-4}$")]:
    d=np.exp(beta*lL); b.plot(mus,d,color=c,label=lab)
b.set_xlabel("$\\mu$");b.set_ylabel("$p_\\beta(\\mu\\,|\\,y)$ (peak = 1)");b.set_yscale("linear");b.legend(fontsize=8);b.set_title("tempered posteriors",fontsize=9)
fig.tight_layout();fig.savefig("figs/fig4_tempering.pdf")

# 5 schematic stochastic complexity
K=np.arange(1,6); fitterm=np.array([100,40,5,4.2,4.0]); occam=np.array([2,5,8,12,16.])
F=fitterm+occam
fig,a=plt.subplots(figsize=(3.6,2.5))
a.plot(K,fitterm,"o--",color=C["kern"],label="misfit  $-\\langle\\log L\\rangle$")
a.plot(K,occam,"s--",color=C["acc"],label="Occam penalty")
a.plot(K,F,"o-",color=C["truth"],lw=2,label="$F(K)=-\\log Z_K$")
a.set_xticks(K);a.set_xlabel("number of peaks $K$");a.set_yticks([]);a.set_ylabel("(schematic)")
a.legend(fontsize=7.5,loc="upper right");fig.tight_layout();fig.savefig("figs/fig5_evidence.pdf")

# 6 result
p=fits[3].x; order=np.argsort(p[3:]); I=p[:3][order]; mu=p[3:][order]
fig,(a,b)=plt.subplots(2,1,figsize=(6.2,3.2),sharex=True,gridspec_kw={"height_ratios":[3,1]})
a.plot(x,y,".",ms=3.5,color=C["data"],label="data")
for i,m in zip(I,mu): a.fill_between(x,0,i*kern(x,m),alpha=.25,color=C["acc"])
a.plot(x,model(x,I,mu),color=C["fit"],lw=1.6,label="$K=3$ fit"); a.legend(fontsize=8);a.set_ylabel("intensity")
b.plot(x,(y-model(x,I,mu))/SIG,".",ms=3,color=C["data"]);b.axhline(0,color="k",lw=.5);b.set_ylabel("resid. /$\\sigma$");b.set_xlabel("$x$")
fig.tight_layout();fig.savefig("figs/fig6_result.pdf")
