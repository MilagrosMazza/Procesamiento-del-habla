import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.linalg import sqrtm
from matplotlib import colors as mcolors
from scipy.special import logsumexp

# Specification of some HMMs for classification

class hmm:
    
    def __init__(self,means,vrs,trans):
        self.means = means
        self.vrs = vrs
        self.trans = trans
        
        self.dim = means.shape[1]
        self.numStates = means.shape[0] + 2
        self.devs = np.array([sqrtm(M) for M in vrs])

    def logprobstate( self, x ):
        N = self.numStates - 2
        T = len(x)
        log_b = -np.ones((N,T))*np.log(2*np.pi)
        for k in range(N):
            log_b[k,:] -= 0.5 * np.log(np.linalg.det(self.vrs[k]))
            x_unbiased = x - self.means[k,np.newaxis]
            inv_sig =  np.linalg.inv(self.vrs[k])
            log_b[k,:] -= 0.5 * np.einsum('ij,ij->i',x_unbiased @ inv_sig, x_unbiased).T

        return log_b
 
    
def genhmm(hmm):

    stateSeq = [0]
    while stateSeq[-1] != hmm.numStates-1:
        stateSeq.append(int(np.random.choice(hmm.numStates, 1, p=hmm.trans[stateSeq[-1],:])))

    stateSeq = np.array(stateSeq)
    x = np.add.reduce(np.matmul(np.random.randn(stateSeq.size-2, hmm.dim)[:,np.newaxis,:],hmm.devs[(stateSeq[1:-1]-1)]),axis=1) + hmm.means[(stateSeq[1:-1]-1)]
    
    return x, stateSeq


def GenData():
    
    # GENDATA Generation of simulation data for HMM lab.
    # Specification of simulated vowels statistics.
    Pa = 0.25
    mu_a = np.array([730., 1090.])
    std_a = np.array([[35.,  20.],
                      [20.,  230.]])
    var_a = std_a.dot(std_a)

    Pe = 0.3
    mu_e = np.array([530., 1840.])
    std_e = np.array([[120.,  25.],
                      [25.,  190.]])
    var_e = std_e.dot(std_e)

    Pi = 0.25
    mu_i = np.array([270., 2290.])
    std_i = np.array([[50.,  5.],
                      [5.,  190.]])
    var_i = std_i.dot(std_i)

    Po = 0.15
    mu_o = np.array([570., 840.])
    std_o = np.array([[40.,  20.],
                      [20.,  140.]])
    var_o = std_o.dot(std_o)

    Py = 0.05
    mu_y = np.array([440., 1020.])
    std_y = np.array([[80.,  40.],
                      [40.,  130.]])
    var_y = std_y.dot(std_y)


    # 1: ergodic /aiy/, "unstable"
    means = np.stack((mu_a,mu_i,mu_y))
    vrs = np.stack((var_a,var_i,var_y))
    trans = np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.4, 0.3, 0.3, 0.0],
                      [0.0, 0.3, 0.4, 0.3, 0.0],
                      [0.0, 0.3, 0.3, 0.3, 0.1],
                      [0.0, 0.0, 0.0, 0.0, 1.0]])

    hmm1 = hmm(means, vrs, trans)

    # 2: ergodic /aiy/, "stable"
    means = np.stack((mu_a,mu_i,mu_y))
    vrs = np.stack((var_a,var_i,var_y))
    trans = np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.95, 0.025, 0.025, 0.0],
                      [0.0, 0.025, 0.95, 0.025, 0.0],
                      [0.0, 0.02, 0.02, 0.95, 0.01],
                      [0.0, 0.0, 0.0, 0.0, 1.0]])

    hmm2 = hmm(means, vrs, trans)

    # 3: left-right /aiy/, unstable
    means = np.stack((mu_a,mu_i,mu_y))
    vrs = np.stack((var_a,var_i,var_y))
    trans = np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.5, 0.5, 0.0, 0.0],
                      [0.0, 0.0, 0.5, 0.5, 0.0],
                      [0.0, 0.0, 0.0, 0.5, 0.5],
                      [0.0, 0.0, 0.0, 0.0, 1.0]])

    hmm3 = hmm(means, vrs, trans)

    # 4: left-right /aiy/, stable
    means = np.stack((mu_a,mu_i,mu_y))
    vrs = np.stack((var_a,var_i,var_y))
    trans = np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.95, 0.05, 0.0, 0.0],
                      [0.0, 0.0, 0.95, 0.05, 0.0],
                      [0.0, 0.0, 0.0, 0.95, 0.05],
                      [0.0, 0.0, 0.0, 0.0, 1.0]])

    hmm4 = hmm(means, vrs, trans)

    # 5: left-right /yia/, stable
    means = np.stack((mu_y,mu_i,mu_a))
    vrs  = np.stack((var_y,var_i,var_a))
    trans = np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.95, 0.05, 0.0, 0.0],
                      [0.0, 0.0, 0.95, 0.05, 0.0],
                      [0.0, 0.0, 0.0, 0.95, 0.05],
                      [0.0, 0.0, 0.0, 0.0, 1.0]])

    hmm5 = hmm(means, vrs, trans)

    # 6: left-right /aie/, stable
    means = np.stack((mu_a,mu_i,mu_e))
    vrs  = np.stack((var_a,var_i,var_e))
    trans = np.array([[0.0, 1.0,  0.0,  0.0,  0.0],
                      [0.0, 0.95, 0.05, 0.0,  0.0],
                      [0.0, 0.0,  0.95, 0.05, 0.0],
                      [0.0, 0.0,  0.0,  0.95, 0.05],
                      [0.0, 0.0,  0.0,  0.0,  1.0]]);

    hmm6 = hmm(means, vrs, trans)

    # 7: left-right /aiy/, stable w/ unequal stay probas
    means = np.stack((mu_a,mu_i,mu_y))
    vrs  = np.stack((var_a,var_i,var_y))
    trans = np.array([[0.0, 1.0,  0.0,  0.0,  0.0],
                      [0.0, 0.7,  0.3,  0.0,  0.0],
                      [0.0, 0.0,  0.95, 0.05, 0.0],
                      [0.0, 0.0,  0.0,  0.7,  0.3],
                      [0.0, 0.0,  0.0,  0.0,  1.0]])

    hmm7 = hmm(means, vrs, trans)
    
    return hmm1, hmm2, hmm3, hmm4, hmm5, hmm6, hmm7


def plotseq(hmm, stateSeq, x):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    classes_names = np.array(['Estado '+str(i+1) for i in range(hmm.numStates-2)],dtype=object)
    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).values())

    ax1.plot(x[:,0])
    ax2.plot(x[:,1])

    s = stateSeq[1:-1]-1
    for i in range(hmm.numStates-2):
        mask = s==i
        ax1.scatter(np.arange(s.size)[mask],x[mask,0],color=colors[i],label=classes_names[i])
        ax2.scatter(np.arange(s.size)[mask],x[mask,1],color=colors[i],label=classes_names[i])

    ax1.legend()
    
    return ax1, ax2
    
    
def plotseq2(hmm,stateSeq,x,gauss=False):
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    classes_names = np.array(['Estado '+str(i+1) for i in range(hmm.numStates-2)],dtype=object)
    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).values())

    ax.plot(x[:,0],x[:,1])
    
    for i in range(x.shape[0]):
        ax.annotate(str(i+1),(x[i,0],x[i,1]))
    

    s = stateSeq[1:-1]-1
    for i in range(hmm.numStates-2):
        mask = s==i
        ax.scatter(x[mask,0],x[mask,1],color=colors[i],label=classes_names[i])
        if gauss:
            ellip = covariance_ellipse(hmm.means[i],hmm.vrs[i],ax=ax,color=colors[i])
            ax.add_artist(ellip)
        
    ax.legend()
    
    return ax
    
    
def covariance_ellipse(mu, sigma, ax=None, color="k"):
    
    # Cálculo de los autovalores:
    vals, vecs = np.linalg.eigh(sigma)
    
    # Inclinación de la elipse:
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Semiejes de la elipse:
    w, h = 2 * np.sqrt(vals)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
                
    
    ax.tick_params(axis='both', which='major', labelsize=20)
    ellipse = Ellipse(xy=mu, width=w, height=h, angle=theta)
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.2)
    ellipse.set_edgecolor(color)
    ellipse.set_facecolor(color)
#     ax.add_artist(ellipse)
    
    return ellipse


# Cálculo del b_t(j)

def logpdfnorm(mu,sigma,x):
    """
    
        Evaluación de los puntos x en K gaussianas de medias mu y varianzas sigma.
        
        mu: (K,d) array que contiene las medias de las gaussianas en las filas.
        sigma: (K,d,d) array que contiene las matrices de covarianzas de las gaussianas.
        x: (N,d) array que contiene los lugares en donde evaluar las gaussianas.
        
        return: (N,K) array con los valores de la normal en los puntos x
    """
        
    d = x.shape[1]
    x_unbiased = x - mu[:,np.newaxis]
    sigma_inv = np.linalg.inv(sigma)
    y = - (d/2) * np.log(2*np.pi) - .5 * np.log(np.linalg.det(sigma)) - .5 * np.einsum('ijk,ijk->ij',np.matmul(x_unbiased,sigma_inv),x_unbiased).T

    return y


# Recursiones:

def logdot(log_a, log_b):
    """
        Función para calcular el producto de matrices expresadas en logaritmos
        https://stackoverflow.com/questions/23630277/numerically-stable-way-to-multiply-log-probability-matrices-in-numpy
    """
    max_log_a, max_log_b = np.max(log_a), np.max(log_b)
    a, b = log_a - max_log_a, log_b - max_log_b
    np.exp(a, out=a)
    np.exp(b, out=b)
    log_c = np.log(np.dot(a, b))
    log_c += max_log_a + max_log_b
    return log_c



def logfw(X,hmm):
    """
        Función para implementar la recursión alpha.
    """
    
    T = X.shape[0] # Cantidad de mediciones
    N = hmm.numStates - 2 # Cantidad de estados sin contar el final y el inicial
#     zero = 1e-100 # Valor mínimo para evitar división por cero

    tmp = np.copy(hmm.trans)
    tmp[tmp<1e-320] = 1e-320
    logA = np.log(tmp)

    # Evaluación del logaritmo de la normal de cada estado y cada x_t
#     log_b = logpdfnorm(hmm.means,hmm.vrs,X)
    
    log_b = -np.ones((N,T))*np.log(2*np.pi)
    for k in range(N):
        log_b[k,:] -= 0.5 * np.log(np.linalg.det(hmm.vrs[k]))
        x_unbiased = X - hmm.means[k,np.newaxis]
        inv_sig =  np.linalg.inv(hmm.vrs[k])
        log_b[k,:] -= 0.5 * np.einsum('ij,ij->i',x_unbiased @ inv_sig, x_unbiased).T
            
    # Matriz de transición para la recursión
#     log_trans = hmm.trans.T[1:-1,1:-1].copy()
#     log_trans[log_trans<zero] = zero
#     np.log(log_trans, out=log_trans)
    
    # Inicialización del log_alpha
#     log_alpha = np.zeros((N,T))
    log_alpha = np.empty((N,T))
    log_alpha[:,0] = log_b[:,0] + logA[0,1:-1].T
#     log_a0 = hmm.trans[0,1:-1].copy()
#     log_a0[log_a0<zero] = zero
#     np.log(log_a0, out=log_a0)
#     log_alpha[:,0] = log_b[0,:] + log_a0
    
    # Recursión alfa
    for t in range(1,T):
        for k in range(N):
#             log_alpha[k,t] = log_b[t,k] + logsumexp(log_trans[:,k]+log_alpha[:,t-1])
                log_alpha[k,t] = log_b[k,t] + logsumexp(logA[1:-1,k+1] + log_alpha[:,t-1])
    # Cálculo de la probabilidad
#     log_af = hmm.trans[1:-1,-1].copy()
#     log_af[log_af<zero] = zero
#     np.log(log_af, out=log_af)
    log_prob = logsumexp(logA[1:-1,-1]+log_alpha[:,-1])

    
    return log_prob, log_alpha



# Algoritmo de Viterbi:

def logvit(x,hmm):
    
    N = hmm.numStates - 2
    T = len(x)

    tmp = np.copy(hmm.trans)
    tmp[tmp<1e-320] = 1e-320
    logA = np.log(tmp)

    # Evaluación del logaritmo de la normal de cada estado y cada x_t
    log_b = hmm.logprobstate(x)
#     T = X.shape[0] # Cantidad de mediciones
#     N = hmm.numStates - 2 # Cantidad de estados sin contar el final y el inicial
#     zero = 1e-100 # Valor mínimo para evitar división por cero
    
#     # Evaluación del logaritmo de la normal 
#     log_b = logpdfnorm(hmm.means,hmm.vrs,X)
        
#     # Matriz de transición para la recursión
#     log_trans = hmm.trans[1:-1,1:-1].copy()
#     log_trans[log_trans<zero] = zero
#     np.log(log_trans, out=log_trans)
    
    # Inicialización del log_phi
    log_phi = np.zeros((N,T),dtype=float)
#     log_phi0 = hmm.trans[0,1:-1].copy()
#     log_phi0[log_phi0<zero] = zero
#     np.log(log_phi0, out=log_phi0)
    log_phi[:,0] = log_b[:,0] + logA[0,1:-1].T
    
    # Inicialización del log_psi
    log_psi = np.ones((N,T)).astype(int)

    # Recursión forward
    for i in range(1,T):        
        m = logA[1:-1,1:-1] + log_phi[:,i-1].reshape(N,1)
        log_psi[:,i] = np.argmax(m,axis=0)
        log_phi[:,i] = log_b[:,i] + np.max(m,axis=0)
#     log_af = hmm.trans[1:-1,-1].copy()
#     log_af[log_af<zero] = zero
#     np.log(log_af, out=log_af)
    m = log_phi[:,-1] + logA[1:-1,-1]
    costOpt = np.max(m)
    
    # Terminación:
    seqOpt = np.ones(T+1).astype(int)
    seqOpt[-1] = N
    seqOpt[-2] = np.argmax(m)
    
    # Backtracking:
    for i in range(T-1,0,-1):
        seqOpt[i-1] = log_psi[seqOpt[i],i].copy()
                
    return costOpt, log_phi, np.hstack((0,seqOpt+1))
