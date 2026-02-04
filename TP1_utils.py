import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML, Javascript

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"



def PlotPoints(pts, clr='k', phi=lambda x:x, withgrid=True, ng=50,
               title=None, animate=False):
    # Fonction pour l'affichage d'une configuration de points, avec ou sans déformation.
    # arguments :
    #    pts : tableau (N,2), coordonnées de points additionnels.
    #    clr : optionnel, caractère (ex 'r' ou 'b') spécifiant la couleur utilisée pour afficher les points.
    #    phi : optionnel, fonction Python renvoyée par MatchingLinear ou une méthode similaire,
    #          donnant la transformation phi. Si phi est donné, la transformation phi est appliquée 
    #          sur les points pts.
    #    withgrid : optionnel, si False, on n'affiche pas la grille de déformation.
    #    title : optionnel, titre de la figure
    fig = plt.gcf()
    plt.axis('equal')
    plt.axis('off')
    if withgrid:
        # définition d'une grille carrée adaptée aux points
        mn, mx = pts.min(axis=0)[0], pts.max(axis=0)[0]
        c, sz = (mn+mx)/2, 1.2*(mx-mn)
        a, b = c-sz/2, c+sz/2
        X1, X2 = torch.meshgrid(torch.linspace(a[0],b[0],ng, device=device),
                                torch.linspace(a[1],b[1],ng, device=device),
                                indexing="ij")
        grid = torch.concatenate((X1.reshape(ng*ng,1),X2.reshape(ng*ng,1)),axis=1)
        phigrid = phi(grid)
        phiX1 = phigrid[:,0].reshape(ng,ng)
        phiX2 = phigrid[:,1].reshape(ng,ng)
        gridplot1 = plt.plot(phiX1.cpu(),phiX2.cpu(),clr,linewidth=.25)
        gridplot2 = plt.plot(phiX1.cpu().T,phiX2.cpu().T,clr,linewidth=.25)
    phipts = phi(pts)
    markersize = 10**(1.5-.5*np.log10(pts.shape[0]))
    ptsplot, = plt.plot(phipts[:,0].cpu(),phipts[:,1].cpu(),'.'+clr,markersize=markersize)
    if title:
        plt.title(title)
    if animate:
        frames = 10
        interval = 100
        plt.close()
        def animate(i):
            phiptsi = pts + (i/(frames-1)) * (phipts-pts)
            ptsplot.set_xdata(phiptsi[:,0].cpu())
            ptsplot.set_ydata(phiptsi[:,1].cpu())
            if withgrid:
                phigridi = grid + (i/(frames-1)) * (phigrid-grid)
                phiX1i = phigridi[:,0].reshape(ng,ng)
                phiX2i = phigridi[:,1].reshape(ng,ng)
                for k in range(ng):
                    g1,g2 = gridplot1[k], gridplot2[k]
                    g1.set_xdata(phiX1i[k,:].cpu())
                    g1.set_ydata(phiX2i[k,:].cpu())
                    g2.set_xdata(phiX1i[:,k].cpu())
                    g2.set_ydata(phiX2i[:,k].cpu())
                return (ptsplot,*gridplot1,*gridplot2)
            else:
                return (ptsplot,)
        anim = animation.FuncAnimation(fig, animate, 
                                   frames=frames, interval=interval, 
                                   blit=True, repeat=False)
        rc('animation', html='jshtml')
        return anim

def show_anim_transport(xs,xt,G0):

    frames = 10
    interval = 100

    # First set up the figure, the axis, and the plot element we want to animate
    plt.axis('off')
    fig = plt.gcf()
    x = torch.vstack((xs,xt))
    mn, mx = x.min(axis=0)[0], x.max(axis=0)[0]
    c, sz = (mn+mx)/2, 1.2*(mx-mn)
    a, b = c-sz/2, c+sz/2
    plt.axis([a[0].item(), b[0].item(), a[1].item(), b[1].item()])
    xs = xs[:,None,:]
    v = xt[None,:,:] - xs
    x0 = xs + 0 * v
    n = xs.shape[0]
    G0 = G0.reshape(n**2)
    ind = torch.nonzero(G0).flatten()
    G0 = G0[ind]
    N = len(ind)
    x0 = x0.reshape((n**2,2))[ind,:]
    v = v.reshape((n**2,2))[ind,:]
    markersize = 10**(1.5-.5*np.log10(N))
    scat = plt.scatter(x0[:,0].cpu(), x0[:,1].cpu(), marker='.', alpha=G0.cpu()*n,
                      s=markersize)
    plt.scatter(xt[:,0].cpu(), xt[:,1].cpu(), marker='.',
                      s=markersize)

    # animation function. This is called sequentially  
    def animate(i):
        xi = x0 + (i/(frames-1)) * v
        scat.set_offsets(xi.cpu())
        return (scat,)
  
    anim = animation.FuncAnimation(fig, animate, 
                                   frames=frames, interval=interval, 
                                   blit=True, repeat=False)
    rc('animation', html='jshtml')
    plt.close()
    return anim
        
def load(fname='store.pckl'):
    # chargement d'un fichier pickle
    import pickle
    f = open(fname, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj