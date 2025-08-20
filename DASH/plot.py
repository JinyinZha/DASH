import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
import numpy as np

class Plot():
    def __init__(self,outdir):
        self.outdir = outdir   
    def total(self,loss,z):
        if loss:
            self.loss(loss)
        self.z(z)
      
    def loss(self,loss,labels=None,name=None):
        plt.figure(dpi=500)
        if labels != None:
            for i,l in enumerate(loss):
                plt.plot(l,label=labels[i])
            plt.legend()
        else:
            plt.plot(loss)
        plt.xticks(fontproperties="Times New Roman",fontsize=12)
        plt.yticks(fontproperties="Times New Roman",fontsize=12)
        plt.xlabel("Epoch",fontdict={"family":"Times New Roman","size":13})
        plt.ylabel("Loss",fontdict={"family":"Times New Roman","size":13})
        if name == None:
            name = "loss"
        plt.savefig("%s/%s.jpg"%(self.outdir,name))
        plt.close()
        
    def z(self,z,bx=30,by=30,xlim=(),ylim=()):
        #Hist2D
        s1=23
        s2=s1+2
        plt.figure(dpi=300,figsize=(9,9))
        plt.hist2d(z[:,0],z[:,1],bins=(bx,by),norm=LogNorm(),cmap="rainbow")
        plt.xticks(fontsize=s1)
        plt.yticks(fontsize=s1)
        plt.xlabel("Z1",fontdict={"size":s2})
        plt.ylabel("Z2",fontdict={"size":s2})
        plt.tight_layout()
        plt.savefig("%s/hist2d.jpg"%(self.outdir))
        plt.close()
        #Contour
        kB = 8.314 * 0.00023885 #kcal/(mol*K)
        T = 300
        zz,x,y=np.histogram2d(z[:,0],z[:,1],bins=(bx,by))
        zz = zz.T
        xx = np.array([(x[i+1]+x[i])/2 for i in range(bx)])
        yy = np.array([(y[i+1]+y[i])/2 for i in range(by)])
        e = np.zeros(zz.shape)
        e_max = -1
        for i in range(len(zz)):
            for j in range(len(zz[i])):
                if zz[i,j] != 0:
                    tmp_e = -np.log(zz[i,j] / np.max(zz))
                    if tmp_e > e_max:
                        e_max = tmp_e
                    e[i,j] = tmp_e
                else:
                    e[i,j] = 99999
        plt.figure(dpi=300,figsize=(10,6))
        C=plt.contourf(xx,yy,e,levels=np.linspace(0,e_max,15),cmap="rainbow",vmin=0)
        if len(xlim) == 2:
            plt.xlim(xlim)
        if len(ylim) == 2:
            plt.ylim(ylim)
        '''for c in C.collections:
            c.set_edgecolor("face")'''
        plt.xticks(fontsize=s1)
        plt.yticks(fontsize=s1)
        plt.xlabel("CV1",fontdict={"size":s2})
        plt.ylabel("CV2",fontdict={"size":s2})
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=s1)
        cbar.set_label("Free Energy / k$_B$T", fontsize=s2, rotation=270,labelpad=40)
        plt.tight_layout()
        plt.savefig("%s/countour.jpg"%(self.outdir))
        plt.close()

    def z_scatter(self,z,sts,eds):
        plt.figure(dpi=500)
        for i in range(len(sts)):
            plt.plot(z[sts[i]:eds[i],0],z[sts[i]:eds[i],1],"o",ms=1)
        plt.savefig("%s/scatter.jpg"%(self.outdir))
        plt.close()
        
    def cvs_on_oldz(self,z0,z,name,s=2,c=None,cm=None):
        kB = 8.314 * 0.00023885 #kcal/(mol*K)
        T = 300
        bx = 20
        by = 20
        zz,x,y=np.histogram2d(z0[:,0],z0[:,1],bins=(bx,by))
        zz = zz.T
        xx = np.array([(x[i+1]+x[i])/2 for i in range(bx)])
        yy = np.array([(y[i+1]+y[i])/2 for i in range(by)])
        e = np.zeros(zz.shape)
        for i in range(len(zz)):
            for j in range(len(zz[i])):
                if zz[i,j] != 0:
                    e[i,j] = -kB*T*np.log(zz[i,j] / np.max(zz))
                else:
                    e[i,j] = -1
        plt.figure(dpi=500)
        C=plt.contourf(xx,yy,e,levels=np.linspace(0,np.max(e),50))
        #plt.plot(z[:,0],z[:,1],lw=0.5) 
        if c:
             plt.scatter(z[:,0],z[:,1],s=s,color=c)
        else:
            if cm:
                plt.scatter(z[:,0],z[:,1],s=s,cmap="rainbow",c=cm)
            else:
                plt.scatter(z[:,0],z[:,1],s=s,cmap="rainbow",c=list(range(len(z))))
        cb=plt.colorbar()
        #plt.xlim(z0[:,0].min(),z0[:,0].max())
        #plt.ylim(z0[:,1].min(),z0[:,1].max())
        plt.savefig("%s/%s.jpg"%(self.outdir,name))
        plt.close()

    def plot_energy_surface(self,x,y,z,name):
        xi = np.linspace(min(x), max(x), 100)
        yi = np.linspace(min(y), max(y), 100)
        XI, YI = np.meshgrid(xi, yi)
        ZI = griddata((x, y), z, (XI, YI), method='linear')
        plt.figure(dpi=500)
        plt.contourf(XI, YI, ZI,cmap='rainbow')
        #plt.colorbar()
        plt.savefig("%s/Energy_Surface_%s.jpg"%(self.outdir,name))
        plt.close()
        
    def pre_ref_line(self,x,x_ref,name):
        allx = np.hstack((x,x_ref))
        plt.figure(dpi=500)
        plt.plot(x,x_ref,"o",ms=2)
        plt.plot([np.min(allx),np.max(allx)],[np.min(allx),np.max(allx)],lw=4,color="black")
        plt.xlabel("pre")
        plt.ylabel("ref")
        plt.savefig("%s/%s.jpg"%(self.outdir,name))
        plt.close()

        
        
if __name__ == "__main__":
    import sys
    folder = "train_splitcontrastmae_1_1/Train_SplitContrastMAE_egfr_DES"
    xlim = ()
    ylim = ()
    pt = Plot(folder)
    z = np.load("%s/z.npy"%(folder))
    pt.z(z,bx=20,by=20,xlim=xlim,ylim=ylim)