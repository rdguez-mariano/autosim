import ctypes
import cv2
import numpy as np
from matplotlib import pyplot as plt
import functools
import operator
from library import *
   
class RepeatedObject:
    def __init__(self, kplist, matches, img, lambda_descr = 6):
        self.mask = np.zeros(np.shape(img)[0:2], dtype=np.int32)
        self.kplists = [kplist]
        self.NumberOfCopies = len(kplist)
        self.lambda_descr = lambda_descr        
        X,Y=np.meshgrid(np.arange(np.shape(img)[1]),np.arange(np.shape(img)[0]))
        for i in range(len(kplist)):
            x, y = kplist[i].pt
            radius = self.DescRadius(kplist[i])
            self.mask[np.sqrt((X-x)**2+(Y-y)**2)<=radius] = i+1
        self.matches = [matches]
        self.weigth = np.sum([m.distance for m in matches])
    
    def __lt__(self, other): 
        if(self.weigth<=other.weigth): 
            return True
    
    def __eq__(self, other): 
        ''' True if two atoms should correspond to the same object
        '''
        if(self.NumberOfCopies!=other.NumberOfCopies): 
            return False
        for i in range(self.NumberOfCopies):            
            if (np.unique(other.mask[self.mask==i+1])>0).sum() !=1 or (np.unique(self.mask[other.mask==i+1])>0).sum() !=1 :
                return False                   
        return True

    def __ne__(self, other):
        ''' are the objects different objects?
        '''
        return (self.mask!=other.mask).any()


    def AddMoreAtoms(self,other):
        for i in range(self.NumberOfCopies):            
            val = sorted(np.unique(other.mask[self.mask==i+1]))
            # val[-1] is always the value different than 0     
            self.mask[other.mask==val[-1]] = i+1
        self.kplists.append(other.kplists)
                
    def ObjectInImage(self, img):        
        img2show = np.repeat(np.expand_dims(self.mask==0,2),3, axis=2)*img*1.0 
        for i in range(self.NumberOfCopies):
            colorvec = np.random.rand(3)            
            colorzone = np.zeros(np.shape(img),dtype=np.float)            
            colorzone[self.mask==i+1,:] = colorvec
            img2show += colorzone*img*0.5
        img2show = img2show.astype(np.uint8)
        for i in range(self.NumberOfCopies):
            img2show = cv2.putText(img2show, '%d'%i, tuple(np.int32(self.kplists[0][i].pt)) ,
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0),2, cv2.LINE_AA )
        return img2show

    def GetMask(self):
        return self.mask        
    
    def DescRadius(self,kp, InPyr=False):
        ''' Computes the Descriptor radius with respect to either an image
            in the pyramid or to the original image.
        '''
        factor = self.lambda_descr
        if InPyr:
            _, _, s = unpackSIFTOctave(kp)
            return( np.float32(kp.size*s*factor*0.5) )
        else:
            return( np.float32(kp.size*factor*0.5) )
        
   
   
class GroupingStrategy(object):
    def __init__(self,libASpath,ac_img_path):
        self.libAS = ctypes.cdll.LoadLibrary(libASpath)
        self.listinfo = []
        self.listacinfo = []
        self.imgfloat = []
        self.img = []
        self.imgkplist = []
        self.h = 0
        self.w = 0
        self.data_len = 5
        self.libAS.New_GS.argtypes = [ctypes.c_float, ctypes.c_int]
        self.libAS.New_GS.restype = ctypes.c_void_p
        self.libAS.Add_match.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float]
        self.libAS.Add_match.restype = None
        self.libAS.Initialize.argtypes = [ctypes.c_void_p]
        self.libAS.Initialize.restype = None
        self.libAS.Analyse.argtypes = [ctypes.c_void_p]
        self.libAS.Analyse.restype = None
        self.libAS.PrintGroups.argtypes = [ctypes.c_void_p]
        self.libAS.PrintGroups.restype = None
        self.libAS.ACMatcher.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.libAS.ACMatcher.restype = None

        self.libAS.LastGroup.argtypes = [ctypes.c_void_p]
        self.libAS.LastGroup.restype = ctypes.c_void_p
        self.libAS.FirstGroup.argtypes = [ctypes.c_void_p]
        self.libAS.FirstGroup.restype = ctypes.c_void_p
        self.libAS.NextGroup.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.libAS.NextGroup.restype = ctypes.c_void_p
        self.libAS.PrevGroup.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.libAS.PrevGroup.restype = ctypes.c_void_p

        self.libAS.NumberOfMatches.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        self.libAS.NumberOfMatches.restype = ctypes.c_int
        self.libAS.NumberOfKPs.argtypes = [ctypes.c_void_p]
        self.libAS.NumberOfKPs.restype = ctypes.c_int
        self.libAS.GetMatches.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool]
        self.libAS.GetMatches.restype = None

        self.libAS.getImagesFromGroup.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self.libAS.getImagesFromGroup.restype = None

        # self.libAS.Bind_KPs.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self.libAS.Bind_KPs.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_bool]
        self.libAS.Bind_KPs.restype = None

        img = cv2.imread(ac_img_path) # trainImage
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        klistac, dlistac = sift.detectAndCompute(gray,None)
        self.obj = self.libAS.New_GS(ctypes.c_float(4.0), ctypes.c_int(6000))
        self.Bind_KPs(klistac,dlistac,True)

    def getImagesFromGroup(self, g):
        rgb = np.zeros(self.w*self.h*3, dtype = ctypes.c_float)
        rgb_rich = np.zeros(self.w*self.h*3, dtype = ctypes.c_float)
        floatp = ctypes.POINTER(ctypes.c_float)
        self.libAS.getImagesFromGroup(g, self.imgfloat.ctypes.data_as(floatp), self.w, self.h, rgb.ctypes.data_as(floatp), rgb_rich.ctypes.data_as(floatp) )
        rgb_out =  np.zeros((self.h,self.w,3), dtype=np.uint8)
        rgb_rich_out =  np.zeros((self.h,self.w,3), dtype=np.uint8)
        # rgb_out[:,:,1] = np.reshape(rgb[0:self.w*self.h],(self.h,self.w))
        rgb_out[:] = np.reshape(rgb,(self.h,self.w,3),order='F')
        rgb_rich_out[:] = np.reshape(rgb_rich,(self.h,self.w,3),order='F')

        return rgb_out, rgb_rich_out

    def FirstLast_Groups(self):
        return self.libAS.FirstGroup(self.obj), self.libAS.LastGroup(self.obj)

    def NextGroup(self, g):
        return self.libAS.NextGroup(self.obj, g)

    def PrevGroup(self, g):
        return self.libAS.PrevGroup(self.obj, g)

    def Add_match(self,sim, id1, x1, y1, o1, s1, a1, id2, x2, y2, o2, s2, a2):
        # Add_match(GroupingStrategy* gs, float sim, int id1, float x1, float y1,int o1,float s1,float a1, int id2, float x2, float y2,int o2,float s2,float a2)
        sim, id1, x1, y1, o1, s1, a1, id2, x2, y2, o2, s2, a2 = np.float(sim), np.int32(id1), np.float(x1), np.float(y1), np.int32(o1), np.float(s1), np.float(a1), np.int32(id2), np.float(x2), np.float(y2), np.int32(o2), np.float(s2), np.float(a2)
        self.libAS.Add_match(self.obj, sim, id1, x1, y1, o1, s1, a1, id2, x2, y2, o2, s2, a2)

    def KPinfo_from_opencv(self,klist, dlist):
        Nquery, dim = dlist.shape[:2]
        x = np.zeros(Nquery, dtype = ctypes.c_float)
        y = np.zeros(Nquery, dtype = ctypes.c_float)
        octcode = np.zeros(Nquery, dtype = ctypes.c_int)
        size = np.zeros(Nquery, dtype = ctypes.c_float)
        angle = np.zeros(Nquery, dtype = ctypes.c_float)
        desc = np.zeros(Nquery*dim, dtype = ctypes.c_float)
        for i in range(0,Nquery):
            # x, y, octcode, size, angle, len, desc_dim, desc
            x[i] = klist[i].pt[0]
            y[i] = klist[i].pt[1]
            octcode[i] = klist[i].octave
            size[i] = klist[i].size
            angle[i] = klist[i].angle
            desc[(i*dim):((i+1)*dim)] = dlist[i,:].astype(ctypes.c_float)
        return x, y, octcode, size, angle, Nquery, dim, desc


    def Bind_KPs(self, klist, dlist, am_i_ac):
        x, y, octcode, size, angle, len, desc_dim, desc = self.KPinfo_from_opencv(klist, dlist)
        # point these variables so python do not trash them
        if am_i_ac:
            self.listinfoac = x, y, octcode, size, angle, len, desc_dim, desc
        else:
            self.listinfo = x, y, octcode, size, angle, len, desc_dim, desc

        # float *x, float *y, int *octcode,float *size,float *angle, int len, int desc_dim, float* desc)
        intp = ctypes.POINTER(ctypes.c_int)
        floatp = ctypes.POINTER(ctypes.c_float)
        xp = x.ctypes.data_as(floatp)
        yp = y.ctypes.data_as(floatp)
        octcodep = octcode.ctypes.data_as(intp)
        sizep = size.ctypes.data_as(floatp)
        anglep = angle.ctypes.data_as(floatp)
        len = ctypes.c_int(len)
        desc_dim = ctypes.c_int(desc_dim)
        descp = desc.ctypes.data_as(floatp)

        self.libAS.Bind_KPs(self.obj, xp, yp, octcodep, sizep, anglep, len, desc_dim, descp, am_i_ac)

    def LookForAutoSims(self, img, matchratio):
        self.img = img
        gray1= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self.h, self.w = gray1.shape[:2]
        self.imgfloat = np.zeros(self.w*self.h, dtype = ctypes.c_float)
        self.imgfloat[:] = np.array(gray1.data).flatten()
        sift = cv2.xfeatures2d.SIFT_create()
        klist, dlist = sift.detectAndCompute(gray1,None)
        self.imgkplist = klist
        self.Bind_KPs(klist,dlist,False)
        self.ACMatcher(matchratio)

    def NumberOfKPs(self,g):
        return self.libAS.NumberOfKPs(g)

    def GroupNumberOfMatches(self,g):
        Nint = self.libAS.NumberOfMatches(g, ctypes.c_bool(True))
        Next = self.libAS.NumberOfMatches(g, ctypes.c_bool(False))
        return Nint, Next

    def ShowGroup(self, g, Flag=None, img2use=None):        
        floatp = ctypes.POINTER(ctypes.c_float)
        Interior = False
        if Flag is not None:
            Interior = Flag
        
        trueKP = False
        NFM = self.libAS.NumberOfMatches(g, ctypes.c_bool(Interior))
        FM = np.zeros(self.data_len*NFM, dtype = ctypes.c_float)
        self.libAS.GetMatches(g, FM.ctypes.data_as(floatp), ctypes.c_bool(Interior), ctypes.c_bool(trueKP))                        

        ptlist = [[np.complex(real = float(FM[self.data_len*i]), imag = float(FM[self.data_len*i+1])),
                   np.complex(real = float(FM[self.data_len*i+2]), imag = float(FM[self.data_len*i+3]))
                    ]  for i in range(NFM)]
        
        kplist = [[cv2.KeyPoint(x = float(FM[self.data_len*i]), y = float(FM[self.data_len*i+1]),
                    _size = 6.0, _angle = 0.0, _response = 0.9,
                     _octave = packSIFTOctave(-1,0), _class_id = 0),
                    cv2.KeyPoint(x = float(FM[self.data_len*i+2]), y = float(FM[self.data_len*i+3]),
                    _size = 6.0, _angle = 0.0, _response = 0.9,
                     _octave = packSIFTOctave(-1,0), _class_id = 0)
                    ]  for i in range(NFM)]
        kplist = functools.reduce(operator.iconcat, kplist, [])
        ptlist = functools.reduce(operator.iconcat, ptlist, [])
        
        ptlist, indices, inv_ind = np.unique(ptlist, return_index=True, return_inverse=True)

        kplist = [kplist[i] for i in indices]
        Matches = [cv2.DMatch(inv_ind[2*i],inv_ind[2*i+1],FM[self.data_len*i+4]) for i in range(NFM)]

        ro = RepeatedObject(kplist,self.img)

        # Matches = [cv2.DMatch(inv_ind[2*i],inv_ind[2*i+1],,FM[self.data_len*i+4]) for i in range(NFM)]
        if img2use is not None:
            img2show = img2use.copy()
        else:
            img2show = self.img.copy()
        if Interior:
            colorvec = np.random.rand(3)*255
            img2show = (ro.GetMask()==0)*img2show/2 + (ro.GetMask()>0)*img2show
        else:
            colorvec = (0,0,0)
        for m in Matches:            
            img2show = cv2.line(img2show, tuple(np.int32(np.round(kplist[m.queryIdx].pt))), tuple(np.int32(np.round(kplist[m.trainIdx].pt))), colorvec, 2) 
        for kp in kplist:
            # pass
            img2show = cv2.circle(img2show, tuple(np.int32(np.round(kp.pt))), 9, colorvec, -1)#, lineType=0)         

        if Flag is None:
            return self.ShowGroup(g, Flag=True, img2use=img2show)
        else:
            return img2show

    def GetRepeatedObjectFromGroup(self, g):        
        floatp = ctypes.POINTER(ctypes.c_float)
        Interior = True        
        trueKP = False

        NFM = self.libAS.NumberOfMatches(g, ctypes.c_bool(Interior))
        FM = np.zeros(self.data_len*NFM, dtype = ctypes.c_float)
        self.libAS.GetMatches(g, FM.ctypes.data_as(floatp), ctypes.c_bool(Interior), ctypes.c_bool(trueKP))                        

        if trueKP:
            kplist = [ [self.imgkplist[int(FM[self.data_len*i])], 
                        self.imgkplist[int(FM[self.data_len*i+1])] ] for i in range(NFM)]
            kplist = functools.reduce(operator.iconcat, kplist, [])
            matches = [cv2.DMatch(2*i,2*i+1,FM[self.data_len*i+4]) for i in range(NFM)]
        else:
            ptlist = [[np.complex(real = float(FM[self.data_len*i]), imag = float(FM[self.data_len*i+1])),
                    np.complex(real = float(FM[self.data_len*i+2]), imag = float(FM[self.data_len*i+3]))
                        ]  for i in range(NFM)]
            
            kplist = [[cv2.KeyPoint(x = float(FM[self.data_len*i]), y = float(FM[self.data_len*i+1]),
                        _size = 6.0, _angle = 0.0, _response = 0.9,
                        _octave = packSIFTOctave(-1,0), _class_id = 0),
                        cv2.KeyPoint(x = float(FM[self.data_len*i+2]), y = float(FM[self.data_len*i+3]),
                        _size = 6.0, _angle = 0.0, _response = 0.9,
                        _octave = packSIFTOctave(-1,0), _class_id = 0)
                        ]  for i in range(NFM)]
            kplist = functools.reduce(operator.iconcat, kplist, [])
            ptlist = functools.reduce(operator.iconcat, ptlist, [])
            
            ptlist, indices, inv_ind = np.unique(ptlist, return_index=True, return_inverse=True)

            kplist = [kplist[i] for i in indices]
            matches = [cv2.DMatch(inv_ind[2*i],inv_ind[2*i+1],FM[self.data_len*i+4]) for i in range(NFM)]

        ro = RepeatedObject(kplist,matches,self.img)
        return ro

    def Analyse(self):
        self.libAS.Initialize(self.obj)
        self.libAS.Analyse(self.obj)

    def PrintGroups(self):
        self.libAS.PrintGroups(self.obj)

    def getGroups(self):
        groups = []
        f, l = self.FirstLast_Groups()
        cNKP = self.NumberOfKPs(l)
        ros = []
        while True:            
            if self.NumberOfKPs(l)!=cNKP: 
                ros = sorted(ros,reverse=True)            
                groups.append( (cNKP, ros) )
                ros = []
                cNKP = self.NumberOfKPs(l)
            ros.append(self.GetRepeatedObjectFromGroup(l))
            if (l==f):
                ros = sorted(ros,reverse=True)
                groups.append( (cNKP, ros) )
                break
            l = gs.PrevGroup(l)
        return groups

    def ACMatcher(self, matchratio):
        self.libAS.ACMatcher(self.obj,ctypes.c_float(matchratio))
    
    @staticmethod
    def MergeWithMe(ro,ros):
        resros = []
        merged = False
        for ro2 in ros:
            if ro==ro2 and ro!=ro2: 
                # equality says that they can be merged
                # not equal means they are not the same atom
                merged = True
                ro.AddMoreAtoms(ro2)
            else:
                resros.append(ro2)
        return resros, merged




gs=GroupingStrategy('./build/libautosim.so', 'im3_sub.png')


img1 = cv2.imread('coca.png')          # queryImage
# img1 = cv2.imread('build/20180322121141_20561BWsym.png')          # queryImage
# img1 = cv2.imread('build/20180322121253_18903BWsym.png')          # queryImage
# img1 = cv2.imread('build/coca.png')          # queryImage

gs.LookForAutoSims(img1, 0.8)

gs.Analyse()
gs.PrintGroups()
groups = gs.getGroups()





import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, mixture, neighbors
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
def getClusters(ptlist,n_clusters, img):
    plt.figure()
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    plot_num = 1
    Ncolors = int(n_clusters+ 1)
    params = {'quantile': .3, 'eps': .3, 'damping': .9, 'preference': -200,
                'n_neighbors': int(n_clusters*1.4), 'n_clusters': n_clusters,
                'min_samples': 20, 'xi': 0.05, 'min_cluster_size': 0.1}
    X = np.array(ptlist)
    # X = StandardScaler().fit_transform(X)
    persub_y_pred = [i for i in range(n_clusters)]

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity)
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors",n_neighbors = params['n_neighbors'])
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full',
        means_init = np.array(X[0:params['n_clusters'],:]))
    dpgmm = mixture.BayesianGaussianMixture(n_components=params['n_clusters'], 
        covariance_type='full', weight_concentration_prior=1e-2,
        weight_concentration_prior_type='dirichlet_process',
        mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(2),
        init_params="random", max_iter=100, random_state=2)
    knn = neighbors.KNeighborsClassifier(1)
    knn.fit(np.array(X[0:params['n_clusters'],:]), persub_y_pred)

    clustering_algorithms = (
        ('SpectralClustering', spectral),
        # ('Ward', ward),
        # ('AgglomerativeClustering', average_linkage),
        # ('Birch', birch),
        ('GaussianMixture', gmm),
        ('KNN', knn),
        # ('BayesianGaussianMixture', dpgmm)
    )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            if name != 'KNN':
                algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)
        
        # print(np.shape(y_pred),y_pred)
        for i in range(int(len(y_pred)/n_clusters)):
            sub_y_pred = np.array(y_pred[i*n_clusters:(i+1)*n_clusters])
            if np.all(np.isin(persub_y_pred,sub_y_pred)):
                # print(persub_y_pred,sub_y_pred)
                pass
            else:
                # print(persub_y_pred,sub_y_pred)
                y_pred[i*n_clusters:(i+1)*n_clusters] = Ncolors

        plt.subplot(1, len(clustering_algorithms), plot_num)        
        plt.title(name)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      Ncolors)))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=100, color=colors[y_pred])

        # plt.xlim(-2.5, 2.5)
        # plt.ylim(-2.5, 2.5)
        imggray = np.repeat(np.expand_dims(np.sum(img,axis=-1)/3.0, axis=-1), 3, axis=-1)        
        imggray = imggray.astype(np.uint8)
        
        plt.imshow(imggray)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.5fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1
    plt.savefig('clustering.png', format='png', dpi=300)
    plt.show()



for i,(cNKP,ros) in enumerate(groups):
    print(cNKP, len(ros))
    if cNKP == 6 :
        ptlist= [kp.pt for ro in ros for kp in ro.kplists[0]]
        getClusters(ptlist, cNKP, gs.img.copy())
print(np.shape(ptlist))
