#!/usr/bin/env python
import sys, getopt

import ctypes
import cv2
import numpy as np
from matplotlib import pyplot as plt
import functools
import operator
from library import *
from AffRANSAC import *
   
class RepeatedObject:
    def __init__(self, kplist, matches, truematches, img, lambda_descr = 6):
        self.mask = np.zeros(np.shape(img)[0:2], dtype=np.int32)
        self.kplists = [kplist]
        self.NumberOfCopies = len(kplist)        
        self.AffineMaps = [None for _ in range(len(kplist)*len(kplist))]
        self.lambda_descr = lambda_descr        
        # X,Y=np.meshgrid(np.arange(np.shape(img)[1]),np.arange(np.shape(img)[0]))
        # for i in range(len(kplist)):
        #     x, y = kplist[i].pt
        #     radius = self.DescRadius(kplist[i])
        #     self.mask[np.sqrt((X-x)**2+(Y-y)**2)<=radius] = i+1
        self.matches = [matches]
        self.truematches = [truematches] # with respect to the true kplist on the image
        self.weigth = np.sum([m.distance for m in matches])
    
    def getAffineMap(self,i,j): # Affine maps going from atom i to atom j
        return self.AffineMaps[i*self.NumberOfCopies+j]

    def isAffineMapUnset(self,i,j): 
        return self.AffineMaps[i*self.NumberOfCopies+j] is None
    
    def setAffineMap(self,i,j,A): # Affine maps going from atom i to atom j
        if self.AffineMaps[i*self.NumberOfCopies+j] is None:
            self.AffineMaps[i*self.NumberOfCopies+j] = A  
            # self.AffineMaps[j*self.NumberOfCopies+i] = np.linalg.inv(A)             
            self.AffineMaps[j*self.NumberOfCopies+i] = cv2.invertAffineTransform(A)                         
            return True
        else:
            return False

    def UpdateKeys(self,truekp, kpind):
        trueradius = self.DescRadius(truekp)
        for (j,kp) in enumerate(self.kplists[0]):
            A = self.getAffineMap(kpind,j)                   
            temp = AffineKPcoor([truekp], A)
            kp.angle = temp[0].angle
            Adecomp = affine_decomp(A,doAssert=False)
            kp.size = trueradius*Adecomp[0]

    def CreateMask(self,fixed_inds=None):            
        X,Y=np.meshgrid(np.arange(np.shape(self.mask)[1]),np.arange(np.shape(self.mask)[0]))
        # first keypoints correspond to highly score matches
        # so, their mask should not be overwritten
        ckplist = reversed(features_deepcopy(self.kplists[0]))
        if fixed_inds is not None:
            for (i,n) in enumerate(fixed_inds):
                x, y = self.kplists[0][n].pt
                radius = self.DescRadius(self.kplists[0][n])
                self.mask[np.sqrt((X-x)**2+(Y-y)**2)<=radius] = i+1
        else:            
            for (i,kp) in enumerate(ckplist):
                x, y = kp.pt
                radius = self.DescRadius(kp)
                self.mask[np.sqrt((X-x)**2+(Y-y)**2)<=radius] = i+1
    
    def CompleteAffineMaps(self, diag=True):        
        if diag:
            for i in range(self.NumberOfCopies):
                self.AffineMaps[i*self.NumberOfCopies+i] = np.double([[1, 0, 0], [0, 1, 0]])        
        ReDo = False
        for j in range(self.NumberOfCopies): 
            for i in range(self.NumberOfCopies):
                if self.AffineMaps[i*self.NumberOfCopies+j] is None:
                    found = False
                    for z in range(self.NumberOfCopies):
                        if z not in [i,j] and not self.isAffineMapUnset(i,z) and not self.isAffineMapUnset(z,j):
                            self.setAffineMap(i,j,ComposeAffineMaps( self.getAffineMap(z,j) ,self.getAffineMap(i,z) ))
                            found = True
                            break
                    if not found:
                        ReDo = True
        if ReDo:
            # print("Need to relaunch CompleteAffineMaps")
            self.CompleteAffineMaps(diag=False)

    def VisualizeAffineMaps(self,img):
        fixi = 2#np.random.randint(1,self.NumberOfCopies)
        w, h = self.kplists[0][fixi].pt
        pxl_radius = 20
        keys2seek = [cv2.KeyPoint(x = w+pxl_radius*i - pxl_radius/2, y = h +pxl_radius*j - pxl_radius/2,
            _size =  6.0, 
            _angle =  0.0,
            _response =  1.0, _octave =  packSIFTOctave(-1,0),
            _class_id =  i*2+j) for i in range(0,2) for j in range(0,2)]
        keys2seek.append(cv2.KeyPoint(x = w, y = h,
            _size =  6.0, 
            _angle =  0.0,
            _response =  1.0, _octave =  packSIFTOctave(-1,0),
            _class_id =  -1))        
        
        img = WriteImgKeys(img, keys2seek) 
        for j in range(self.NumberOfCopies):
            temp = AffineKPcoor(keys2seek, self.getAffineMap(fixi,j))
            img = WriteImgKeys(img, temp) 
        return img
    
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
    def __init__(self,libASpath,ac_img_path,rho = 4.0, maxNumMatches = 6000):
        self.libAS = ctypes.cdll.LoadLibrary(libASpath)
        self.listinfo = []
        self.listacinfo = []
        self.imgfloat = []
        self.img = []
        self.imgkplist = []
        self.Akps = []
        self.h = 0
        self.w = 0
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
        self.obj = self.libAS.New_GS(ctypes.c_float(rho), ctypes.c_int(maxNumMatches))
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

    def LookForAutoSims(self, img, matchratio, max_width = np.inf):
        #percent by which the image is resized
        scale = max_width/img.shape[1]
        if scale < 1:
            #calculate the 50 percent of original dimensions
            width = int(img.shape[1] * scale)
            height = int(img.shape[0] * scale)
            # dsize
            dsize = (width, height)
            # resize image
            img = cv2.resize(img, dsize)
        self.img = img
        gray1= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self.h, self.w = gray1.shape[:2]
        self.imgfloat = np.zeros(self.w*self.h, dtype = ctypes.c_float)
        self.imgfloat[:] = np.array(gray1.data).flatten()
        sift = cv2.xfeatures2d.SIFT_create()
        klist, dlist = sift.detectAndCompute(gray1,None)
        self.imgkplist = klist
        self.Akps = [kp2LocalAffine(kp) for kp in klist]
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
        FM = np.zeros(data_len*NFM, dtype = ctypes.c_float)
        self.libAS.GetMatches(g, FM.ctypes.data_as(floatp), ctypes.c_bool(Interior), ctypes.c_bool(trueKP))                        

        ptlist = [[np.complex(real = float(FM[data_len*i]), imag = float(FM[data_len*i+1])),
                   np.complex(real = float(FM[data_len*i+2]), imag = float(FM[data_len*i+3]))
                    ]  for i in range(NFM)]
        
        kplist = [[cv2.KeyPoint(x = float(FM[data_len*i]), y = float(FM[data_len*i+1]),
                    _size = 6.0, _angle = 0.0, _response = 0.9,
                     _octave = packSIFTOctave(-1,0), _class_id = 0),
                    cv2.KeyPoint(x = float(FM[data_len*i+2]), y = float(FM[data_len*i+3]),
                    _size = 6.0, _angle = 0.0, _response = 0.9,
                     _octave = packSIFTOctave(-1,0), _class_id = 0)
                    ]  for i in range(NFM)]
        kplist = functools.reduce(operator.iconcat, kplist, [])
        ptlist = functools.reduce(operator.iconcat, ptlist, [])
        
        ptlist, indices, inv_ind = np.unique(ptlist, return_index=True, return_inverse=True)

        kplist = [kplist[i] for i in indices]
        Matches = [cv2.DMatch(inv_ind[2*i],inv_ind[2*i+1],FM[data_len*i+4]) for i in range(NFM)]

        ro = RepeatedObject(kplist,Matches,self.img)

        # Matches = [cv2.DMatch(inv_ind[2*i],inv_ind[2*i+1],,FM[data_len*i+4]) for i in range(NFM)]
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
        trueKP = True

        if trueKP:
            data_len=7
        else:
            data_len=5

        NFM = self.libAS.NumberOfMatches(g, ctypes.c_bool(Interior))
        FM = np.zeros(data_len*NFM, dtype = ctypes.c_float)
        self.libAS.GetMatches(g, FM.ctypes.data_as(floatp), ctypes.c_bool(Interior), ctypes.c_bool(trueKP))                        
        
        ptlist = [[np.complex(real = float(FM[data_len*i]), imag = float(FM[data_len*i+1])),
                np.complex(real = float(FM[data_len*i+2]), imag = float(FM[data_len*i+3]))
                    ]  for i in range(NFM)]
        
        kplist = [[cv2.KeyPoint(x = float(FM[data_len*i]), y = float(FM[data_len*i+1]),
                    _size = 6.0, _angle = 0.0, _response = 0.9,
                    _octave = packSIFTOctave(-1,0), _class_id = 0),
                    cv2.KeyPoint(x = float(FM[data_len*i+2]), y = float(FM[data_len*i+3]),
                    _size = 6.0, _angle = 0.0, _response = 0.9,
                    _octave = packSIFTOctave(-1,0), _class_id = 0)
                    ]  for i in range(NFM)]
        kplist = functools.reduce(operator.iconcat, kplist, [])
        ptlist = functools.reduce(operator.iconcat, ptlist, [])
        
        ptlist, indices, inv_pt = np.unique(ptlist, return_index=True, return_inverse=True)

        kplist = [kplist[i] for i in indices]
        matches = [cv2.DMatch(inv_pt[2*i],inv_pt[2*i+1],FM[data_len*i+4]) for i in range(NFM)]
        truematches = [cv2.DMatch(int(FM[data_len*i+5]), int(FM[data_len*i+6]), FM[data_len*i+4]) for i in range(NFM)]

        ro = RepeatedObject(kplist,matches,truematches,self.img)

        # Affine maps from atom i to atom j
        if trueKP:            
            A_p1_to_p2 = [np.float32([[1, 0, 0], [0, 1, 0]]) for _ in truematches]                        
            for (i,m) in enumerate(truematches):                
                A_query_to_p2 = ComposeAffineMaps(A_p1_to_p2[i], self.Akps[m.queryIdx])
                # A_p1_to_target = ComposeAffineMaps( cv2.invertAffineTransform(Akp[m.trainIdx]), A_p1_to_p2 )
                A_query_to_target = ComposeAffineMaps( cv2.invertAffineTransform(self.Akps[m.trainIdx]), A_query_to_p2 )               
                ro.setAffineMap(inv_pt[2*i],inv_pt[2*i+1], A_query_to_target)
                # ro.setAffineMap(inv_pt[2*i],inv_pt[2*i+1],Aq2t[0:2,0:2])
            ro.CompleteAffineMaps()
            ro.UpdateKeys(self.imgkplist[truematches[0].queryIdx],inv_pt[0])            
            ro.CreateMask()
            
        # # # patch=cv2.drawKeypoints(self.img.copy(), ro.kplists[0],self.img.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)        
        # # # cv2.imwrite("./temp/kps.png",patch)
        # # cv2.imwrite("./temp/mask.png",ro.mask*255.0/ro.NumberOfCopies)
        # # cv2.imwrite("./temp/kps.png",ro.VisualizeAffineMaps(self.img.copy()))        
        # exit()

        # persub_y_pred = [i for i in range(ro.NumberOfCopies)]
        # if np.all(np.isin(persub_y_pred,np.unique(ro.mask.ravel()))):
        #     pass
        # else:
        #     print('Error',ro.NumberOfCopies,persub_y_pred,np.unique(ro.mask.ravel()))
        #     for kp in kplist:
        #         kp.size *= 6
        #     patch=cv2.drawKeypoints(self.img.copy(), kplist,self.img.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #     cv2.imwrite("./temp/mask.png",ro.mask*255.0/ro.NumberOfCopies)
        #     cv2.imwrite("./temp/kps.png",patch)
        #     exit()
        return ro

    def getHomographyConsistentObject(self,ros, AffInfo=1):
        dataransac = RepData4ransac(ros)                        
        RepAff_RANSAC_H(dataransac, self.img, AffInfo=AffInfo)   
        return dataransac        

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
        while cNKP>0:            
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
            l = self.PrevGroup(l)
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



def main(argv):
    try:
        opts, args = getopt.getopt(argv,'hq:r:m:i:n:a:l:w:',['queryimage','rho','matchratio', 'affinfo', 'maxNumMatches','acimage','librarypath','max_width'])
    except getopt.GetoptError:
        print('-q <queryimage> -r <rhovalue> -m <matchratio> -i <affinfo> -n <maxNumMatches> -w <maxImgWidth>')
        sys.exit(2)
    # default parameters
    img1 = cv2.imread('coca.png')
    acpath = 'im3_sub.png'
    rho = 4.0
    maxNumMatches = 6000
    affinfo = 1
    matchratio = 0.8
    lpath = './build/libautosim.so'
    maxwidth = np.inf

    for opt, arg in opts:
        if opt == '-h':
            print('test.py -q <queryimage> -r <rhovalue> -m <matchratio> -i <affinfo> -n <maxNumMatches>')
            sys.exit()
        elif opt in ("-q", "--queryimage"):
            img1 = cv2.imread(arg)       
        elif opt in ("-r", "--rho"):
            rho = float(arg)
        elif opt in ("-m", "--matchratio"):
            matchratio = float(arg)
        elif opt in ("-i", "--affinfo"):
            affinfo = int(arg)
        elif opt in ("-n", "--maxNumMatches"):
            maxNumMatches = int(arg)
        elif opt in ("-a", "--acimage"):
            acpath = arg
        elif opt in ("-l", "--librarypath"):
            lpath = arg
        elif opt in ("-w", "--max_width"):
            maxwidth = int(arg)

    gs=GroupingStrategy(lpath, acpath, rho=rho, maxNumMatches=maxNumMatches)

    gs.LookForAutoSims(img1, matchratio, max_width=maxwidth)

    gs.Analyse()
    gs.PrintGroups()
    groups = gs.getGroups()

    nfavec = []
    dataransacvec = []
    print("Best negative logNFA for fixed C cardinalities")
    for i,(cNKP,ros) in enumerate(groups):        
        dataransac = gs.getHomographyConsistentObject(ros,AffInfo=affinfo)
        # dataransac.visualizeRepeatedObject(0,self.img)
        # dataransac.visualizeHomograhy(1,self.img)                
        if dataransac.logNFA<0:        
            print("   C = %d ---> logNFA = %.3f, consensus: %d out of %d  " % (cNKP, dataransac.logNFA, len(dataransac.H_listconsensus), len(ros)))
            nfavec.append( dataransac.logNFA )
            dataransacvec.append( dataransac )        

    num2show = 3
    org = (10, 30)
    ind = np.argsort(nfavec)
    ind = ind[0:3]
    for (i,oi) in enumerate(ind):
        dataransac = dataransacvec[oi]
        imggray = dataransac.visualizeFullObjects(gs.img.copy())
        # imggray = cv2.putText(imggray, "C = %d, logNFA = %.2f, %d out of %d  " % (dataransac.n_clusters, dataransac.logNFA, len(dataransac.H_listconsensus), len(dataransac.ros)), org, cv2.FONT_HERSHEY_SIMPLEX ,  
        #                1.0, (0, 0, 255) , 2, cv2.LINE_AA) 
        imggray = cv2.putText(imggray, "%.2f" % (dataransac.logNFA), org, cv2.FONT_HERSHEY_SIMPLEX ,  
                    1.0, (0, 0, 255) , 2, cv2.LINE_AA) 
        cv2.imwrite("output%d.png"%i , imggray )              


    for i in range(len(ind),num2show):
        imggray = np.repeat(np.expand_dims(np.sum(gs.img.copy(),axis=-1)/3.0, axis=-1), 3, axis=-1)        
        imggray = imggray.astype(np.uint8)    
        imggray = cv2.putText(imggray, 'No detection', org, cv2.FONT_HERSHEY_SIMPLEX ,  
                    1.0, (0, 0, 255) , 2, cv2.LINE_AA) 
        cv2.imwrite("output%d.png"%i, imggray )

if __name__ == "__main__":
   main(sys.argv[1:])