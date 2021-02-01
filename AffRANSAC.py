import cv2
import sys
sys.path.append(".")
from library import *
import random
# from libLocalDesc import *
from matplotlib import pyplot as plt
# plt.switch_backend('agg')
from sklearn import cluster, mixture, neighbors

inlier_thresholds = {'dist':10, 'lambda':np.inf, 'phi':np.pi/2, 'tilt':np.inf, 'psi':np.inf}


class RepData4ransac:
    def __init__(self,ros):
        y_pred = self.initialClusterPredictions(ros,iters=50)
        self.ros = ros
        self.n_clusters = ros[0].NumberOfCopies
        self.mainClusterId = 0
        self.copiedClustersIds = [i for i in range(self.n_clusters) if i!=self.mainClusterId]
        self.y_pred = y_pred
        self.genLabels()
        self.label_locations = [] # for each label (i.e. cluster) it stores its locations in y_pred        
        self.roKPI = [] # for each label (i.e. cluster) it stores the corresponding kplist indices bound to the Keypoints
        self.rosI = [] # for each RepeatedObject(RO) it stores the corresponding ros index
        self.labeledKPs = [] # for each label (i.e. cluster) it stores the corresponding Keypoints
        for y_pred_i in range(self.n_clusters):
            li_loc = [k for (k,l) in enumerate(y_pred) if l==y_pred_i]
            self.label_locations.append( li_loc )            
            cluster_ro_KPnodeId = np.mod(li_loc, self.n_clusters)
            cluster_rosIndices = np.int32( (li_loc - cluster_ro_KPnodeId)/self.n_clusters )
            self.roKPI.append( cluster_ro_KPnodeId )            
            self.labeledKPs.append( features_deepcopy([self.ros[cluster_rosIndices[i]].kplists[0][kpi] for (i,kpi) in enumerate(cluster_ro_KPnodeId)]) )
        self.rosI = cluster_rosIndices
        assert np.all( [len(self.labeledKPs[0])==len(vec) for vec in self.labeledKPs] )
        self.NfeasibleRO = len(self.labeledKPs[0])
        self.A_main2copied_list = []
        for copiedid in self.copiedClustersIds:
            afflist = [ self.ros[self.rosI[i]].getAffineMap(self.roKPI[self.mainClusterId][i], 
                                                            self.roKPI[copiedid][i]) for i in range(self.NfeasibleRO) ]
            self.A_main2copied_list.append( afflist )
        self.H_list = []
        self.H_listconsensus = []
        self.logNFA = np.inf
    
    @staticmethod
    def initialClusterPredictions(ros,clustermethod='voronoi',iters=100):        
        n_clusters = ros[0].NumberOfCopies
        persub_y_pred = [i for i in range(n_clusters)]        
        X = np.array([kp.pt for ro in ros for kp in ro.kplists[0]])        
        
        bestscore, best_y_pred = 0.0, []

        # searchindices = np.random.permutation(range(len(ros)))[0:iters]               
        searchindices = range(len(ros))[0:iters]               
        for idx in searchindices:            
            Xrand = X[idx*n_clusters:(idx+1)*n_clusters,:]
            if clustermethod=='gmm':
                algorithm = mixture.GaussianMixture(
                    n_components=n_clusters, covariance_type='full',
                    means_init = np.array(Xrand))
                algorithm.fit(X)
            elif clustermethod=='voronoi':
                algorithm = neighbors.KNeighborsClassifier(1)
                algorithm.fit(np.array(Xrand), persub_y_pred)                      
            
            y_pred = algorithm.predict(X)
            
            score = 0            
            for i in range(len(ros)):
                sub_y_pred = np.array(y_pred[i*n_clusters:(i+1)*n_clusters])
                if np.all(np.isin(persub_y_pred,sub_y_pred)):
                    score += 1
            if bestscore<score:
                bestscore = score
                best_y_pred = y_pred
        return best_y_pred

    def getMatchingProposalsFromMainCluster(self,copyId):
        mainkplist = self.labeledKPs[self.mainClusterId]
        copiedCluster_kplist = self.labeledKPs[self.copiedClustersIds[copyId]]
        return  mainkplist, copiedCluster_kplist, self.A_main2copied_list[copyId]
    
    def visualizeHomograhy(self,copyId,img):        
        tkplist1 = [self.labeledKPs[self.mainClusterId][i] for i in self.H_listconsensus]
        tkplist2 = [self.labeledKPs[copyId][i] for i in self.H_listconsensus]

        img2show = img.copy()           
        colorvec = np.random.rand(3)*255
        for kp in tkplist1:        
            img2show = cv2.circle(img2show, tuple(np.int32(np.round(kp.pt))), 9, colorvec, -1)#, lineType=0)                         
        colorvec = np.random.rand(3)*255
        for kp in tkplist2:        
            img2show = cv2.circle(img2show, tuple(np.int32(np.round(kp.pt))), 9, colorvec, -1)#, lineType=0)         
        Matches_all = [cv2.DMatch(i,i,1.0) for i in range(len(tkplist1))]
        colorvec = np.random.rand(3)*255
        assert len(tkplist1)==len(tkplist2)==len(Matches_all)
        for m in Matches_all:            
            img2show = cv2.line(img2show, tuple(np.int32(np.round(tkplist1[m.queryIdx].pt))), tuple(np.int32(np.round(tkplist2[m.trainIdx].pt))), colorvec, 2)                 

        h, w = img.shape[:2]
        warp_AID = cv2.warpPerspective(img.copy(), self.H_list[copyId],(w, h))
        warp_AID = warp_AID*0.5 + img.copy()*0.5
        cv2.imwrite('./temp/%d_panorama.png'%copyId,warp_AID.astype(np.uint8))
        cv2.imwrite("./temp/%d_copy.png"%copyId,img2show)

    def visualizeRepeatedObject(self,ro_id, img):
        imggray = np.repeat(np.expand_dims(np.sum(img,axis=-1)/3.0, axis=-1), 3, axis=-1)        
        imggray = imggray.astype(np.uint8)                
        colorvec = np.random.rand(3)*255
        kplist = self.retreiveRepeatedObject(ro_id)
        img2show = imggray
        for kp in kplist:        
            img2show = cv2.circle(img2show, tuple(np.int32(np.round(kp.pt))), 9, colorvec, -1)#, lineType=0)         
        plt.imshow(img2show)
        plt.xticks(())
        plt.yticks(())
        plt.show()

    def visualizeFullObjects(self,img, shownow=False):
        imggray = np.repeat(np.expand_dims(np.sum(img,axis=-1)/3.0, axis=-1), 3, axis=-1)        
        imggray = imggray.astype(np.uint8)                

        for n in range(self.n_clusters):
            colorvec = np.random.rand(3)*255
            kplist = [self.labeledKPs[n][i] for i in self.H_listconsensus]
            img2show = imggray
            for kp in kplist:        
                img2show = cv2.circle(img2show, tuple(np.int32(np.round(kp.pt))), 9, colorvec, -1)#, lineType=0)         
        if shownow:
            plt.imshow(img2show)
            plt.xticks(())
            plt.yticks(())
            plt.show()        
        return img2show

    def retreiveRepeatedObject(self,ro_id):
        ''' Retreive kepoints from repeated object stored
            at position ro_id in self.ros
        '''
        return [self.labeledKPs[n][ro_id] for n in range(self.n_clusters)]

    def genLabels(self):
        ''' Generate labels from feasible predictions
            last label means not to be used
        '''
        persub_y_pred = [i for i in range(self.n_clusters)]
        Nlabels = int(self.n_clusters + 1)
        for i in range(int(len(self.y_pred)/self.n_clusters)):
            sub_y_pred = np.array(self.y_pred[i*self.n_clusters:(i+1)*self.n_clusters])
            if np.all(np.isin(persub_y_pred,sub_y_pred)):
                # print(persub_y_pred,sub_y_pred)
                pass
            else:
                # print(persub_y_pred,sub_y_pred)
                self.y_pred[i*self.n_clusters:(i+1)*self.n_clusters] = Nlabels        






class NFAclass:
    def __init__(self,ImageArea,Ndata,Nsample=2, maxThresholds=inlier_thresholds):
        self.Nsample = Nsample
        self.Ndata = Ndata
        self.maxThresholds = maxThresholds
        self.logc_n = [self.log_n_choose_k(Ndata,k) for k in range(Ndata+1)]
        self.logc_k = [self.log_n_choose_k(k,Nsample) for k in range(Ndata+1)]                
        self.logconstant = np.log10( Ndata-Nsample )
        self.logalpha_base = np.log10( np.pi/(ImageArea) ) + np.log10( 0.5/np.pi )
        self.epsilon = 0.00000001

    @staticmethod
    def log_n_choose_k(n,k):
        if k>=n or k<=0:
            return 0.0
        if n-k<k:
            k = n-k
        r = 0.0
        for i in np.arange(1,k+1):#(int i = 1; i <= k; i++)
            r += np.log10(np.double(n-i+1))-np.log10(np.double(i))
        return r
        
    def compute_logNFA(self, inliersdata):
        # dist and angle thresholds obtaining k inliers
        NoN, k, dist, phi = inliersdata['NumOfNodes'], inliersdata['nInliers'], inliersdata['maxDist'], inliersdata['maxPhi']         
        logalpha = self.logalpha_base + 2.0*np.log10(dist + self.epsilon) 
        if phi>0:
            logalpha += np.log10(phi + self.epsilon)
        return self.logconstant + logalpha*(NoN-1)*(k-self.Nsample)+self.logc_n[k]+self.logc_k[k]
        

# nfaobj = NFAclass(800*600,100)
# inliersdata = {'nInliers': 25, 
#                'maxDist': 10,
#                'maxPhi': np.pi/6}
# print( nfaobj.compute_logNFA(inliersdata) )


def HomographyFit(X0, Y0=[], Aff=[]):
    ''' Fits an homography from coordinate correspondances and 
    if local affine info is present then we also use it to better fit
    the homography.
    Remarks:
        1. If there is no affine info (Aff=[]) then you need at least 4 correspondances.
        2. If you do provide affine infos you only need 2 correspondances.
    '''
    Affeqqs = True if len(Aff)>0 else False
    assert ( len(X0)==len(Aff) or Affeqqs==False ) and len(X0)>0
    n = len(X0)
    if len(X0[0])==3:
        Xi = np.array( [[X0[i][0]/X0[i][2], X0[i][1]/X0[i][2]]  for i in range(n)] )
    else:
        Xi = np.array( [[X0[i][0], X0[i][1]]  for i in range(n)] )
    
    if Affeqqs:
        eqN = 6
        Yi = np.array( [np.matmul(Aff[i][0:2,0:2],Xi[i]) + Aff[i][:,2] for i in range(n)] )
    else:
        eqN = 2
        if len(Y0[0])==3:
            Yi = np.array( [[Y0[i][0]/Y0[i][2], Y0[i][1]/Y0[i][2]]  for i in range(n)] )
        else:
            Yi = np.array( [[Y0[i][0], Y0[i][1]]  for i in range(n)] ) 

    A = np.zeros((eqN*n,9),dtype=np.float)
    for i in range(n):
        # Coordinates constraints
        j = eqN*i
        A[j,0] = Xi[i,0]
        A[j,1] = Xi[i,1]        
        A[j,2] = 1.0
        A[j,6] = - Yi[i,0] * Xi[i,0]
        A[j,7] = - Yi[i,0] * Xi[i,1]
        A[j,8] = - Yi[i,0]

        j = eqN*i + 1
        A[j,3] = Xi[i,0]
        A[j,4] = Xi[i,1]
        A[j,5] = 1.0
        A[j,6] = - Yi[i,1] * Xi[i,0]
        A[j,7] = - Yi[i,1] * Xi[i,1]
        A[j,8] = - Yi[i,1]
        
        if Affeqqs:
            AA = Aff[i][0:2,0:2]

            # Affine constraints
            j = eqN*i + 2
            A[j,0] = 1.0
            A[j,6] = - Yi[i,0] - AA[0,0] * Xi[i,0]
            A[j,7] = - AA[0,0] * Xi[i,1]
            A[j,8] = - AA[0,0]        

            j = eqN*i + 3
            A[j,1] = 1.0
            A[j,6] = - AA[0,1] * Xi[i,0]
            A[j,7] = - Yi[i,0] - AA[0,1] * Xi[i,1]
            A[j,8] = - AA[0,1]

            j = eqN*i + 4
            A[j,3] = 1.0
            A[j,6] = - Yi[i,1] - AA[1,0] * Xi[i,0]
            A[j,7] = - AA[1,0] * Xi[i,1]
            A[j,8] = - AA[1,0]

            j = eqN*i + 5
            A[j,4] = 1.0
            A[j,6] = - AA[1,1] * Xi[i,0]
            A[j,7] = - Yi[i,1] - AA[1,1] * Xi[i,1]
            A[j,8] = - AA[1,1]
        

    _, _, v = np.linalg.svd(A)
    h = np.reshape(v[8], (3, 3))
    if h.item(8)!=0.0:
        h = (1/h.item(8)) * h
    return h


def Look4Inliers(dataransac, Xi, Yi_list, H_list, Affnetdecomp=[],  thres = inlier_thresholds):
        goodM = []
        if not np.all([np.linalg.matrix_rank(H) == 3 for H in H_list]):
            return goodM, -1        
        affdiffthres = np.array( [thres['lambda'], thres['phi'], thres['tilt'], thres['psi']] )
        AvDist = 0
        Hinv_list = [ np.linalg.inv(H) for H in H_list ]
        vec1, vec2 = np.zeros(shape=(4,1),dtype=np.float), np.zeros(shape=(4,1),dtype=np.float)
        
        for i in range(len(Xi)): # number of matches
            thisdist = np.zeros(dataransac.n_clusters-1,dtype=np.float)
            x = Xi[i]
            x = np.array(x).reshape(3,1)
            for cc in range(dataransac.n_clusters-1):
                H = H_list[cc]   
                Hi = Hinv_list[cc]                             
                Hx = np.matmul(H, x)
                Hx = Hx/Hx[2]
                
                y = Yi_list[cc][i]
                y = np.array(y).reshape(3,1)
                Hiy = np.matmul(Hi, y)
                Hiy = Hiy/Hiy[2]

                vec1[0:2,0] = Hx[0:2,0]
                vec1[2:4,0] = x[0:2,0]
                vec2[0:2,0] = y[0:2,0]
                vec2[2:4,0] = Hiy[0:2,0]

                thisdist[cc] = cv2.norm(vec1,vec2)

            if len(Affnetdecomp)==0 and np.all(thisdist <= thres['dist']):
                goodM.append([i,thisdist])
                AvDist += thisdist
            elif len(Affnetdecomp)>0 and np.all(thisdist <= thres['dist']):
                willpass = True
                for cc in range(dataransac.n_clusters-1):
                    avecnet = Affnetdecomp[cc][i][0:4]
                    avec = affine_decomp(FirstOrderApprox_Homography(H_list[cc],Xi[i]), doAssert=False)[0:4]

                    Affdiff = [ avec[0]/avecnet[0] if avec[0]>avecnet[0] else avecnet[0]/avec[0], 
                                AngleDiff(avec[1],avecnet[1],InRad=True), 
                                avec[2]/avecnet[2] if avec[2]>avecnet[2] else avecnet[2]/avec[2] , 
                                AngleDiff(avec[3],avecnet[3],InRad=True) ]
                    if not (Affdiff<affdiffthres).all():
                        willpass = False
                if willpass:
                    goodM.append([i,thisdist])
                    AvDist += thisdist              
        if len(goodM)>0:                
            AvDist = AvDist/len(goodM)    
        else:
            AvDist = -1    
        return goodM, AvDist



def RepAff_RANSAC_H(dataransac, img, Niter= 1000, AffInfo = 0, thres = inlier_thresholds):
    '''
    AffInfo == 0 - RANSAC Vanilla
    AffInfo == 1 - Fit Homography to affine info + Classic Validation
    AffInfo == 2 - Fit Homography to affine info + Affine Validation
    '''        

    Yi_list, Aq2t_list, Aq2tdecomp = [], [], []
    for cc in range(dataransac.n_clusters-1):
        cvkeys1, cvkeys2, Aq2t = dataransac.getMatchingProposalsFromMainCluster(cc)                       
        Aq2t_list.append( Aq2t )
        Aq2tdecomp.append( [affine_decomp(Aq2t[n]) for n in range(len(cvkeys1))] )

        Xi = [np.array( kp.pt+tuple([1]) ) for kp in cvkeys1]                
        if False:    
            # update cvkeys2 with refined positions
            Yi = np.array( [np.matmul(Aq2t[n][0:2,0:2],Xi[n][0:2]/Xi[n][2]) + Aq2t[n][:,2] for n in range(len(cvkeys1))] )
            Yi_list.append( Yi )
            for n in range(len(cvkeys1)):  
                # print(cvkeys2[n].pt,tuple(Yi[n]))  
                cvkeys2[n].pt = tuple(Yi[n])
        else:    
            # keep cvkeys2 as it is
            Yi_list.append( [np.array( kp.pt+tuple([1]) ) for kp in cvkeys2] )

    
    # RANSAC
    bestH = []
    bestCount = 0
    bestMatches = []
    if dataransac.NfeasibleRO<4:
        return bestCount, bestH, bestMatches
    
    Ns = 2 if AffInfo>0 else 4
    h,w = img.shape[0:2]
    NFA = NFAclass(h*w, len(dataransac.ros), Nsample=Ns)

    for _ in range(Niter):
        m = -1*np.ones(Ns,np.int)
        for j in range(Ns):
            m1 = np.random.randint(0,len(cvkeys1))
            while m1 in m:
                m1 = np.random.randint(0,len(cvkeys1))
            m[j] = m1
        if AffInfo>0:
            # print('Affine Info', Ns)
            H_list = []
            for cc in range(dataransac.n_clusters-1):
                H_list.append( HomographyFit([Xi[mi] for mi in m], Aff=[Aq2t_list[cc][mi] for mi in m]) )
            if AffInfo==1:
                goodM, _ = Look4Inliers(dataransac, Xi, Yi_list, H_list, Affnetdecomp = [], thres=thres )
            elif AffInfo==2:
                goodM, _ = Look4Inliers(dataransac, Xi, Yi_list, H_list, Affnetdecomp = Aq2tdecomp, thres=thres )
        else:
            # print('No affine Info', Ns)
            H_list = []
            for cc in range(dataransac.n_clusters-1):
                # print('-----------------------')
                # print([Xi[mi] for mi in m])
                # print([Yi_list[cc][mi] for mi in m])
                H_list.append( HomographyFit([Xi[mi] for mi in m], Y0=[Yi_list[cc][mi] for mi in m]) )            
            goodM, _ = Look4Inliers(dataransac, Xi, Yi_list, H_list, Affnetdecomp = [], thres=thres )
        
        if len(goodM)>=4 and bestCount<len(goodM):
            bestCount = len(goodM)
            bestH = H_list
            bestMatches = goodM
    if bestCount==0:
        return 0, [],[]
    dataransac.H_list = H_list
    dataransac.H_listconsensus = [i for (i,ds) in bestMatches] 
    # dists = [ds for (i,ds) in bestMatches] 
    inliersdata = {'NumOfNodes':dataransac.n_clusters,
                    'nInliers': len(bestMatches),
                    'maxDist':inlier_thresholds['dist'], 
                    # 'maxDist':np.max(np.ravel(dists)),                    
                    'maxPhi':-1 if AffInfo<=1 else inlier_thresholds['phi']
                    }    
    dataransac.logNFA = NFA.compute_logNFA(inliersdata)    





def Aff_RANSAC_H(img1, cvkeys1, img2, cvkeys2, cvMatches, pxl_radius = 20, Niter= 1000, AffInfo = 0, thres = inlier_thresholds, Aq2t=None):
    '''
    AffInfo == 0 - RANSAC Vanilla
    AffInfo == 1 - Fit Homography to affine info + Classic Validation
    AffInfo == 2 - Fit Homography to affine info + Affine Validation
    '''        
        
    if Aq2t is None:
        return 0, np.zeros((3,3)), []
                        
    Affdecomp = [affine_decomp(Aq2t[n]) for n in range(len(cvMatches))]

    Xi = [np.array( cvkeys1[cvMatches[n].queryIdx].pt+tuple([1]) ) for n in range(len(cvMatches))]                
    if True:    
        # update cvkeys2 with refined positions
        Yi = np.array( [np.matmul(Aq2t[n][0:2,0:2],Xi[n][0:2]/Xi[n][2]) + Aq2t[n][:,2] for n in range(len(cvMatches))] )
        for n in range(len(cvMatches)):    
            cvkeys2[cvMatches[n].trainIdx].pt = tuple(Yi[n])
    else:    
        # keep cvkeys2 as it is
        Yi = [np.array( cvkeys2[cvMatches[n].trainIdx].pt+tuple([1]) ) for n in range(len(cvMatches))]        



    # RANSAC
    bestH = []
    bestCount = 0
    bestMatches = []
    if len(cvMatches)<4:
        return bestCount, bestH, bestMatches
    
    Ns = 2 if AffInfo>0 else 4
    for _ in range(Niter):
        m = -1*np.ones(Ns,np.int)
        for j in range(Ns):
            m1 = np.random.randint(0,len(cvMatches))
            while m1 in m:
                m1 = np.random.randint(0,len(cvMatches))
            m[j] = m1
        if AffInfo>0:
            # print('Affine Info', Ns)
            H = HomographyFit([Xi[mi] for mi in m], Aff=[Aq2t[mi] for mi in m])
            if AffInfo==1:
                goodM, _ = Look4Inliers(cvMatches,cvkeys1, cvkeys2, H, Affnetdecomp = [], thres=thres )
            elif AffInfo==2:
                goodM, _ = Look4Inliers(cvMatches,cvkeys1, cvkeys2, H, Affnetdecomp = Affdecomp, thres=thres )
        else:
            # print('No affine Info', Ns)
            H = HomographyFit([Xi[mi] for mi in m], Y0=[Yi[mi] for mi in m])
            goodM, _ = Look4Inliers(cvMatches,cvkeys1, cvkeys2, H, Affnetdecomp = [], thres=thres )
        
        if bestCount<len(goodM):
            bestCount = len(goodM)
            bestH = H
            bestMatches = goodM
    return  bestCount, bestH, bestMatches