import numpy as np

class facesIdTracker():


    def __init__(self):

        #Parameters:
        self.minIouForMatch = 0.7 #this is the intersection of union threshold to match two rects.
        self.nHistoryPerFace = 50 #the maximal number of temprature measures for averaging per id
        self.forgetAfterNframes =2 #forget faces that didnt have a match in 2 frames
        ######################

        self.savedFacesRect = []
        self.savedFacesIds = []
        self.savedFacesLastSeenFrame = []
        self.tempDict = {}
        self.currentId = 1
        self.frameCounter = 0


    #top = box[0], bottom =  box[2] left= box[1] right =  box[3]
    def iou(self, box1, box2):
        top1 = box1[1]
        left1 = box1[0]
        bottom1 = box1[3]
        right1 = box1[2]

        top2 = box2[1]
        left2 = box2[0]
        bottom2 = box2[3]
        right2 = box2[2]


        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        area1 = h1*w1

        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
        area2 = h2 * w2

        maxTop   = max(top1   , top2   )
        minBtm   = min(bottom1, bottom2)
        maxLeft  = max(left1  , left2  )
        minRight = min(right1 , right2 )



        iw = minRight-maxLeft
        ih = minBtm-maxTop
        if (iw>0) and (ih>0):
            areaIntersect = iw*ih
            areaUnion = area1+area2 - areaIntersect
            return np.float(areaIntersect) / areaUnion
        else:
            return 0


    def giveFacesIds(self, faces, tempraturePerFace):
        # giveFacesIds function
        # Gets a list of bounding boxes of new faces
        # for each face, it returns an ID. if the bounding box is close enough to a detection in the previous frame then it returns the previous ID for this face. otherwise it gives a ned ID.
        # It also computes the mean temprature for each face in the last 50 frames

        # input:
        #   faces - a numpy array of rectangles of faces. size is N_Faces X 4.
        #   tempraturePerFace - a list of temprature per face (list or numpy array is ok)
        #
        # Output:
        #   ids = a list which contains an id for each input rectangle
        #   meanTemp = the mean temprature across frames for each id

        self.frameCounter = self.frameCounter+1

        facesIouMat = np.zeros((faces.shape[0], len(self.savedFacesRect)))
        newIds = np.zeros(faces.shape[0])


        if len(self.savedFacesRect) > 0:
            for i in range(faces.shape[0]):
                for j in range(len(self.savedFacesRect)):
                    facesIouMat[i][j] = self.iou(faces[i],self.savedFacesRect[j])




            for i in range(min(faces.shape[0], len(self.savedFacesRect))): #loop to find matches
                curMaxInd = np.argmax(facesIouMat)
                row = np.int(curMaxInd) / np.int(facesIouMat.shape[1])
                col = np.int(curMaxInd) % np.int(facesIouMat.shape[1])
                if (facesIouMat[row,col] > self.minIouForMatch):
                    newIds[row] = self.savedFacesIds[col]
                    self.savedFacesLastSeenFrame[col] = self.frameCounter
                    self.savedFacesRect[col] = faces[row]
                    self.tempDict[newIds[row]].append(tempraturePerFace[row])
                    if len ( self.tempDict[newIds[row]] ) > self.nHistoryPerFace:
                        self.tempDict[newIds[row]] = self.tempDict[newIds[row]][-self.nHistoryPerFace:]
                    facesIouMat[row,:] = -1
                    facesIouMat[:, col] = -1
                else:
                    break

            #add the new faces to savedFacesRect:
            for i in range(faces.shape[0]):
                if newIds[i]==0: # no match with previous
                    newIds[i] = self.currentId
                    self.currentId = self.currentId+1
                    self.savedFacesRect.append(faces[i])
                    self.savedFacesLastSeenFrame.append(self.frameCounter)
                    self.savedFacesIds.append(newIds[i])
                    self.tempDict[newIds[i]] = [tempraturePerFace[i]]



            # forget faces we havnt seen if 2 frames:
            keepIdx = []
            for i in range(len(self.savedFacesRect)):
                if self.frameCounter - self.savedFacesLastSeenFrame[i] < self.forgetAfterNframes:
                    keepIdx.append(i)
                else:
                    pass#just for debug

            self.savedFacesRect          = [self.savedFacesRect[i]          for i in keepIdx]
            self.savedFacesLastSeenFrame = [self.savedFacesLastSeenFrame[i] for i in keepIdx]
            self.savedFacesIds           = [self.savedFacesIds[i]           for i in keepIdx]
            self.tempDict = dict((k, self.tempDict[k]) for k in self.savedFacesIds)
            ###############################################3

        else:

            for i in range(faces.shape[0]):
                newIds[i] = self.currentId
                self.currentId = self.currentId + 1
                self.savedFacesRect.append(faces[i])
                self.savedFacesLastSeenFrame.append(self.frameCounter)
                self.savedFacesIds.append(newIds[i])
                self.tempDict[newIds[i]] = [tempraturePerFace[i]]

        meanTemp = np.zeros(len(newIds))
        for k in range(len(newIds)):
            meanTemp[k] = np.mean(np.array(self.tempDict[newIds[k]]))


        return newIds, meanTemp



#test code:
if __name__ == "__main__":
    facesTracker = facesIdTracker()
    frame1Faces = np.array([[10,20,30,40], [200,100,300,400] , [50,50,150,150]] )
    temp1 = [37.0 , 38, 36]
    frame2Faces = np.array([[10+5,20,30,40], np.array([200,100,300,400])-500 , np.array([50,50,150,150])+20] )
    temp2 = [38.0, 38, 37]

    frame3Faces = np.array([[10,20,30,40], [200,100,300,400] , np.array([50,50,150,150])+20] )
    temp3 = [40.0, 38, 37]

    ids1, meanTemp1 = facesTracker.giveFacesIds(frame1Faces, temp1)
    ids2, meanTemp2 = facesTracker.giveFacesIds(frame2Faces, temp2)
    ids3, meanTemp3 = facesTracker.giveFacesIds(frame3Faces, temp3)

    print ids1
    print ids2
    print ids3
    print meanTemp1
    print meanTemp2
    print meanTemp3

