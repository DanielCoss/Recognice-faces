import numpy as np
import cv2
from imutils.video import FPS
import time


class Detector:
    def __init__(self, use_cuda = False):
        self.faceModel = cv2.dnn.readNetFromCaffe("models/res10_300x300_ssd_iter_140000.prototxt", 
                                                  caffeModel="models/res10_300x300_ssd_iter_140000.caffemodel")
        
        if use_cuda:
            self.faceModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.faceModel.setPreferableTarfet(cv2.dnn.DNN_TARGET_CUDA)
        
    def processImage(self, imgName):
        self.img = cv2.imread(imgName)
        (self.height, self.width) = self.img.shape[:2]
        
        self.processFrame()
        cv2.imshow('Process Image', self.img)
        cv2.waitKey(0)
    
    def processVideo(self, videoName):
        cap = cv2.VideoCapture(videoName)
        if(cap.isOpened() == False):
            print("Error opening video")
            return
        (sucess, self.img) = cap.read()
        (self.height, self.width) = self.img.shape[:2]
        
        fps = FPS().start()
        
        while sucess:
            self.processFrame()
            cv2.imshow("Process video", self.img)
            (sucess, self.img) = cap.read()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break;
            fps.update()
            
        
        fps.stop()
        print ("End of the video \n FPS: {:,2f}" .format(fps.fps()))
        cap.release()
        cv2.destroyAllWindows()
        
    def processCamera(self):
        cap = cv2.VideoCapture(0)
        (sucess, self.img) = cap.read()
        (self.height, self.width) = self.img.shape[:2]
        while sucess:
            self.processFrame()
            cv2.imshow("Process Camera", self.img)
            key = cv2.waitKey(1) & 0xFF
            (sucess, self.img) = cap.read()
            if key == ord("q"):
                break;

        cap.release()
        cv2.destroyAllWindows()
        
    def processFrame(self):
        
        start_time = time.time()
        
        blob = cv2.dnn.blobFromImage(self.img, 1.0, (300,300), (104.0, 177.0, 123.0), 
                                     swapRB = False,
                                     crop = False)
        self.faceModel.setInput(blob)
        predictions = self.faceModel.forward()
        for i in range(0,predictions.shape[2]):
            if predictions[0,0,i,2] > 0.35:
                bbox = predictions[0,0,i,3:7] * np.array([self.width, self.height, self.width, self.height])
                (xmin, ymin, xmax, ymax) = bbox.astype('int')
                cv2.rectangle(self.img, (xmin, ymin), (xmax,ymax), (0,0,255), 2)
                
        print("Frame: --- %s seconds ---" % (time.time() - start_time))
            
        