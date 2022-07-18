from Detector_face_class import *

start_time = time.time()

detector = Detector()
img_route = "people.jpg"
vid_route = 'test2.mp4'

# detector.processImage(img_route)

detector.processVideo(vid_route)

#detector.processCamera()

print("PROGRAM TOTAL: --- %s seconds ---" % (time.time() - start_time))