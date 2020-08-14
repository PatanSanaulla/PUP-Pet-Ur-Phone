import requests
import cv2
import numpy as np

url = "http://10.0.0.207:8080/shot.jpg"

def generateSIFTKeyPts(img1, img2):
    BEST_MATCH_PERCENT = 0.1
    sift = cv2.ORB_create(500) #Generates 500 Max features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    keypts1, dscptr1 = sift.detectAndCompute(img1, None)
    keypts2, dscptr2 = sift.detectAndCompute(img2, None)

    matches = matcher.match(dscptr1, dscptr2, None)

    matches.sort(key=lambda x: x.distance, reverse=False) #sort the matches based on score
    bestPoints = int(len(matches) * BEST_MATCH_PERCENT)
    matches = matches[:bestPoints]

    imMatches = cv2.drawMatches(img1, keypts1, img2, keypts2, matches, None)
    #cv2.imwrite("matches.jpg", imMatches)
    #cv2.imshow("keypoints", imMatches)

    SIFTpts1 = np.zeros((len(matches), 2), dtype=np.float32)
    SIFTpts2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        SIFTpts1[i, :] = keypts1[match.queryIdx].pt
        SIFTpts2[i, :] = keypts2[match.trainIdx].pt

    return SIFTpts1, SIFTpts2




while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content),dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.waitKey(100)
    img_resp2 = requests.get(url)
    img_arr2 = np.array(bytearray(img_resp2.content),dtype=np.uint8)
    img2 = cv2.imdecode(img_arr2, -1)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("AndroidCam", img)
    cv2.imshow("AndroidCam2", img2)

    keypts1, keypts2 = generateSIFTKeyPts(img, img2)
    print(keypts1-keypts2)
    if((keypts1[0]-keypts2[0]).all(0)):
        print("no change")
    else:
        print("change")


    if cv2.waitKey(1) == 27:
       break
