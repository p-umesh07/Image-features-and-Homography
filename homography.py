import cv2
import random
import numpy as np

MIN_MATCH_COUNT = 10

img1 = cv2.imread("data/mountain1.jpg", 0)
img2 = cv2.imread("data/mountain2.jpg", 0)

img1_colour = cv2.imread("data/mountain1.jpg", 1)
img2_colour = cv2.imread("data/mountain2.jpg", 1)

sift = cv2.xfeatures2d.SIFT_create()

(kp1, des1) = sift.detectAndCompute(img1_colour, None)
(kp2, des2) = sift.detectAndCompute(img2_colour, None)

sift_img1 = cv2.drawKeypoints(img1_colour, kp1, outImage=np.array([]))
sift_img2 = cv2.drawKeypoints(img2_colour, kp2, outImage=np.array([]))

print("Image1 - Keypoints: {}, descriptors: {}".format(len(kp1), des1.shape))
print("Image2 - Keypoints: {}, descriptors: {}".format(len(kp2), des2.shape))

cv2.imwrite("Result/task1_sift1.jpg", sift_img1)
cv2.imwrite("Result/task1_sift2.jpg", sift_img2)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if (len(good) > MIN_MATCH_COUNT):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    matchesMask_inliers = (mask.ravel()==1).tolist()
    
    list_random_kp = []
    for i in range(0, 10):
        ran_number = random.randint(0, 244)
        list_random_kp.append(ran_number)
    
    
    for i in range(0,len(matchesMask_inliers)):
        if (i in list_random_kp):
            continue
        else:
            matchesMask_inliers[i]=False

h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)

def warpImages(img2, img1, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    
    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
    
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)
    
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    
    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1
    return output_img

warp_img = warpImages(img1_colour, img2_colour, H)
cv2.imwrite("Result/task1_pano.jpg", warp_img)

print("The homography matrix is")
print(H)

draw_params = dict(matchColor=(255, 0, 0),  # draw matches in blue color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw inliers and outliers
                   flags=2)

draw_params_inliers = dict(matchColor=(0, 0, 255),  # draw matches in red color
                           singlePointColor=None,
                           matchesMask=matchesMask_inliers,  # draw only inliers
                           flags=2)

img_knn_matches = cv2.drawMatches(img1_colour, kp1, img2_colour, kp2, good, None, **draw_params)
cv2.imwrite("Result/task1_matches_knn.jpg", img_knn_matches)

img_inliers_matches = cv2.drawMatches(img1_colour, kp1, img2_colour, kp2, good, None, **draw_params_inliers)
cv2.imwrite("Result/task1_matches.jpg", img_inliers_matches)
