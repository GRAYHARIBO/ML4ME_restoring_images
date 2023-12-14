import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def sift_extractor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #sift = cv2.xfeatures2d.SIFT_create(nfeatures=5000)
    sift = cv2.SIFT_create(nfeatures = 5000)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def orb_extractor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

def brute_force_matching(descriptors1, descriptors2):
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    return matches

def flann_matching(descriptors1, descriptors2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.match(descriptors1, descriptors2)
    return matches

def apply_ransac_and_transform(img1, keypoints1, img2, keypoints2, matches):
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width, channels = img2.shape
    img1_transformed = cv2.warpPerspective(img1, H, (width, height))

    return img1_transformed, H, mask

def homography(target_img, ref_img, MASK_INDEX, save_dir, BBOX):
    print('Extracting Feature Points...')
    # Sampling keypoint descriptors
    # kp1, des1 = orb_extractor(target_img)
    # kp2, des2 = orb_extractor(ref_img)
    kp1, des1 = sift_extractor(target_img)
    kp2, des2 = sift_extractor(ref_img)

    print('Matching Feature Points...')
    # Extracting image matches
    # matches = brute_force_matching(des2, des1)
    matches = flann_matching(des2, des1)

    print('Applying Ransac and Transforming Image...')
    # Applying RANSAC and transforming image
    transformed_ref_img, homography_matrix, inliers_mask = apply_ransac_and_transform(ref_img, kp2, target_img, kp1, matches)

    print('Generating Fixed Image...')
    ##################################################################################
    # Find the area you want to erase
    erase_area = MASK_INDEX
    ##################################################################################

    # Erase people in the target image
    erased_target_img = target_img.copy()
    erased_target_img[MASK_INDEX] = [0, 0, 0]

    # Find the corresponding region in the transformed reference image
    corresponding_region = transformed_ref_img.copy()
    corresponding_region[~MASK_INDEX] = [0,0,0]

    # Fill the earsed part of the target image with the corresponding reference image
    reconstructed_target_img = erased_target_img.copy()
    reconstructed_target_img[MASK_INDEX] = corresponding_region[MASK_INDEX]

    # Convert BGR to RGB for matplotlib display
    img1_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    transformed_img_rgb = cv2.cvtColor(transformed_ref_img, cv2.COLOR_BGR2RGB)
    erased_img_rgb = cv2.cvtColor(erased_target_img, cv2.COLOR_BGR2RGB)
    corresponding_region_rgb = cv2.cvtColor(corresponding_region, cv2.COLOR_BGR2RGB)
    reconstructed_img_rgb = cv2.cvtColor(reconstructed_target_img, cv2.COLOR_BGR2RGB)

    # Display results using matplotlib
    fig, axs = plt.subplots(2, 3, figsize=(50, 20), tight_layout=True)
    axs[0, 0].imshow(img1_rgb)
    axs[0, 0].set_title('Original Target Image')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(img2_rgb)
    axs[0, 1].set_title('Original Reference Image')
    axs[0, 1].axis('off')
    axs[0, 2].imshow(transformed_img_rgb)
    axs[0, 2].set_title('Transformed Reference Image')
    axs[0, 2].axis('off')
    axs[1, 0].imshow(erased_img_rgb)
    axs[1, 0].set_title('Erased Target Image')
    axs[1, 0].axis('off')
    axs[1, 1].imshow(corresponding_region_rgb)
    axs[1, 1].set_title('Corresponding Reference Image')
    axs[1, 1].axis('off')
    axs[1, 2].imshow(reconstructed_img_rgb)
    axs[1, 2].set_title('Reconstructed Target Image')
    axs[1, 2].axis('off')

    fig.savefig(os.path.join(save_dir, 'final_img.jpg'))
    return reconstructed_img_rgb

