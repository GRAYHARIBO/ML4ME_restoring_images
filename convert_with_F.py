import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from skimage.measure import block_reduce
import argparse


'''feature extraction'''
def extract(img1, img2, method = 'sift'):
    global m
    m = method
    if method == 'sift':
        sift = cv2.SIFT_create(nfeatures = 10000)
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
    
    elif method == 'orb':
        orb = cv2.ORB_create(nfeatures = 10000)
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)
    
    elif method == 'kaze':
        kaze = cv2.KAZE_create()
        kp1, des1 = kaze.detectAndCompute(img1,None)
        kp2, des2 = kaze.detectAndCompute(img2,None)
    
    elif method == 'akaze':
        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(img1,None)
        kp2, des2 = akaze.detectAndCompute(img2,None)

    if (des1.dtype != np.float32): des1 = des1.astype(np.float32) 
    if (des2.dtype != np.float32): des2 = des2.astype(np.float32) 

    return kp1, des1, kp2, des2


'''FLANN matching algorithm'''
def matching(kp1, kp2, des1, des2):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    
    return np.int32(pts1), np.int32(pts2)


def yh2ds(yh):
    #요한 : N*2 -> 동수 : 3*N
    N = np.shape(yh)[0]
    a = np.ones((1,N))
    
    ds = np.concatenate((yh.T, a), axis=0)

    return ds

def ds2yh(ds):
    #동수 : 3*N - > 요한 : N*2
    
    yh = ds.T
    yh = yh[:,:2]
    
    return yh    

def get_residual_distance(F, x0, x1):
    if np.shape(x0)[-1] == 2: #요한 타입
        x0 = yh2ds(x0)
        x1 = yh2ds(x1)
        
    # Epipolar lines
#     raise Exception(np.shape(F), np.shape(x1))
    l0 = F @ x1
    l1 = F.T @ x0

    # Normalize
    x0 = x0 / x0[2]
    x1 = x1 / x1[2]
    l0 = l0 / np.hypot(l0[0], l0[1])
    l1 = l1 / np.hypot(l1[0], l1[1])

    # Distance
    d0 = np.abs(np.sum(l0 * x0, axis=0))
    d1 = np.abs(np.sum(l1 * x1, axis=0))

    return d0, d1

def get_residual_error(F, x0, x1):
    if np.shape(x0)[-1] == 2: #요한 타입
        x0 = yh2ds(x0)
        x1 = yh2ds(x1)
        
    d0, d1 = get_residual_distance(F, x0, x1)
    return 0.5 * (np.mean(d0) + np.mean(d1))


def inpaint(img, d=10):
    # make noise mask
    h, w, _ = img.shape
    noise = np.zeros((h, w))
    noise[np.where(img[:,:,0] == 0)] = 1
    noise = noise.astype(np.uint8)
    print(img.shape)
    print(noise.shape)

    inpainted = cv2.inpaint(img, noise, d, cv2.INPAINT_TELEA)

    return inpainted, noise

def nan_pool(img, b_size = (2,2,1)):
    img = img.astype('float')
    img[img == 0] = np.nan

    pooled = block_reduce(img, block_size = b_size, func = np.nanmean)
    pooled = pooled.astype(np.uint8)

    return pooled


def draw(img1, img2, pts1, pts2, F, line = True, pt_sz=30, l_sz=5, ):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    
    r,c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    img1_ = img1.copy()
    img2_ = img2.copy()
    
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)

    for r1, r2, pt1, pt2 in zip(lines1,lines2,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())

        x0,y0 = map(int, [0, -r1[2]/r1[1]])
        x1,y1 = map(int, [c, -(r1[2]+r1[0]*c)/r1[1]])

        if (line == True) : img1 = cv2.line(img1, (x0,y0), (x1,y1), color, l_sz)
        img1 = cv2.circle(img1,tuple([int(pt1[0]), int(pt1[1])]),pt_sz,color,-1)

        x0,y0 = map(int, [0, -r2[2]/r2[1]])
        x1,y1 = map(int, [c, -(r2[2]+r2[0]*c)/r2[1]])

        if (line == True): img2 = cv2.line(img2, (x0,y0), (x1,y1), color, l_sz)
        img2 = cv2.circle(img2,tuple([int(pt2[0]), int(pt2[1])]),pt_sz,color,-1)
    
    alpha = 0.7 # Transparency
    img1 = cv2.addWeighted(img1, alpha, img1_, 1 - alpha, 0)
    img2 = cv2.addWeighted(img2, alpha, img2_, 1 - alpha, 0)

    ax1,_ = plt.subplot(121),plt.imshow(img1)
    ax1.title.set_text('extracted by '+m)
    ax2,_ = plt.subplot(122),plt.imshow(img2)
    ax2.title.set_text('extracted by '+m)
    plt.show()

    return img1, img2



def draw_only_points(img1, img2, point_a, point_b, pts1, pts2, pt_sz=30):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    
    a0 = int(round(point_a[0]))
    a1 = int(round(point_a[1]))
    b0 = int(round(point_b[0]))
    b1 = int(round(point_b[1]))
    
    point_a = [a0, a1]
    point_b = [b0, b1]
    
    r,c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    img1_ = img1.copy()
    img2_ = img2.copy()
    
    for pt1, pt2 in zip(pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())

        img1 = cv2.circle(img1,tuple([int(pt1[0]), int(pt1[1])]),pt_sz,color,-1)
        img2 = cv2.circle(img2,tuple([int(pt2[0]), int(pt2[1])]),pt_sz,color,-1)
    
    img1 = cv2.circle(img1,tuple([int(pt1[0]), int(pt1[1])]),pt_sz*2, (255, 0, 0),-1)
    img2 = cv2.circle(img2,tuple([int(pt2[0]), int(pt2[1])]),pt_sz*2, (255, 0, 0),-1)
    
    alpha = 0.7 # Transparency
    img1 = cv2.addWeighted(img1, alpha, img1_, 1 - alpha, 0)
    img2 = cv2.addWeighted(img2, alpha, img2_, 1 - alpha, 0)

    ax1,_ = plt.subplot(121),plt.imshow(img1)
    ax1.title.set_text('extracted by '+m)
    ax2,_ = plt.subplot(122),plt.imshow(img2)
    ax2.title.set_text('extracted by '+m)
    plt.show()
    
def find_ip(NP, pts1, pts2, n = 3):
    '''
    shape of interest points,  NP  : [u, v] 
    거리 기반!!!!!
    '''
    NP = np.asarray(NP)
    dist = []

    for i in range(pts1.shape[0]):
        d = np.linalg.norm(pts1[i,:] - NP)
        dist.append(d)

    dist = np.asarray(dist)
    ind = np.argsort(dist)[::-1]
    pts1 = pts1[ind[:],:]
    pts2 = pts2[ind[:],:]
    
    #1번 이미지의 NP와 가까운 pts1들과 그에 해당하는 이미지 2의 pts2들
    return pts1[:n,:], pts2[:n,:]

def find_ip_ver2(NP, pts1, pts2, len = 500):
    '''
    shape of interest points,  NP  : [u, v] 
    박스 기반!!!!!
    '''
    NP = np.asarray(NP)
    count = 0
    near_ip1 = []
    near_ip2 = []
    
    while True:
        for i in range(pts1.shape[0]):
            delta = pts1[i] - NP
            if (np.abs(delta[0])<len/2) and (np.abs(delta[1])<len/2):
                near_ip1.append(pts1[i])
                near_ip2.append(pts2[i])
                count += 1
        if count > 3:
            break
        else:
            count = 0
            near_ip1 = []
            near_ip2 = []
            len = len*1.5 #length 키우기
                
#     print("ip in box / count : ", count)
    
    #1번 이미지의 NP와 가까운 pts1들과 그에 해당하는 이미지 2의 pts2들
    return np.asarray(near_ip1), np.asarray(near_ip2)

def solve_non_linear(B):
    U,s,VT = np.linalg.svd(B)
    V = VT.T
    y = V[:,-1]
    y = y/y[-1]
#     print("y :",y)
    return y

def opt_weight(NP, near_ip):
    n, _ = near_ip.shape
    if n>=3:
        A = np.concatenate((np.asarray(near_ip), np.ones((n,1))), axis = 1)
        pt = -np.array([[NP[0], NP[1], 1]])
        B = np.concatenate((A, pt), axis = 0)
#         print(B.shape)
        weight = solve_non_linear(B.T)
    
    return weight[:-1]

def find_closeip_weight(NP, pts1, pts2, len = 500, use_all = 0):
    """
    이게 메인임!!!!!
    list 형식 NP를 주면 img2에서 해당되는 점, 가까운 ip들과 weight를 주는 함수
    """
    if use_all == 0: #박스 안에 있는 점 쓰기
        p1, p2 = find_ip_ver2(NP, pts1, pts2) #img1, img2에서의 ip
    else: #ip 전부 쓰기
        p1 = pts1
        p2 = pts2
#     print("p1,p2 :", p1,p2)
    coef = opt_weight(NP, p1)
#     print("p1, coef : ",p1, coef)

    # 원래 값 복원
    a = p1.T@coef.T #img1에서 NP 좌표
    b = p2.T@coef.T #img2에서 NP가 해당되는 점
#     print("a,b : ",a,b)
    
    return a, b, coef, p1, p2

def est_pt(NP1, NP2, F):
    """
    NP2를 F로 구한 직선으로 사영하는 함수
    """
    # image2의 NP 위치
    x = NP2[0]; y = NP2[1]
    NP1 = np.append(NP1,1)

    # line obtained from F mat: au + bv + c = 0, [a, b, c]
    line = F @ NP1.T
#     print(line)

    a = line[0]; b = line[1]; c = line[2]

    # perpendicular foot
    d = a**2 + b**2
    u = np.int32(  (b*(b*x - a*y) - a*c) / d)
    v = np.int32(  (-a*(b*x - a*y) - b*c) / d)

    return np.array([np.int32(u), np.int32(v), 1])

def img_F_transform(img1, pts1, pts2, F, bd_box=(0,100,0,100), l=600, u_all = 0):
    
    """
    이것도 메인임!!!
    img1의 pix을 img2의 꼴로 변환하는 것
    """
    
    x1, y1, x2, y2 = bd_box
    h, w, _ = img1.shape
    img_new = np.zeros_like(img1)
    in_count = 0
    count = 0 
    total_pix = (x2-x1)*(y2-y1)
    start = time.time()
    print("총 돌려야 되는 픽셀 수: ", total_pix)
    for r, row in enumerate(img1):
        for c, pix in enumerate(row):
            if ((x1 - c)*(x2-c)>0 or (y1-r)*(y2-r) >0): #image1의 범위(range로 정의된)에서 벗어남
                continue
            a, b, coef, p1, p2 = find_closeip_weight([c,r], pts1, pts2, len = l, use_all = u_all)            
            count += 1
            '''
            img 1 : NP = a, ip들 = p1
            img 2 : NP = b, ip들 = p2
            coef = alpha1, ... , alphaN
            '''
            if count==100:
                print("Estimated time: ", total_pix*(time.time()-start)//100)
            
            if count % (total_pix//10) == 0:
                print("지금까지 처리한 pixel 수 : ", count)
            b_new = est_pt(a, b, F)
            u,v,_ = b_new #새로운 사영점 x y
            
            if ((round(u))*(round(u)-w)<0 and (round(v)) * (round(v)-h) <0): #image2의 범위에 들어옴
                img_new[round(v)][round(u)][:] = pix
                in_count += 1
#             print("img2에 in_count : ", in_count)
    plt.imshow(img_new)
    plt.show()
    
    return img_new

def bd_box_tar2ref(image_ref, image_tar, reduce=0.3, meth = 'sift', bd_box_tar = [272, 484, 544, 847]):
        
    """
    yolo에서 target의 bd 받으면 
    """
    # print("reduce" , reduce)

    ## gray scale로 바꾸기
    img1 = cv2.imread(image_ref, cv2.IMREAD_GRAYSCALE)  #img ref
    img2 = cv2.imread(image_tar, cv2.IMREAD_GRAYSCALE) #img tar
    ## 화질 줄이기
    img1 = cv2.resize(img1, None, fx=reduce, fy=reduce)
    img2 = cv2.resize(img2, None, fx=reduce, fy=reduce)
    ## feature matching
    kp1, des1, kp2, des2 = extract(img1, img2, method = meth)
    pts1, pts2 = matching(kp1, kp2, des1, des2)

    
    bd_box_ref = []
    
    a, b, _, _, _ = find_closeip_weight([bd_box_tar[0], bd_box_tar[1]], pts2, pts1, len = 600, use_all = 1)
    bd_box_ref.append(int(round(b[0])))
    bd_box_ref.append(int(round(b[1])))
    c, d, _, _, _ = find_closeip_weight([bd_box_tar[2], bd_box_tar[3]], pts2, pts1, len = 600, use_all = 1)
    bd_box_ref.append(int(round(d[0])))
    bd_box_ref.append(int(round(d[1])))
    
    # bd box 크기 키우기
    w = int((bd_box_ref[2]-bd_box_ref[0])*0.4)
    h = int((bd_box_ref[3]-bd_box_ref[1])*0.4)
    
    bd_box_ref[0] = bd_box_ref[0]-w #x1
    bd_box_ref[1] = bd_box_ref[1]-h #y1
    bd_box_ref[2] = bd_box_ref[2]+w #x2
    bd_box_ref[3] = bd_box_ref[3]+h #y2
    
    if bd_box_ref[0]<0 :
        bd_box_ref[0] = 0
        
    if bd_box_ref[1]<0 :
        bd_box_ref[1] = 0
        
    if bd_box_ref[2]>img1.shape[1] :
        bd_box_ref[2] = img1.shape[1]
        
    if bd_box_ref[3]>img1.shape[0] :
        bd_box_ref[3] = img1.shape[0]
        
    print("img1.shape", img1.shape)    
    # print("a,c : ",a,c)
    # print("b,d : ",b,d)
    print("넣어준 bd_box_tar : ", bd_box_tar)
    print("bd_box_ref : ",bd_box_ref)
    
    return(bd_box_ref)

def get_proj_img(image_ref, image_tar, reduce=0.3, meth = 'sift', bd_box = (272, 484, 544, 847)):
    
    """
    이미지 2개 경로 ->
    
    변수명 :
    image_ref, image_tar = 파일 경로 ex)image_o = 'testdata/data2.jpg'
    reduce = resize 얼마나?
    bd_box = reference 이미지에서 어디 크롭?
    """

    ## gray scale로 바꾸기
    img1 = cv2.imread(image_ref, cv2.IMREAD_GRAYSCALE)  #img ref
    img1_rgb = cv2.imread(image_ref, cv2.IMREAD_COLOR) #img ref_rgb
    img2 = cv2.imread(image_tar, cv2.IMREAD_GRAYSCALE) #img tar
    img2_rgb = cv2.imread(image_tar, cv2.IMREAD_COLOR) #img tar_rgb

    ## 화질 줄이기
    img1 = cv2.resize(img1, None, fx=reduce, fy=reduce)
    img1_rgb = cv2.resize(img1_rgb, None, fx=reduce, fy=reduce)
    img2 = cv2.resize(img2, None, fx=reduce, fy=reduce)
    img2_rgb = cv2.resize(img2_rgb, None, fx=reduce, fy=reduce)

    ## feature matching
    kp1, des1, kp2, des2 = extract(img1, img2, method = meth)
    pts1, pts2 = matching(kp1, kp2, des1, des2)

    ####### F 구하기 ######
    ## 방법1. F with ransac = 동수
    # F, pts1, pts2, e = F_with_ransac(pts1, pts2, eps=10, n_iter=3000)

    ## 방법2. findFundamentalMat= cv = 주민
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    ####### Feature matching 된 거 plot하기
    # img1, img2, pts1, pts2, F, line = True, pt_sz=30, l_sz=5, 
    img_1, img_2 = draw(img1, img2, pts1, pts2, F, line = False, pt_sz=30)
    e = get_residual_error(F, pts1, pts2)
    print("residual_error : ", e," pix")
    print(pts1.shape)

    """
    FM_7POINT 
    FM_8POINT 
    FM_LMEDS 
    FM_RANSAC 
    """

    ####### 어디 crop 할건지 plot하기
    print("img1.shape : ",img1.shape)
    h, w = img1.shape
    bd = [];    db = []
    
    bd.append([bd_box[0], bd_box[1]])
    bd.append([bd_box[2], bd_box[1]])
    bd.append([bd_box[0], bd_box[3]])
    bd.append([bd_box[2], bd_box[3]])

    #print("bd : ", bd)
    for i in bd:
        a, b, weight, p1, p2 = find_closeip_weight(i, pts1, pts2, len = 600, use_all = 1)
        db.append([int(round(b[0])), int(round(b[1]))])
    #print("db ; ", db)

    draw_only_points(img1, img2, [1,1],[1,1], bd, db, pt_sz=30)


    ######## 사영 시키기 #######
    img_new = img_F_transform(img1_rgb, pts1, pts2, F, bd_box = bd_box, l=200, u_all = 0)
    
    img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(img1_rgb, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_rgb, cv2.COLOR_BGR2RGB)
    
    return img_new, img1_rgb, img2_rgb

def remove_scratch(img_tar, img_new1, p = 4, d = 10, bd_yolo=[500, 600, 800, 950]):
    """
    img_new1 = 변환시킨 녀석
    img_tar = 인생샷
    """
    img_pool = nan_pool(img_new1, (p,p,1))
    plt.imshow(img_pool)
    plt.show()

    img_inp, _ = inpaint(img_pool, d = d)
    plt.imshow(img_inp)
    plt.show()

    target_new = cv2.resize(img_tar, None, fx=1/p, fy=1/p)
    plt.imshow(target_new)
    plt.show()

    x1,y1,x2,y2 = np.asarray(bd_yolo)/p
#     print("x1,y1,x2,y2 : ", x1,y1,x2,y2)

    for r, row in enumerate(img_inp):
        for c, pix in enumerate(row):
            if ((c-x1)*(c-x2)<0) and ((r-y1)*(r-y2)<0): # 기존에 정해준 bd_yolo 안에 들어온 경우
                target_new[r][c] = pix 

    plt.imshow(target_new, cmap='gray')
    plt.show()
    
    return target_new

########################
###### main ############
########################

parser = argparse.ArgumentParser(description='이 프로그램의 설명(그 외 기타등등 아무거나)')    # 2. parser를 만든다.

# 3. parser.add_argument로 받아들일 인수를 추가해나간다.
parser.add_argument('n', help='폴더 번호 ex) 3')    # 필요한 인수를 추가
parser.add_argument('bd_yolo', help='boundary box : x1,y1,x2,y2 ex) 300,400,240,560')
parser.add_argument('method', help='method ex) sift')
parser.add_argument('reduce', help='사진 화질을 몇배로 줄일건지 ex) 0.3')

args = parser.parse_args()    # 4. 인수를 분석

n = int(args.n)
reduce = float(args.reduce)
bd = args.bd_yolo.split(',')
bd = np.asarray([int(bd[0]), int(bd[1]), int(bd[2]), int(bd[3])]) * reduce
bd = np.floor(bd).astype(int)
bd_yolo = bd.tolist()
method = args.method


print("n : ", n)
print("bd_yolo : ", bd_yolo)
print("method : ", method)
print("reduce : ", reduce,"\n\n")

"""
'sift'
'orb'
'kaze'
'akaze'

python convert_with_F.py 7 1900,2000,2970,3370 sift 0.3
"""
###### 위에 변수 채우기 ####

folder_path = 'ML4ME_dataset/set'+str(n)+'/'
image_tar = folder_path+'target.jpg'
image_ref1 = folder_path+'ref1.jpg'

asdf = cv2.imread(image_tar)
# plt.imshow(asdf)

bd_box_ref = bd_box_tar2ref(image_ref1, image_tar, reduce=reduce, meth = 'sift', bd_box_tar = bd_yolo)
img_new1,img_ref1, img_tar = get_proj_img(image_ref1, image_tar, reduce=reduce, meth = 'sift', bd_box = bd_box_ref)
# img_new1 = cv2.cvtColor(img_new1, cv2.COLOR_BGR2RGB)
plt.imshow(img_new1)
# img_tar = cv2.cvtColor(img_tar, cv2.COLOR_BGR2RGB)
plt.imshow(img_tar)

target_new=remove_scratch(img_tar, img_new1, p = 4, d = 10, bd_yolo=bd_yolo)

target_new.shape
plt.imshow(target_new)
plt.show()

cv2.imwrite('ML4ME_dataset/results/'+method+'_F_result'+str(n)+'.jpg', cv2.cvtColor(target_new, cv2.COLOR_RGB2BGR))
