# ML4ME_restoring_images (with F matrix)

<div align="center">
<img width="540" alt="ml4me" src="https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/e1ff728e-5105-4097-843c-fca9d0295c28">

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FGRAYHARIBO%2FML4ME_restoring_images&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

</div>

# Team 기머링
> **서울대학교 ML4ME 수업 기말 프로젝트** <br/> **개발기간: 2023.11 ~ 2023.12**
<img width="859" alt="스크린샷 2023-12-10 213731" src="https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/d38258e0-4dfc-4aee-acb8-71b0d9fd5004">

## 프로젝트 소개

### 다른 POV를 가진 사진 한 장으로 Target 이미지에서 원하지 않는 object를 제거하고 뒷 배경을 복원한다

이를 위해 우선 feature extraction을 진행한다
<img width="874" alt="스크린샷 2023-12-10 213617" src="https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/deeba06e-6b5e-44c2-89c6-498fc0dfcd11">
<img width="828" alt="스크린샷 2023-12-10 213627" src="https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/9ffd1934-a3ab-41b0-b28c-9166b86d2865">
'sift', 'orb', 'kaze', 'akaze' 총 4가지 방식을 사용하였다.

이를 통해 구한 interest points들로 Fundamental Matrix, F를 구한다.
<img width="803" alt="스크린샷 2023-12-10 213653" src="https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/7540841a-b222-417d-9627-d27adb93779b">

이후 reference 이미지 pixel들에 대해 아래의 연산을 각각 해준다.
1) pixel 주위(default=200 pix)에 있는 interest points를 찾는다
2) 이에 대해 linear assumption을 진행해준다. (SVD 사용)
3) Fundamental Matrix를 사용해 epipolar line을 그려주고 assumption 된 점을 line 위에 사영시킨다.
<img width="669" alt="스크린샷 2023-12-10 213710" src="https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/ff6cd668-e5bc-4d11-8d89-fc8546e6a734">

위와 같이 구한 pixel들을 target 이미지에 덧입힌다. 

## 시작 가이드
### Requirements
For building and running the application you need:

- [opencv-python 4.8.1.78](https://docs.opencv.org/4.x/)
- [scikit-image 0.22.0](https://scikit-image.org/)

### Installation
``` bash
$ git@github.com:GRAYHARIBO/ML4ME_restoring_images.git
```

---
## How to use

### ⭐️ Data preparation
- /ML4ME_dataset/set"폴더번호" 폴더 생성 ex)/ML4ME_dataset/set36
- 복원하고 싶은 이미지는 target.jpg
- 배경으로 참고할 이미지는 ref1.jpg으로 이름 변경해 /set"폴더번호" 폴더 내 저장

### ⭐️ When you run this code
- parser argument 설명
``` bash
$ python convert_with_F.py 25 2280,1830,2820,2450 sift 0.6
```
- n : 폴더 번호   ex) 3
- bd_yolo : 타겟 이미지에서의 boundary box, x1,y1,x2,y2   ex) 300,400,240,560
- method : feature extraction 방법 'sift', 'orb', 'kaze', 'akaze' 사용 가능   ex) sift
- reduce : 사진 화질을 몇배로 줄일건지 	ex) 0.3


---
## 실제 Result
![sift_F_result25](https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/2f3e0c7d-d561-4bbb-80c3-b3cedcae6aa9)

![sift_F_result1](https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/594e8ee7-ccdb-4c86-b498-b40bc5c93bd3)

![sift_F_result7](https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/d3d1b1c3-525f-4a86-83f6-5d804b2123ff)

![sift_F_result21](https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/16953032-7da9-4d98-97c8-93546aad3b59)
