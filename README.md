# ML4ME_restoring_images

<div align="center">
<img width="540" alt="ml4me" src="https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/e1ff728e-5105-4097-843c-fca9d0295c28">

# Team 기머링
> **서울대학교 ML4ME 수업 기말 프로젝트** <br/> **개발기간: 2023.11 ~ 2023.12**

## 프로젝트 소개

다른 POV를 가진 사진 한 장으로 Target 이미지에서 원하지 않는 object를 제거하고 뒷 배경을 복원한다

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
- 예시
``` bash
$ python convert_with_F.py 25 2280,1830,2820,2450 sift 0.6
```
- parser argument 설명
'n' : 폴더 번호   ex) 3
'bd_yolo' : 타겟 이미지에서의 boundary box, x1,y1,x2,y2   ex) 300,400,240,560
'method' : feature extraction 방법   ex) sift
	'sift', 'orb', 'kaze', 'akaze' 사용 가능
'reduce' : 사진 화질을 몇배로 줄일건지 	ex) 0.3
