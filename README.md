# ML4ME_restoring_images (with F matrix)
---

<div align="center">
<img width="540" alt="ml4me" src="https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/e1ff728e-5105-4097-843c-fca9d0295c28">

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FGRAYHARIBO%2FML4ME_restoring_images&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

</div>

---
## Project Introduction

Remove unwanted objects from the target image and restore the background using a different perspective photograph.

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

---
## QnA of our project

Q1.
I do have seen apps that do similar things. If you could go in to detail what kind of further benefits could be obtained from using the various methods mentioned in the video, it would make a much more robust problem definition!
A1.
As you can see from the comparison with existing applications (HAMA, SNAPEDIT)  based on deep learning, our model outperfroms them both quantitatively and qualitatively

Q2.
I think they need to develop algorithms to determine the number of main subjects and which ones are main subjects.
A1.
A code for GUI was implemented into the YOLO segmentation code allows for the selection of the subject to be removed for each image.

Q3.
방법론적으로는 흠잡을 것 없는 좋은 주제인 것 같습니다. 다만, SLAM 분야에 활용될 수 있다고는 하지만 기계공학분야와의 연관성이 잘 와닿지 않아서 관련 예시를 실제로 수행해보거나 인트로에서의 설명을 조금 더 보완하면 좋을 것 같습니다.
A3.
The model to remove displaced objects can be very useful in Visual SLAM, because it allows us to map only the stationary parts.

Q4.
많은 데이터셋이 필요할 것 같은데, 25개의 데이터만으로 학습이 잘 될지 궁금하다. 또한 사람에 의해 가려진 부분이 많다면 다른 사진에서 그 부분을 정확히 매칭하는것이 어려울 것 같은데 이 부분은 어떻게 해결할지 궁금하다.
A4.
Our model is not based on learning, therefore the number of images does not influence the performance of the model.
Rather, the performance is largely dependent on the number of features extractable from a single image.
The purpose of the 25 datasets was to evaluate the performace of the model under various conditions with varying number of features.

Q5.
From your dataset explanation, I am curious what is the difference between groundtruth image and target image. Does a target image mean the generated image from two reference images with your own algorithm? Or is your team planning to get a target image with existing algorithm or method for evaluation? I am also wondering if there are more reference images(input data), your algorithm performs better.
A5.
Target Image: Image taken with the obstacle
Ground Truth: Image taken in the same POV as Target Image, but without the obstacle
Reference Image: Image taken in a different POV so that the backgroud obscurred by the obstacle in Target Image is visible

---
## How to Run our code

<details>
<summary>Requirements</summary>
<div>
For building and running the application you need:

- [opencv-python 4.8.1.78](https://docs.opencv.org/4.x/)
- [scikit-image 0.22.0](https://scikit-image.org/)
</div>
</details>


<details>
<summary>Installation</summary>
<div>
``` bash
$ git@github.com:GRAYHARIBO/ML4ME_restoring_images.git
```
</div>
</details>


<details>
<summary>Data preparation</summary>
<div>
/ML4ME_dataset/set"폴더번호" 폴더 생성 ex)/ML4ME_dataset/set36
복원하고 싶은 이미지는 target.jpg
배경으로 참고할 이미지는 ref1.jpg으로 이름 변경해 /set"폴더번호" 폴더 내 저장
</div>
</details>


<details>
<summary>In Terminal</summary>
<div>
- parser argument example
``` bash
$ python convert_with_F.py n bd_yolo method reduce
$ python convert_with_F.py 3 300,400,240,560 sift 0.3
```

n : Folder number ex) 3
bd_yolo : Boundary box in target image : x1,y1,x2,y2 ex) 300,400,240,560
method : Feature extraction method ex) sift
scale : Image resize scale for calculation ex) 0.3
</div>
</details>

---
## Result
![슬라이드3](https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/37b0bc93-d1c1-465f-9b82-f8a8dc79622b)
![슬라이드2](https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/aa139c3a-d2fb-4d60-9669-dd6fe5eb6174)
![슬라이드1](https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/7a8f993a-73cd-46db-b994-53226abed4c6)
![슬라이드8](https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/b9f21943-f5db-4a5b-bb08-d3d22bd28b1d)
![슬라이드5](https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/e1a162ed-a507-4751-8d99-6ca5a27cd918)
![슬라이드6](https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/2e8d1a45-3bbe-4171-8e8a-07981dad2c1b)
![슬라이드7](https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/d8e46ae9-bffc-4b55-b84c-9624d9a33054)

---
## About our team
### Team 기머링
> **SNU ML4ME final project** <br/> **Development period: 2023.11 ~ 2023.12**
<img width="859" alt="스크린샷 2023-12-10 213731" src="https://github.com/GRAYHARIBO/ML4ME_restoring_images/assets/96507797/d38258e0-4dfc-4aee-acb8-71b0d9fd5004">

Please don't hesitate to reach out to us via the email provided below.   
josephjung24@snu.ac.kr

