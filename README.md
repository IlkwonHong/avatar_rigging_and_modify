# 시간 문제 
- torch maya hier에서 **rot이 오래 걸리는데** with torch no grad 켜놔서 오래 걸리나 싶기도
- 근데 하체 최적화는 시간이 짧게 걸리네
- 리깅 함수 내에서 시간 체크해보기
    - 상체는 조인트가 많아서 그런것 같기도
    - **상체를 팔 아래 부분은 없앤 DR용 클래스를 만드는것도 방법**

# 진행은?
- 일단 상하체 머리크기까지 하는걸 끝내고 해보자

# 고칠것들
- 몸통만 rasterize하려고 몸통 버텍스만 떼옴 -> 최적화 -> 실제보다 어깨가 좁게 나올듯 
    - 어깨만 스케일링 해주면 될듯?  