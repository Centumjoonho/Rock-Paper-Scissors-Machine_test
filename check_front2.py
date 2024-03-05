import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# OpenCV 캡처 객체 생성
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    success, img = cap.read()
    
    # MediaPipe 처리
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 피사체 리스트 초기화
    subjects = []
    
    # 각 랜드마크에 대한 처리
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
            subjects.append((x, y))
        
        # 피사체 크기 계산 및 최대 크기 피사체 찾기
        if len(subjects) > 1:  # 여러 사람이 있는 경우
            max_subject = max(subjects, key=lambda s: np.linalg.norm(np.array(s) - np.mean(subjects, axis=0)))
        else:  # 한 명만 있는 경우
            max_subject = subjects[0]
        
        # 최대 크기 피사체 강조 표시
        for subject in subjects:
            if subject == max_subject:
                cv2.circle(img, subject, 10, (0, 0, 255), -1)
                # 가장 큰 피사체의 중심점을 빨간색으로 표시
                cv2.putText(img, 'Focused', (subject[0]-50, subject[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(img, subject, 5, (0, 255, 0), -1)
    
    # 결과 프레임 출력
    cv2.imshow("Image", img)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
