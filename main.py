import cv2
import detect as dt
import modules.feature_extractor as pr

DISPLAY = True
COLLECT = True

def main():
    cap = cv2.VideoCapture(0) # 0 : built-in camera
    while cap.isOpened():
        success, image = cap.read() # read 30 images per sec
        if not success:
            break
        
        detector = dt.PoseDetector(image)

        # Make key landmarks for preprocess of training
        if COLLECT == True:
            dots = detector.key_landmarks_in_image()
            # TODO : preprocess 과정을 통한 데이터 전처리 후 학습모델로 데이터 전달
            feature_extractor = pr.FeatureExtractor(dots)

        # Process the detection result. In this case, visualize it
        if DISPLAY:
            annotated_image = detector.draw_landmarks_on_image()
            cv2.imshow("Display",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # If press "q" for 100ms : quit
        if cv2.waitKey(100) & 0xFF == ord('q'): # 0xFF : pure keyboard val (last 8 bits)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()