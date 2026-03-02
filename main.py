import cv2
import detect as dt
import modules.feature_extractor as pr
from collections import deque
import numpy as np

DISPLAY = True
COLLECT = True

def main():
    # Make buffer that makes filtered frames
    landmark_buffer = deque(maxlen=15)

    detector = dt.PoseDetector()

    cap = cv2.VideoCapture(0) # 0 : built-in camera
    while cap.isOpened():
        success, image = cap.read() # read 30 images per sec
        if not success:
            break
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detector(image_rgb)

        # Make key landmarks for preprocess of training
        if COLLECT == True:
            dots = detector.key_landmarks_in_image()

            if dots is not None:
                landmark_buffer.append(dots)

            if len(landmark_buffer) == 15:
                stable_dots = {}

                for key in landmark_buffer[0].keys():
                    # 2. Create an array (15, 3) by collecting only the coordinates of a specific part (key) from 15 frames
                    key_coords = np.array([frame[key] for frame in landmark_buffer])
                    
                    # 3. Find the median of the array (15, 3), make it into a (3,) array, and convert it into a list
                    stable_dots[key] = np.median(key_coords, axis=0).tolist()
            
                landmark_buffer.clear()

                # TODO : preprocess 과정을 통한 데이터 전처리 후 학습모델로 데이터 전달
                feature_extractor = pr.FeatureExtractor()
                pr_dots = feature_extractor.extract_vector_from_dots(stable_dots)
                print(pr_dots)

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