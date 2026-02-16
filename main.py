import mediapipe as mp
import cv2
import detect as dt

DISPLAY = False

def main():
    cap = cv2.VideoCapture(0) # 0 : built-in camera
    while cap.isOpened():
        success, image = cap.read() # read 30 images per sec

        if not success:
            break
        
        # Load the input image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Detect pose landmarks from the input image
        detection_result = dt.DETECTOR.detect(mp_image)

        # Make key landmarks
        head, left_shoulder, right_shoulder = dt.key_landmarks_in_image(detection_result)

        # Process the detection result. In this case, visualize it
        if DISPLAY:
            annotated_image = dt.draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
            cv2.imshow("Display",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # If press "q" for 100ms : quit
        if cv2.waitKey(100) & 0xFF == ord('q'): # 0xFF : pure keyboard val (last 8 bits)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()