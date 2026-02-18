import mediapipe as mp
import numpy as np
import os
import urllib.request
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
from mediapipe.framework.formats import landmark_pb2

class PoseDetector:
  def __init__(self, image):
    # Convert image format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    self.mp_iamge = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    # Model install
    model_asset_path = os.path.join("models", "pose_landmarker.task") # available in windows / linux
    if not os.path.exists(model_asset_path):
      model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
      urllib.request.urlretrieve(model_url, model_asset_path)

    # Create an PoseLandmarker object
    # Fallback to model_asset_buffer when absolute path contains non-ASCII characters (e.g. Korean)
    abs_model_path = os.path.abspath(model_asset_path)
    try:
      abs_model_path.encode("ascii")
      base_options = python.BaseOptions(model_asset_path=abs_model_path)
    except UnicodeEncodeError:
      with open(abs_model_path, "rb") as f:
        model_data = f.read()
      base_options = python.BaseOptions(model_asset_buffer=model_data)

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)

    # Detect pose landmarks from the input image
    detector = vision.PoseLandmarker.create_from_options(options)
    self.detection_result = detector.detect(self.mp_iamge)

  # Built-in function - draw landmarks
  def draw_landmarks_on_image(self):
    np_image = self.mp_iamge.numpy_view()
    pose_landmarks_list = self.detection_result.pose_landmarks
    annotated_image = np.copy(np_image)

  #   # Dot all points(33) and make line between dots
  #   pose_landmark_style = mp_drawing_styles.get_default_pose_landmarks_style() # Wet drawing style

  #   for pose_landmarks in pose_landmarks_list:
  #       ## All of these steps are intended to relieve differences between versions
  #       # 1. Convert list into landmark_pb2.NormalizedLandmarkList
  #       pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
  #       pose_landmarks_proto.landmark.extend([
  #           landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
  #           for landmark in pose_landmarks
  #       ])

  #       # 2. Deliver converted proto object to "landmark_list"
  #       mp_drawing.draw_landmarks(
  #           image=annotated_image,
  #           landmark_list=pose_landmarks_proto,
  #           connections=mp.solutions.pose.POSE_CONNECTIONS, # Connections info
  #           landmark_drawing_spec=pose_landmark_style,
  #           connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
  #       )
  #   return annotated_image

    # Extract (0: Nose, 2,5: Eyes(Left-Middle,Right-Middle), 11,12:Shoulder) 5 Dots in 33
    for pose_landmarks in pose_landmarks_list:
      # 1. Select indices
      target_indices = [0, 2, 5, 11, 12]
      
      # 2. Create Proto object
      pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      
      # 3. Check index in list then extend "Only essential dots"
      pose_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=pose_landmarks[i].x, 
                                          y=pose_landmarks[i].y, 
                                          z=pose_landmarks[i].z) 
          for i in target_indices # Direct access for essential dots (Not circulate)
      ])

      # 4. Draw dots
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=pose_landmarks_proto,
          connections=None # Do not draw line
      )
    return annotated_image

  # Using x, y, z data(Update : .visibility & .presence)
  def key_landmarks_in_image(self):
    if not self.detection_result.pose_landmarks:
        return None, None, None
    
    pose_landmarks_list = self.detection_result.pose_landmarks[0]
    eyes_mid_x = (pose_landmarks_list[2].x + pose_landmarks_list[5].x) / 2
    eyes_mid_y = (pose_landmarks_list[2].y + pose_landmarks_list[5].y) / 2
    eyes_mid_z = (pose_landmarks_list[2].z + pose_landmarks_list[5].z) / 2

    head_x = (eyes_mid_x + pose_landmarks_list[0].x) / 2
    head_y = (eyes_mid_y + pose_landmarks_list[0].y) / 2
    head_z = (eyes_mid_z + pose_landmarks_list[0].z) / 2
    
    head = [head_x, head_y, head_z]

    left_shoulder = [pose_landmarks_list[11].x, pose_landmarks_list[11].y, pose_landmarks_list[11].z]
    right_shoulder = [pose_landmarks_list[12].x, pose_landmarks_list[12].y, pose_landmarks_list[12].z]

    return head, left_shoulder, right_shoulder

if __name__ == "__main__":
    img_path = os.path.join("test", "minji.jpg")
    img = cv2.imread(img_path)
    cv2.imshow("IMG", img)
    detector = PoseDetector(img)
    annotated_image = detector.draw_landmarks_on_image()
    cv2.imshow("Display",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    