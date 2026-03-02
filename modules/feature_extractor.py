import numpy as np

class FeatureExtractor:
  def __init__(self):
    # A space to data for learning
    self.raw_data_history = []

  def analyze_forward_head_posture(self, dots_dict):
    # Transfer coordinate data into numpy data
    nose = np.array(dots_dict['nose'])
    l_eye = np.array(dots_dict['l_eye'])
    r_eye = np.array(dots_dict['r_eye'])
    l_shld = np.array(dots_dict['l_shld'])
    r_shld = np.array(dots_dict['r_shld'])

    # Standard points
    mid_shoulder = (l_shld + r_shld) / 2
    mid_eye = (l_eye + r_eye) / 2

    # Feature 1: Sagittal Offset
    sagittal_offset = nose[2] - mid_shoulder[2]

    # Feature 2: Craniovertebral Angle (CVA)
    dy = mid_shoulder[1] - nose[1]  # difference in height
    dz = mid_shoulder[2] - nose[2]  # difference in depth (possitive means : protuding nose)

    # Calculate degree by using inverse trigonometric function
    cva_angle = np.degrees(np.arctan2(dy, dz))

    # Feature 3: Head Tilt Index
    head_tilt = nose[1] - mid_eye[1]

    return {
    "sagittal_offset": sagittal_offset,
    "cva_angle_deg": cva_angle,
    "head_tilt_index": head_tilt
    }

  def extract_vector_from_dots(self, dots_dict, use_head_posture=True):
    """Form
    {
    "nose": [0.51, 0.32, -0.15],
    "l_eye": [0.53, 0.28, -0.12],
    "r_eye": [0.49, 0.28, -0.12],
    "l_shld": [0.65, 0.55, 0.05],
    "r_shld": [0.35, 0.55, 0.05]
    }
    """
    raw_coords = []
    for key in dots_dict.keys():
      raw_coords.extend(dots_dict[key])

    vector = np.array(raw_coords)

    if use_head_posture:
      posture_features = self.analyze_forward_head_posture(dots_dict)
      additional_features = np.array(list(posture_features.values()))
      vector = np.concatenate([vector, additional_features])

    self.raw_data_history.append(vector)
    return vector

  def extract_from_image(self, image_tensor):
    # Transform image into feature (CNN)
    # Resize, Normalization (/255.0)
    processed_img = image_tensor / 255.0
    return processed_img

if __name__ == "__main__":
  feature_extractor = FeatureExtractor()
  dot_vector = feature_extractor.extract_vector_from_dots({
  "nose": [0.51, 0.32, -0.15],
  "l_eye": [0.53, 0.28, -0.12],
  "r_eye": [0.49, 0.28, -0.12],
  "l_shld": [0.65, 0.55, 0.05],
  "r_shld": [0.35, 0.55, 0.05]
  })
  print(dot_vector)
