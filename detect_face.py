import mediapipe as mp
import matplotlib.pyplot as plt
import math
import cv2
import imutils
from PIL import Image
import numpy as np
from typing import Tuple, Union, List

from passport_photo_specs import PassportPhotoSpecFactory, PassportPhotoSpec
from change_background import ChangeBackground


def max_img_size(width, height, target_ratio):
    """Calculates the maximum size possible conforming to the target aspect ratio"""

    current_ratio = width / height

    if current_ratio > target_ratio:
        new_width = int(target_ratio * height)
        new_height = height

    else:
        new_width = width
        new_height = int(width / target_ratio)

    if new_width > width:
        new_height = new_height * width / new_width
        new_width = width

    if new_height > height:
        new_width = new_width * height / new_height
        new_height = height

    new_width, new_height = (int(new_width), int(new_height))
    return new_width, new_height


class FaceDetector:
    def __init__(self, detector_model_path: str = None):
        # setting up face detecting algorithm
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a face detector instance with the image mode:
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=detector_model_path),
            running_mode=VisionRunningMode.IMAGE,
        )
        detector = FaceDetector.create_from_options(options)

        self._detector = detector

    def detect(self, image: mp.Image):
        return self._detector.detect(image)


class TransformImageToPassportSpecs:
    def __init__(
        self,
        passport_type: "str" = "Germany",
        detector_model_path: str = "/passport_photo/model_ckpts/blaze_face_short_range.tflite",
        background_changer_model_path: str = "/passport_photo/model_ckpts/u2net_rmbg.pth",
    ):
        self.detector = FaceDetector(detector_model_path)
        self.background_changer = ChangeBackground(background_changer_model_path)
        self.passport_spec: PassportPhotoSpec = PassportPhotoSpecFactory(passport_type)

    def transform(self, image: np.ndarray) -> Image.Image:
        """Transforms image to passport specs"""

        background_image = cv2.imread(
            f"/passport_photo/photos/backgrounds/{self.passport_spec.background_color}.png"
        )[:, :, ::-1]
        image = self.change_backgound(image, background_image)
        image = self.straghten_face(image)
        image = self.crop(image)
        image = self.match_aspect(image)
        return image

    def straghten_face(self, image: np.ndarray) -> np.ndarray:
        """Straightens the face by rotating the image till the eyes are horizontal"""

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detector_result = self.detector.detect(mp_image)

        # image =
        height, width, _ = image.shape
        keypoints_px = self.get_keypoints_px(detector_result, width, height)

        left_eye, right_eye, nose, mouth, left_ear, right_ear = keypoints_px
        dx = right_eye[0] - left_eye[0]
        dy = -(left_eye[1] - right_eye[1])
        alpha1 = math.degrees(math.atan2(dy, dx))

        dx = right_ear[0] - left_ear[0]
        dy = -(left_ear[1] - right_ear[1])
        alpha2 = math.degrees(math.atan2(dy, dx))

        dx = nose[0] - mouth[0]
        dy = -(nose[1] - mouth[1])
        alpha3 = math.degrees(math.atan2(dy, dx))
        alpha3 = 90 - alpha3

        alpha = (alpha3 + alpha2 + alpha3) / 3
        rotated_image = imutils.rotate(image, alpha)
        return rotated_image

    def crop(self, image: np.ndarray):
        """Crops the image to the given specifications"""

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detector_result = self.detector.detect(mp_image)

        height, width, _ = image.shape
        keypoints_px = self.get_keypoints_px(detector_result, width, height)

        left_eye, right_eye, nose = keypoints_px[0:3]
        bounding_box = detector_result.detections[0].bounding_box

        chin_y = bounding_box.origin_y + bounding_box.height
        eyes_y = (left_eye[1] + right_eye[1]) / 2
        chin_to_eyes = abs(chin_y - eyes_y)

        # chin_to_eyes ---> 16.1 in german passport
        factor = chin_to_eyes / self.passport_spec.chin_to_eyes

        # to update the algorithm to support passportphotospec factory
        y_start = chin_y - int(38 * factor)
        y_end = chin_y + int(7 * factor)

        x_start = nose[0] - int(35 / 2 * factor)
        x_end = nose[0] + int(35 / 2 * factor)

        # sanity check
        if y_start < 0:
            y_start = 0
        if x_start < 0:
            x_start = 0
        if y_end < 0:
            y_end = height
        if x_end < 0:
            x_end = width

        # crop image to required specs
        new_image = image[y_start:y_end, x_start:x_end]
        return new_image

    def change_backgound(
        self, image: np.ndarray, backgound_image: np.ndarray
    ) -> np.ndarray:
        """change backgound to the given background"""
        image = self.background_changer(image, backgound_image)
        return np.asarray(image)

    def match_aspect(self, image: np.ndarray) -> np.ndarray:
        # get the maximum size possible
        new_width, new_height = max_img_size(
            *image.shape[0:2], self.passport_spec.aspect
        )
        # return image.resize((new_width, new_height), Image.BILINEAR)
        return cv2.resize(image, (new_width, new_height), interpolation=2)

    def get_keypoints_px(
        self, detector_result, width: int, height: int
    ) -> List[Tuple[int, int]]:
        """Gets pixel coordinates of all keypoints"""
        keypoints_px = []
        for keypoint in detector_result.detections[0].keypoints:
            keypoint_px = self._normalized_to_pixel_coordinates(
                keypoint.x, keypoint.y, width, height
            )
            keypoints_px.append(keypoint_px)
        return keypoints_px

    def _normalized_to_pixel_coordinates(
        self,
        normalized_x: float,
        normalized_y: float,
        image_width: int,
        image_height: int,
    ) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""

        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px


if __name__ == "__main__":
    transformer = TransformImageToPassportSpecs("Germany")
    image = cv2.imread("/passport_photo/photos/child.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed_image = transformer.transform(image)
    cv2.imwrite(
        "/passport_photo/photos/child_transformed.png", transformed_image[:, :, ::-1]
    )
    # plt.imshow(transformed_image)
    # plt.show()
