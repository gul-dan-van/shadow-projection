import os
import requests
import base64
from types import SimpleNamespace
from typing import Tuple
from time import time

import cv2
import numpy as np
from flask import Flask, request, render_template

from src.utils.config_manager import ConfigManager
from src.utils.reader import ImageReader
from src.composition.image_composition import ImageComposition


class MyApp:
    def __init__(self, config: SimpleNamespace) -> None:
        """
        Initialize the Flask application.

        Args:
            config: Configuration object containing application settings.
        """
        self.app = Flask(__name__, template_folder=os.path.join('./', 'src', 'templates'))
        self.config = config
        # Initializing Image Composition Models
        self.image_composer = ImageComposition(self.config)

        # Initialize routes
        self.app.add_url_rule("/cocreation/", view_func=self.index)
        self.app.add_url_rule("/cocreation/process_image",view_func=self.image_process, methods=["POST"])
        self.app.add_url_rule("/cocreation/process_composite",view_func=self.composite_process, methods=["POST"])
        self.app.add_url_rule("/cocreation/health", view_func=self.health_check, methods=["GET"])

    def index(self):
        """
        Render the main index page.

        Returns:
            str: Rendered HTML for index page.
        """
        return render_template("index.html")
    

    def image_process(self) -> None:

        try:
            start_time = time()
            # Process POST request
            image_reader = ImageReader(request)
        
            try:
                url_dict  =  image_reader.generate_urls("url_id1", "url_id2", "signed_url")

                background_image = image_reader.get_image_from_url(url_dict['url_id1'])
                foreground_image = image_reader.get_image_from_url(url_dict['url_id2'])
                composite_frame, composite_mask = simple_blend(foreground_image, background_image)

                final_image, _ = self.image_composer.process_composite(composite_frame, composite_mask, background_image)

                print(f"Time taken to process image: {time() -  start_time:2f}")
                
                output_url = url_dict['signed_url']
                send_time_start = time()
                send_image_to_gcp(final_image, output_url)
                print(f"Time taken to send image: {time() -  send_time_start:2f}")

            except ValueError as e:
                return 400

        except Exception as e:
            return 500

    def composite_process(self) -> str:
        """
        Process the image composition based on the images provided in the POST request.

        Returns:
            str: Rendered HTML for the result page displaying the composed image.
        """
        try:
            # Process POST request
            image_reader = ImageReader(request)

            try:
                composite_frame = image_reader.get_image_from_request("composite_frame")
                composite_mask = image_reader.get_image_from_request("composite_mask", grayscale=True)
                bg_image = image_reader.get_image_from_request("background_image")

            except ValueError as e:
                return (
                    render_template("error.html", error=f"Error processing image: {e}"),
                    400,
                )

            final_image, _ = self.image_composer.process_composite(composite_frame, composite_mask, bg_image)


            # Convert the final image to base64 for embedding in HTML
            _, final_image_encoded = cv2.imencode(".jpg", final_image)
            final_image_base64 = base64.b64encode(final_image_encoded).decode()

            return render_template("result.html", result=final_image_base64)

        except Exception as e:
            return (
                render_template("error.html", error=f"Internal server error: {e}"),
                500,
            )

    def health_check(self):
        """
        Health check endpoint.
        """
        return "OK"

    def run(self):
        """
        Run the Flask application.
        """
        if not self.config.debug_mode:
            self.app.run(debug=False, port=8000, host="0.0.0.0")
        else:
            self.app.run(debug=True, port=8000, host="0.0.0.0")


if __name__ == "__main__":
    config_manager = ConfigManager(default_config_flag=True)
    config = config_manager.get_config()
    my_app = MyApp(config)
    my_app.run()

def simple_blend(fg_image: np.ndarray, bg_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fg_mask = fg_image[:, :, 3]
    blended_image = np.copy(bg_image)
    blended_image[fg_mask != 0] = fg_image[fg_mask != 0]
    return blended_image, fg_mask


def send_image_to_gcp(image: np.ndarray, signed_url: str) -> bool:
    """ Uploads an image from a NumPy array to GCS using a pre-signed URL. """
    try:
        # Encode image array to bytes
        _, image_encoded = cv2.imencode(".jpg", image)
        image_bytes = image_encoded.tobytes()

        # Perform PUT request to upload the file
        response = requests.put(signed_url, data=image_bytes)
        response.raise_for_status()  # Raise an exception for bad status codes
        print("Image uploaded successfully.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error uploading image: {e}")
        return False
