import os
import requests
import json
import base64
from types import SimpleNamespace
from typing import Tuple
from time import time

import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify

from src.utils.config_manager import ConfigManager
from src.utils.reader import ImageReader, resize_image, compress_image
from src.composition.image_composition import ImageComposition


def simple_blend(fg_image: np.ndarray, bg_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fg_mask = np.where(fg_image[:, :, 3] > 128, 255, 0)
    blended_image = np.copy(bg_image)
    blended_image[fg_mask != 0] = fg_image[fg_mask != 0]
    return blended_image.astype(np.uint8), fg_mask.astype(np.uint8)


def send_image_to_gcp(image: np.ndarray, signed_url: str) -> Tuple[str, str]:
    """ Uploads an image from a NumPy array to GCS using a pre-signed URL. """
    try:
        # Encode image array to bytes
        _, image_encoded = cv2.imencode(".png", image)
        image_bytes = image_encoded.tobytes()

        headers = {
            'Content-Type': 'image/png'
        }
        print(f"Signed URL Provided: {signed_url}")
        # Perform PUT request to upload the file
        response = requests.put(signed_url, data=image_bytes, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        print("Image uploaded successfully.")
        return response.status_code, "Image uploaded successfully."

    except requests.exceptions.RequestException as e:
        print(f"Error uploading image: {e}")
        return response.status_code, f"Error uploading image: {e}"

def send_process_confirmation(process_id: str) -> Tuple[str, str]:
    # Set environment variable for the environment (e.g., "prod" or "dev")
    env = os.getenv("APP_ENV", "dev")

    try:
        # Construct the URL based on the environment
        url = f"https://zingcam.{env}.flamapp.com/engagement-svc/api/v1/engagements/processed/{process_id}"

        data = {
            "status": "processed"
        }
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.status_code, "processed message sent successfully."
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return response.status_code, f"Error sending processed message: {e}"

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
        self.app.add_url_rule("/cocreation/info", view_func=self.get_info_data, methods=["GET"])
        self.app.add_url_rule("/cocreation/health", view_func=self.health_check, methods=["GET"])

    def index(self):
        """
        Render the main index page.

        Returns:
            str: Rendered HTML for index page.
        """
        return render_template("index.html")
    

    def image_process(self) -> str:
        try:
            start_time = time()

            # Process POST request
            image_reader = ImageReader(request)
        
            try:
                url_dict = dict(request.json.items())
                input_urls = url_dict['inputs']
                output_urls = url_dict['outputs']
                process_id = url_dict['process_id']
                background_image = resize_image(image_reader.get_image_from_url(input_urls[0]))
                foreground_image = resize_image(image_reader.get_image_from_url(input_urls[1]))

                # Ensure the images have the same dimensions
                if foreground_image.shape[:2] != background_image.shape[:2]:
                    target_size = (min(foreground_image.shape[1], background_image.shape[1]),
                                   min(foreground_image.shape[0], background_image.shape[0]))
                    foreground_image = cv2.resize(foreground_image, target_size)
                    background_image = cv2.resize(background_image, target_size)

                composite_frame, composite_mask = simple_blend(foreground_image, background_image)
            
            except ValueError as e:
                return ("400", str(e))

            # Send "200" response immediately after blending
            response = "200"

            final_image, _ = self.image_composer.process_composite(composite_frame, composite_mask, background_image.astype(np.uint8))
            print(f"Time taken to process image: {time() - start_time:2f}")
            final_image = compress_image(final_image)

            send_time_start = time()
            
            gcp_sent_status_code, message = send_image_to_gcp(final_image, output_urls[0])
            print(f"Time taken to send image: {time() - send_time_start:2f}")

            if str(gcp_sent_status_code) != '200':
                raise RuntimeError(message)

            processed_message_status_code, message = send_process_confirmation(process_id)

            if str(processed_message_status_code) != '200':
                raise RuntimeError(message)

            return response

        except Exception as e:
            return ("500", str(e))

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

    def get_info_data(self):
        """
        Return application information stored in a separate JSON file.

        Returns:
            JSON response: JSON data containing application information.
        """
        try:
            # Path to the JSON file containing the info data
            json_file_path = os.path.join('./', 'cocreation.json')

            # Open and read the JSON file
            with open(json_file_path, 'r') as json_file:
                info_data = json.load(json_file)

            # Return the JSON response
            return jsonify(info_data), 200

        except FileNotFoundError:
            return jsonify({"error": "Info data file not found."}), 404
        except json.JSONDecodeError:
            return jsonify({"error": "Error decoding JSON data."}), 500
        except Exception as e:
            return jsonify({"error": f"Internal server error: {e}"}), 500

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