import base64
from types import SimpleNamespace

import cv2
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
        self.app = Flask(__name__, template_folder="src/templates/")
        self.config = config
        # Initializing Image Composition Models
        self.image_composer = ImageComposition(self.config)

        # Initialize routes
        self.app.add_url_rule("/cocreation/", view_func=self.index)
        self.app.add_url_rule("/cocreation/process_composite", view_func=self.process, methods=["POST"])
        self.app.add_url_rule("/cocreation/health", view_func=self.health_check, methods=["GET"])

    def index(self):
        """
        Render the main index page.

        Returns:
            str: Rendered HTML for index page.
        """
        return render_template("index.html")

    def process(self) -> str:
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
                composite_mask = image_reader.get_image_from_request(
                    "composite_mask", grayscale=True
                )
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
