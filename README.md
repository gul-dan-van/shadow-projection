# Co-Creation
Co-Creation is an AI-powered Image Composition pipeline designed to seamlessly integrate your favorite stars and celebrities into your personal pics and videos. This tool leverages advanced machine learning techniques to provide realistic and high-quality image compositions.

For more information on the project please go to the following [notion document](https://www.notion.so/flamapp/Co-creation-Blending-66b95b501d8f40468c1b7efb003e73f0).


# 1. Installation Steps

### Prerequisites
- Python 3.10 (as specified in the Dockerfile)
- pip (Python package installer)
- Conda (Package manager, optional)
- GPU with CUDA support (for GPU-based requirements)
The following Python packages are required for this project:

1. torch==1.12.0+cu116
2. torchvision==0.13.0+cu116
3. torchaudio==0.12.0+cu116

These packages are specified in the `requirements.txt` file and are also installed directly in the Dockerfile to ensure compatibility with CUDA-enabled GPU acceleration.

Additionally, other necessary Python packages listed in the `requirements.txt` file are installed to ensure the application runs correctly. These include various libraries that may be used for image processing, data handling, web frameworks, etc., depending on the specific needs of the project.


### Setting up the environment

#### Using pip
1. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```
   This command creates a new directory called `venv` where all the necessary executable files and libraries are stored.
2. **Activate the virtual environment**:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```
   Activating the environment ensures that all Python and pip commands apply to the virtual environment only, keeping dependencies required by different projects separate.

#### Using Conda
1. **Create a Conda environment**:
   ```bash
   conda create --name co-creation python=3.10
   ```
   This command sets up a new Conda environment named `co-creation` specifically for this project.
2. **Activate the Conda environment**:
   ```bash
   conda activate co-creation
   ```

#### Installing Dependencies
After setting up and activating your environment, install the following dependencies:
1. **Install GPU-based dependencies**:
   ```bash
   pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
   ```
   These packages are necessary for utilizing CUDA-enabled GPU acceleration.
2. **Install other required packages from** `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   This command installs all the necessary Python packages listed in the `requirements.txt` file to ensure the application runs correctly.

# 2. Testing the Application
To develop and run the unit tests for the Co-Creation project, follow these steps:

1. Navigate to the `tests` folder in the project directory.
2. Write unit tests using pytest to cover different scenarios and functionalities.
3. Ensure each test is independent and follows the Arrange-Act-Assert (AAA) pattern.
4. Document each test case following the PEP 8 format for better readability.
5. Utilize parametrization to run tests with different inputs.
6. Use fixtures for reusable setup and teardown operations.
7. Validate the expected behavior of the code with meaningful assertions.
8. Check the @README.md file for detailed guidelines on writing effective unit tests.

For more comprehensive instructions on generating and testing the Co-Creation codebase, refer to the [Testing Documentation](./tests/TESTING.md) file in the project's `test` directory.


### - Testing the API using Postman

#### 1. Setting Up Postman
- Download and install [Postman](https://www.postman.com/downloads/).
- Open Postman and create a new collection for organizing your API requests.

#### 2. Testing the Health Check Endpoint
- **Endpoint**: `/cocreation/health`
- **Method**: `GET`
- **URL**: `http://localhost:8000/cocreation/health`

##### Steps:
1. Open Postman and create a new request.
2. Set the request method to `GET`.
3. Enter the URL: `http://localhost:8000/cocreation/health`.
4. Click `Send`.
5. You should receive a response with the text `OK`.

#### 3. Testing the Image Processing Endpoint
- **Endpoint**: `/cocreation/process_image`
- **Method**: `POST`
- **URL**: `http://localhost:8000/cocreation/process_image`
- **Headers**: `Content-Type: application/json`
- **Body**: JSON

- **Sample JSON Body:**
   ```json
   {
   "url_id1": "https://example.com/background_image.png",
   "url_id2": "https://example.com/foreground_image.png",
   "signed_url": "https://example.com/upload_url"
   }
   ```

##### Steps:
1. Open Postman and create a new request.
2. Set the request method to `POST`.
3. Enter the URL: `http://localhost:8000/cocreation/process_image`.
4. Set the `Content-Type` header to `application/json`.
5. In the body, select `raw` and paste the sample JSON body.
6. Click `Send`.
7. You should receive a response with the status code `200` if the image processing and upload were successful.

#### 4. Testing the Composite Processing Endpoint
- **Endpoint**: `/cocreation/process_composite`
- **Method**: `POST`
- **URL**: `http://localhost:8000/cocreation/process_composite`
- **Headers**: `Content-Type: multipart/form-data`
- **Body**: Form-data

- **Form-data Fields**:
   - `composite_frame`: File (Upload the composite frame image)
   - `composite_mask`: File (Upload the composite mask image)
   - `background_image`: File (Upload the background image)

##### Steps:
1. Open Postman and create a new request.
2. Set the request method to `POST`.
3. Enter the URL: `http://localhost:8000/cocreation/process_composite`.
4. Set the `Content-Type` header to `multipart/form-data`.
5. In the body, select `form-data` and add the fields `composite_frame`, `composite_mask`, and `background_image` with their respective file uploads.
6. Click `Send`.
7. You should receive a response with the rendered


# 3. Deploying the Application
1. **Run the application**:
   ```bash
   python test.py
   ```
   This command starts the application, which allows you to select and place your favorite celebrities or characters into your personal pics and videos.

	**NOTE**: By default, the application runs in production mode. In config,env, set `DEBUG_MODE` to `TRUE` to run the application in 	debug mode.

2. **Running on Docker - CPU Version**:
   ```bash
   docker build --rm -p 8000:8000 -f .docker/Dockerfile -t cocreation-cpu:latest .
   ```
   This command builds the Docker image for the CPU version, which can be installed without the requirement of Nvidia GPUs.
   
   ```bash
   docker-compose -f .docker/docker-compose.yml up
   ```
   This compose command starts the application using the CPU version.

3. **Running on Docker - GPU Version**:
   ```bash
   docker build --rm -p 8000:8000 --gpus all -f .docker/Dockerfile_gpu.dockerfile -t co-creation:latest .
   ```
   This command builds the Docker image for the GPU version, which requires CUDA based GPUs to work.
   
   ```bash
   docker-compose -f .docker/docker-compose_gpu.yml up
   ```
   This compose command starts the application using the GPU version.


