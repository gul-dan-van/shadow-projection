# Co-Creation
Co-Creation is an AI-powered Image Composition pipeline designed to seamlessly integrate your favorite stars and celebrities into your personal pics and videos. This tool leverages advanced machine learning techniques to provide realistic and high-quality image compositions.

## 1. Installation Steps

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


## 2. Deploying the Application
1. **Run the application**:
   ```bash
   python main.py
   ```
   This command starts the application, which allows you to select and place your favorite celebrities or characters into your personal pics and videos.

	**NOTE**: By default, the application runs in production mode. In config,env, set `DEBUG_MODE` to `TRUE` to run the application in 	debug mode.

2. **Running on Docker**:
   ```bash
   docker run -d -p 5000:5000 co-creation
   ```
   This command starts the application, which allows you to select and place your favorite celebrities or characters into your personal pics and videos.
   
   ```bash
   docker run --gpus all -d -p 5000:5000 co-creation
   ```
 This command allows to the docker container to access the GPU and increase performance of the application .
