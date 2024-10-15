import pytest
from os.path import join
from app import MyApp
from src.utils.config_manager import ConfigManager

from tests.helper import *

# Pre-read images to avoid multiple I/O operations
composite_input_data = {
    'composite_frame': read_image('input/composite/composite_frame.jpg'),
    'composite_mask': read_image('input/composite/composite_mask.jpg', 0),
    'background_image': read_image('input/composite/background_frame.jpg')
}

ENV_PATH='envs/config.env'
INDEX_PATH='/cocreation/'
API_URL_PATHS={
    'process_composite_frames': join(INDEX_PATH, 'process_composite')
}


@pytest.fixture(scope='class')
def client():
    """
    Pytest fixture to set up the test client for the Flask application.
    
    Yields:
        FlaskClient: A test client for the Flask application.
    """
    config_manager = ConfigManager(default_config_flag=True)
    config = config_manager.get_config()
    my_app = MyApp(config)
    with my_app.app.test_client() as client:
        yield client


class TestMyApp:
    """
    Test suite for the Flask application MyApp.
    """
    
    def test_index_route(self, client):
        """
        Test the index route for a successful response.
        
        Args:
            client (FlaskClient): The test client for the Flask application.
        """
        response = client.get(INDEX_PATH)
        assert response.status_code == 200
        assert b'index' in response.data

    def test_process_route(self, client):
        """
        Test the /process_composite route for a successful response with valid data.
        
        Args:
            client (FlaskClient): The test client for the Flask application.
        """
        data = prepare_multipart_data(composite_input_data)
        response = client.post(API_URL_PATHS['process_composite_frames'], content_type='multipart/form-data', data=data)
        assert response.status_code == 200
        assert b'Result' in response.data

    def test_index_route_error_handling(self, client, monkeypatch):
        """
        Test the index route error handling by mocking an exception.
        
        Args:
            client (FlaskClient): The test client for the Flask application.
            monkeypatch (MonkeyPatch): Pytest's monkeypatch fixture for dynamic attribute setting.
        """
        with monkeypatch.context() as m:
            m.setattr('app.MyApp.index', lambda: 1/0)
            with pytest.raises(Exception):
                response = client.get('/')
                assert response.status_code != 200
                assert b'Error' in response.data

    def test_process_route_invalid_request(self, client):
        """
        Test the /process_composite route with invalid data to ensure error handling.
        
        Args:
            client (FlaskClient): The test client for the Flask application.
        """
        data = {name: (b'', f"{name}.jpg") for name in composite_input_data.keys()}
        response = client.post(API_URL_PATHS['process_composite_frames'], content_type='multipart/form-data', data=data)
        assert response.status_code != 200
        assert b'Error' in response.data

    def test_process_route_valid_request(self, client):
        """
        Test the /process_composite route with valid data and mock processing.
        
        Args:
            client (FlaskClient): The test client for the Flask application.
        """
        data = prepare_multipart_data(composite_input_data)
        response = client.post(API_URL_PATHS['process_composite_frames'], content_type='multipart/form-data', data=data)
        assert response.status_code == 200
        assert b'Result' in response.data

    def test_health_check_route(self, client):
        """
        Test the health check route for a successful response.
        
        Args:
            client (FlaskClient): The test client for the Flask application.
        """
        response = client.get('/cocreation/health')
        assert response.status_code == 200
        assert response.data == b'OK'

    def test_health_check_route_negative(self, client):
        """
        Test the health check route for a negative response.
        
        Args:
            client (FlaskClient): The test client for the Flask application.
        """
        with pytest.raises(Exception):
            response = client.get('/cocreation/health')
            assert response.status_code != 200
            assert response.data != b'OK'
