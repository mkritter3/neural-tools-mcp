"""Test file to verify USES and INSTANTIATES relationship extraction and storage."""

class DatabaseConnection:
    """A database connection class."""
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def connect(self):
        return f"Connected to {self.host}:{self.port}"

class UserService:
    """Service for managing users."""
    def __init__(self):
        self.users = []

    def add_user(self, name):
        self.users.append(name)

def process_data(input_data, config_settings):
    """Process data with configuration.

    This function should create:
    - USES relationships for input_data and config_settings
    - INSTANTIATES relationships for DatabaseConnection and UserService
    """
    # Using variables (should create USES relationships)
    processed = input_data.strip().upper()
    timeout = config_settings.get('timeout', 30)

    # Instantiating classes (should create INSTANTIATES relationships)
    db = DatabaseConnection('localhost', 5432)
    user_service = UserService()

    # More variable usage
    connection_string = db.connect()
    user_service.add_user('test_user')

    return {
        'data': processed,
        'timeout': timeout,
        'connection': connection_string
    }

def helper_function(data_list):
    """Helper that uses variables and instantiates a class."""
    # USES relationship
    total = sum(data_list)
    average = total / len(data_list)

    # INSTANTIATES relationship
    service = UserService()

    return average, service