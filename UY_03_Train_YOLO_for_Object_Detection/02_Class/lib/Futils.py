from dotenv import load_dotenv
import os

class Futils:
    def __init__(self):
        pass
    @staticmethod
    def env(name, root=2):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_directory, '..', '..', '..')
        dotenv_path = os.path.join(project_root, '.env')
        load_dotenv(dotenv_path)
        return os.path.join(project_root, os.getenv(name))
