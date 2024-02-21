# Class to manage uploaded files
import os

class FileManager:
    def __init__(self, directory):
        self.directory = directory

    def upload(self, file_path):
        # Assuming file_path is the absolute path of the file
        os.rename(file_path, os.path.join(self.directory, os.path.basename(file_path)))

    def list_files(self):
        return os.listdir(self.directory)

    def delete(self, file_name):
        os.remove(os.path.join(self.directory, file_name))

    def retrieve(self, file_name):
        with open(os.path.join(self.directory, file_name), 'r') as file:
            return file.read()