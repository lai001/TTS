import logging
import os


class __Logger(object):

    def __init__(self):
        self.root_dir = os.path.dirname(os.path.dirname(__file__))
        self.log_dir_path = os.path.join(self.root_dir, 'Log')
        self.log_file_path = os.path.join(self.log_dir_path, '.log')

        if not os.path.exists(self.log_dir_path):
            os.mkdir(self.log_dir_path)
        self.__logger = logging.getLogger(self.log_file_path)
        self.__logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s: \n%(message)s\n')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(self.log_file_path, encoding='utf-8')
        file_handler.setFormatter(formatter)

        self.__logger.addHandler(stream_handler)
        self.__logger.addHandler(file_handler)

    def info(self, message: str):
        self.__logger.info(message)


shared = __Logger()

