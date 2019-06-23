import os
import Util
from tqdm import tqdm


class ReadCSV:
    def __init__(self):
        self.metadata_filepath = r"D:\Document\LJSpeech-1.1\metadata.csv"
        self.wavs_dirpath = r"D:\Document\LJSpeech-1.1\wavs"

    def start(self):
        """
        dict key:
            1. file_path
            2. content

        :return:
        """

        Util.logger.info(f"读取{self.metadata_filepath}...")
        self.raw_data = []
        with open(self.metadata_filepath, 'r', encoding='utf8') as f:
            lines = f.readlines()

        for line in tqdm(lines):
            rstrip_line = line.rstrip('\n').split('|')
            file_path = f'{os.path.join(self.wavs_dirpath, rstrip_line[0])}.wav'
            self.raw_data.append({'file_path': file_path, 'content': rstrip_line[1]})
        Util.logger.info(f"读取{self.metadata_filepath}完毕")
        return self.raw_data
