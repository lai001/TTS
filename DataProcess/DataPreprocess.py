import numpy
import DataProcess
import concurrent.futures
import multiprocessing
import Util
import os
from tqdm import tqdm


class DataPreprocess:
    def __init__(self):
        # self.root_dir = os.path.dirname(os.path.dirname(__file__))
        self.root_dir = r"D:\Document"
        self.traning_data_dir = os.path.join(self.root_dir, 'TrainingData')
        self.metadata_file_path = os.path.join(self.traning_data_dir, 'metadata.csv')
        if not os.path.exists(self.traning_data_dir):
            os.mkdir(self.traning_data_dir)

        self.csv_raw_data = DataProcess.ReadCSV().start()
        self.audio_process = DataProcess.AudioProcess()
        self.bar = tqdm(total=len(self.csv_raw_data))

    def store_npy(self, csv_raw_datum: dict):
        audio_features = self.audio_process.extract_audio_features(csv_raw_datum['file_path'])
        basename = os.path.basename(csv_raw_datum['file_path'])

        linear_spectrogram_npy_file_path = os.path.join(self.traning_data_dir, f"linear-spectrogram-{basename}.npy")
        mel_spectrogram_npy_file_path = os.path.join(self.traning_data_dir, f"mel-spectrogram-{basename}.npy")
        if os.path.exists(linear_spectrogram_npy_file_path) or os.path.exists(mel_spectrogram_npy_file_path):
            print("已存在, 跳过")
            return
        numpy.save(linear_spectrogram_npy_file_path, audio_features['linear_scale_spectrogram'])
        numpy.save(mel_spectrogram_npy_file_path, audio_features['linear_scale_spectrogram'])

        with open(self.metadata_file_path, mode='a', encoding='utf-8') as writer:
            writer.write(f"{audio_features['frames']}|{linear_spectrogram_npy_file_path}|{mel_spectrogram_npy_file_path}|{csv_raw_datum['content']}\n")
            self.bar.update(1)

    def start(self):
        Util.logger.info(f"数据预处理中..., 保存在{self.traning_data_dir}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            tasks = {executor.submit(self.store_npy, csv_raw_datum): csv_raw_datum for csv_raw_datum in
                     self.csv_raw_data}

            iterator = concurrent.futures.as_completed(tasks).__iter__()

            try:
                while True:
                    iterator.__next__()
            except StopIteration:
                Util.logger.info(f"数据预完成..., 保存在{self.traning_data_dir}")
