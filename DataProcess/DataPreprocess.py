import numpy
import concurrent.futures
import multiprocessing
import os
import tensorflow
import threading

from DataProcess.AudioProcess import AudioProcess
from TextProcess.TextProcessExecutor import TextProcessExecutor
from Util import logger
from tqdm import tqdm


class DataPreprocess:
    isTFRecord = True

    def __init__(self):
        self.LJSpeech_metadata_file_path = r"C:\Users\lmc\Downloads\LJSpeech-1.1\metadata.csv"
        self.LJSpeech_wavs_dir_path = r"C:\Users\lmc\Downloads\LJSpeech-1.1\wavs"

        self.traning_data_dir = os.path.join(r"D:\Document", 'TrainingData')
        self.metadata_file_path = os.path.join(self.traning_data_dir, 'metadata.csv')

        self.record_file_path = os.path.join(self.traning_data_dir, 'train.tfrecord')

        if not os.path.exists(self.traning_data_dir):
            os.mkdir(self.traning_data_dir)

        self.csv_raw_data = self.read_metadata()
        self.audio_process = AudioProcess()
        self.text_process_executor = TextProcessExecutor()
        self.bar = tqdm(total=len(self.csv_raw_data))
        self.lock = threading.Lock()

    def read_metadata(self):
        with open(self.LJSpeech_metadata_file_path, 'r', encoding='utf8') as f:
            lines = f.readlines()

        backslashes = '\n'
        return [
            {'file_path': f"{os.path.join(self.LJSpeech_wavs_dir_path, line.rstrip(backslashes).split('|')[0])}.wav",
             'content': line.rstrip('\n').split('|')[1]} for line in lines]

    def store_npy(self, csv_raw_datum: dict):
        audio_features = self.audio_process.extract_audio_features(csv_raw_datum['file_path'])
        basename = os.path.basename(csv_raw_datum['file_path'])

        linear_spectrogram_npy_file_path = os.path.join(self.traning_data_dir, f"linear-spectrogram-{basename}.npy")
        mel_spectrogram_npy_file_path = os.path.join(self.traning_data_dir, f"mel-spectrogram-{basename}.npy")
        if os.path.exists(linear_spectrogram_npy_file_path) or os.path.exists(mel_spectrogram_npy_file_path):
            print("已存在, 跳过")
            return
        numpy.save(linear_spectrogram_npy_file_path, audio_features['linear_scale_spectrogram'])
        numpy.save(mel_spectrogram_npy_file_path, audio_features['mel_spectrogram'])

        with open(self.metadata_file_path, mode='a', encoding='utf-8') as writer:
            writer.write(
                f"{audio_features['frames']}|{linear_spectrogram_npy_file_path}|{mel_spectrogram_npy_file_path}|{csv_raw_datum['content']}\n")
            self.bar.update(1)

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tensorflow.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tensorflow.train.Feature(bytes_list=tensorflow.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint list."""
        return tensorflow.train.Feature(int64_list=tensorflow.train.Int64List(value=value))

    def example(self, csv_raw_datum: dict, writer: tensorflow.io.TFRecordWriter):
        audio_features = self.audio_process.extract_audio_features(csv_raw_datum['file_path'])

        linear_scale_spectrogram = audio_features['linear_scale_spectrogram'].T  # type: numpy.ndarray
        mel_spectrogram = audio_features['mel_spectrogram'].T  # type: numpy.ndarray
        sequence_text = self.text_process_executor.sequence_text(csv_raw_datum['content'])

        feature = {
            'linear_scale_spectrogram': self._bytes_feature(linear_scale_spectrogram.tobytes()),
            'linear_scale_spectrogram_shape': self._int64_feature(linear_scale_spectrogram.shape),
            'mel_spectrogram': self._bytes_feature(mel_spectrogram.tobytes()),
            'mel_spectrogram_shape': self._int64_feature(mel_spectrogram.shape),
            'sequence_text': self._int64_feature(sequence_text),
        }
        example = tensorflow.train.Example(features=tensorflow.train.Features(feature=feature))
        self.lock.acquire()
        writer.write(example.SerializeToString())
        self.bar.update(1)
        self.lock.release()

    def start(self):
        if DataPreprocess.isTFRecord:
            with tensorflow.io.TFRecordWriter(self.record_file_path) as writer:
                logger.info(f"数据预处理中..., 保存在{self.record_file_path}")
                with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    tasks = {executor.submit(self.example, csv_raw_datum, writer): csv_raw_datum for csv_raw_datum in
                             self.csv_raw_data}
                    iterator = concurrent.futures.as_completed(tasks).__iter__()
                    try:
                        while True:
                            iterator.__next__()
                    except StopIteration:
                        logger.info(f"数据预处理完成..., 保存在{self.record_file_path}")

        else:
            logger.info(f"数据预处理中..., 保存在{self.traning_data_dir}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                tasks = {executor.submit(self.store_npy, csv_raw_datum): csv_raw_datum for csv_raw_datum in
                         self.csv_raw_data}

                iterator = concurrent.futures.as_completed(tasks).__iter__()

                try:
                    while True:
                        iterator.__next__()
                except StopIteration:
                    logger.info(f"数据预完成..., 保存在{self.traning_data_dir}")
