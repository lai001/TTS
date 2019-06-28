import tensorflow
import os
import numpy


class DataFeeder():
    def __init__(self):
        pass

    def make_dataset(self, tfrecord_file: str):
        audio_feature_description = {
            'linear_scale_spectrogram': tensorflow.io.FixedLenFeature([], tensorflow.string),
            'linear_scale_spectrogram_shape': tensorflow.io.FixedLenFeature([2], tensorflow.int64),
            'mel_spectrogram': tensorflow.io.FixedLenFeature([], tensorflow.string),
            'mel_spectrogram_shape': tensorflow.io.FixedLenFeature([2], tensorflow.int64),
            'sequence_text': tensorflow.io.VarLenFeature(tensorflow.int64),
        }

        map_func = lambda example_proto: tensorflow.io.parse_single_example(example_proto,
                                                                            audio_feature_description)
        dataset = tensorflow.data.TFRecordDataset(tfrecord_file)
        dataset = dataset.map(map_func).repeat().shuffle(5)
        return dataset

    def recover_linear_scale_spectrogram(self, audio_features):
        linear_scale_spectrogram_bytes_array = audio_features['linear_scale_spectrogram'].numpy()
        linear_scale_spectrogram_shape = audio_features['linear_scale_spectrogram_shape'].numpy()
        linear_scale_spectrogram_flat = numpy.frombuffer(linear_scale_spectrogram_bytes_array,
                                                         dtype=numpy.float32)  # type: numpy.ndarray
        linear_scale_spectrogram = linear_scale_spectrogram_flat.reshape(
            linear_scale_spectrogram_shape)  # type: numpy.ndarray
        return linear_scale_spectrogram

    def recover_mel_spectrogram(self, audio_features):
        mel_spectrogram_bytes_array = audio_features['mel_spectrogram'].numpy()
        mel_spectrogram_shape = audio_features['mel_spectrogram_shape'].numpy()
        mel_spectrogram_flat = numpy.frombuffer(mel_spectrogram_bytes_array,
                                                dtype=numpy.float32)  # type: numpy.ndarray
        mel_spectrogram = mel_spectrogram_flat.reshape(
            mel_spectrogram_shape)  # type: numpy.ndarray
        return mel_spectrogram

    def recover_sequence_text(self, audio_features):
        return audio_features['sequence_text'].values.numpy()  # type: numpy.ndarray

    def start(self):
        tfrecord_file = r"D:\Document\TrainingData\train.tfrecord"
        if not os.path.exists(tfrecord_file):
            print(f"找不到{tfrecord_file}")
            return

        dataset = self.make_dataset(tfrecord_file)

        return dataset
