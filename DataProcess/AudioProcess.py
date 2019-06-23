import librosa
import numpy
import scipy


class AudioHparams:
    # 音频参数:
    num_mels = 80
    num_freq = 1025
    sample_rate = 20000
    # sample_rate=44100
    frame_length_ms = 50
    frame_shift_ms = 12.5
    preemphasis = 0.97
    min_level_db = -100
    ref_level_db = 20


class AudioProcess:
    def __init__(self):
        pass

    def sfft(self, wav: numpy.ndarray) -> numpy.ndarray:
        """
        短时傅里叶变化

        :param wav:
        :return:
        """
        preemphasis_wav = scipy.signal.lfilter([1, - AudioHparams.preemphasis], [1], wav)  # 预加重

        n_fft = (AudioHparams.num_freq - 1) * 2  # 计算离散傅里叶变换的点数
        hop_length = int(AudioHparams.frame_shift_ms / 1000 * AudioHparams.sample_rate)  # STFT列之间的帧音频数
        window_length = int(AudioHparams.frame_length_ms / 1000 * AudioHparams.sample_rate)  # 窗函数

        stft_matrix = librosa.stft(y=preemphasis_wav, n_fft=n_fft, hop_length=hop_length,
                                   win_length=window_length)  # 短时傅里叶变化后的矩阵
        return stft_matrix

    def linear_to_mel(self, linear_scale_spectrogram) -> numpy.ndarray:
        """
        线性标度频谱图 转为 梅尔谱图

        :param linear_scale_spectrogram:
        :return:
        """
        n_fft = (AudioHparams.num_freq - 1) * 2
        mel_basis = librosa.filters.mel(AudioHparams.sample_rate, n_fft, n_mels=AudioHparams.num_mels)
        return numpy.dot(mel_basis, linear_scale_spectrogram)

    def extract_audio_features(self, file_path: str) -> dict:
        """
        - 提取音频特征
        dict key:
            1. frames
            2. linear_scale_spectrogram
            3. mel_spectrogram

        :param file_path:
        :return:
        """
        wav = self.read(file_path)
        stft_matrix = numpy.abs(self.sfft(wav))

        unnormalize_linear_spectrogram = 20 * numpy.log10(
            numpy.maximum(1e-5, stft_matrix)) - AudioHparams.ref_level_db  # 输入信号的短时傅里叶变换

        linear_scale_spectrogram = numpy.clip(
            (unnormalize_linear_spectrogram - AudioHparams.min_level_db) / -AudioHparams.min_level_db,
            0,
            1).astype(numpy.float32)  # 归一化,获得线性标度频谱图

        frames = linear_scale_spectrogram.shape[1]  # 帧

        unnormalize_mel_S = 20 * numpy.log10(
            numpy.maximum(1e-5, self.linear_to_mel(stft_matrix))) - AudioHparams.ref_level_db  # 输入信号的短时傅里叶变换

        mel_spectrogram = numpy.clip((unnormalize_mel_S - AudioHparams.min_level_db) / -AudioHparams.min_level_db, 0,
                                     1).astype(numpy.float32)  # 归一化，获得梅尔谱图

        return {"frames": frames, "linear_scale_spectrogram": linear_scale_spectrogram,
                "mel_spectrogram": mel_spectrogram}

    def read(self, file_path: str) -> numpy.ndarray:
        return librosa.core.load(file_path, sr=AudioHparams.sample_rate)[0]
