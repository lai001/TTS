from DataProcess.DataPreprocess import DataPreprocess
from DataFeeder import DataFeeder

if __name__ == '__main__':
    DataPreprocess().start()
    feeder = DataFeeder()
    dataset = DataFeeder().start()

    for audio_features in dataset:
        linear_scale_spectrogram = feeder.recover_linear_scale_spectrogram(audio_features)
        mel_spectrogram = feeder.recover_mel_spectrogram(audio_features)
        sequence_text = feeder.recover_sequence_text(audio_features)
        print(len(sequence_text))