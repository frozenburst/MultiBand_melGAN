import sys
sys.path.append(".")

from tqdm import tqdm

import tensorflow as tf
import glob
import os.path as op
import os
import numpy as np
import soundfile as sf
import yaml

from libs.mb_melgan.configs.mb_melgan import MultiBandMelGANGeneratorConfig
from libs.mb_melgan.datasets.mel_dataset import MelDataset
from libs.mb_melgan.models.mb_melgan import TFPQMF, TFMelGANGenerator
from utils import mag_to_mel

print(tf.__version__)


class hp:
    # Training setting
    data_file = 'maestro'   # esc50, maestro, ljs
    # data_dir = '/work/r08922a13/datasets/ESC-50-master/split/test/tf_preprocess'
    # data_dir = '/work/r08922a13/datasets/LJSpeech-1.1/test/preprocess'
    # data_dir = '/work/r08922a13/datasets/maestro-v3.0.0/sr41k/test/preprocess'
    # The directory of test files (spectrogram).
    # test_dir = f'/work/r08922a13/audio_inpainting/test_logs/{data_file}_spadeNet_bs1_NoWeighted_vocol_m100/output'
    test_dir = '/work/r08922a13/similarity/results/maestro/maestro_10/preprocess/'
    wave_dir = op.join(test_dir, 'wave')
    wav_ext = '.wav'
    npy_ext = '.npy'
    mag_suffix = "-mag-raw-feats"
    ckpt = f'libs/mb_melgan/ckpt/{data_file}/generator-800000.h5'
    config = f'libs/mb_melgan/configs/multiband_melgan.{data_file}_v1.yaml'
    sr = 44100
    hop_size = 256
    image_width = 256
    length_5sec = int((sr / hop_size) * 5)
    isMag = True


if __name__ == "__main__":

    # make output dir
    if op.isdir(hp.wave_dir) is False:
        os.mkdir(hp.wave_dir)

    # test_filenames = glob.glob(f'{hp.test_dir}/*.npy')
    test_filenames = glob.glob(f'{hp.test_dir}/*rec-mag-raw-feats.npy')
    num_test_filenames = len(test_filenames)
    print(f'Number of test files is {num_test_filenames}.')

    with open(hp.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    mb_melgan = TFMelGANGenerator(
        config=MultiBandMelGANGeneratorConfig(**config["multiband_melgan_generator_params"]),
        name="multiband_melgan_generator",)
    mb_melgan._build()
    mb_melgan.load_weights(hp.ckpt)
    pqmf = TFPQMF(
        config=MultiBandMelGANGeneratorConfig(**config["multiband_melgan_generator_params"]), name="pqmf")

    for test_file in tqdm(test_filenames):
        test_data = np.load(test_file)
        if test_data.shape[1] > hp.length_5sec:
            test_data = test_data[:, :hp.length_5sec]
        if len(test_data.shape) == 2:
            test_data = test_data[np.newaxis, :, :, np.newaxis]
        basename = op.basename(test_file).split('_3_4_rec')[0]

        mel = mag_to_mel(test_data, hp.sr)
        subbands = mb_melgan(mel, training=False)
        audios = pqmf.synthesis(subbands)

        wav_name = op.join(hp.wave_dir, basename) + hp.wav_ext
        wav = tf.audio.encode_wav(audios[0], hp.sr)
        tf.io.write_file(wav_name, wav)
