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
    data_file = 'ljs'   # esc50, maestro, ljs
    # data_dir = '/work/r08922a13/datasets/ESC-50-master/split/test/tf_preprocess'
    # data_dir = '/work/r08922a13/datasets/LJSpeech-1.1/test/preprocess'
    # data_dir = '/work/r08922a13/datasets/maestro-v3.0.0/sr41k/test/preprocess'
    test_dir = f'/work/r08922a13/generative_inpainting/examples/{data_file}/test_m10'
    wave_dir = op.join(test_dir, './wave')
    wav_ext = '.wav'
    npy_ext = '.npy'
    mag_suffix = "-mag-raw-feats"
    ckpt = f'libs/mb_melgan/ckpt/{data_file}/generator-800000.h5'
    config = f'libs/mb_melgan/configs/multiband_melgan.{data_file}_v1.yaml'
    sr = 44100
    hop_size = 256
    image_width = 256
    length_5sec = int((sr / hop_size) * 5)
    seg_middle = round(length_5sec * 0.2 * 3.5)        # esc50: 0, others: 2.75
    seg_start = seg_middle - (image_width // 2)
    # seg_start = 0
    isMag = True


if __name__ == "__main__":

    # make output dir
    if op.isdir(hp.wave_dir) is False:
        os.mkdir(hp.wave_dir)

    if hp.data_file == 'esc50':
        hp.data_dir = '/work/r08922a13/datasets/ESC-50-master/split/test/tf_preprocess'
        hp.sr = 44100
        hp.seg_start = 0
    elif hp.data_file == 'maestro':
        hp.data_dir = '/work/r08922a13/datasets/maestro-v3.0.0/sr41k/test/preprocess'
    elif hp.data_file == 'ljs':
        hp.data_dir = '/work/r08922a13/datasets/LJSpeech-1.1/test/preprocess'
        hp.sr = 22050
        hp.length_5sec = int((hp.sr / hp.hop_size) * 5)
        hp.seg_middle = round(hp.length_5sec * 0.2 * 3.5)
        hp.seg_start = hp.seg_middle - (hp.image_width // 2)

    test_filenames = glob.glob(f'{hp.test_dir}/*.npy')
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
        basename = op.basename(test_file).split('.')[0]

        ori_filename = basename + hp.mag_suffix + hp.npy_ext
        ori_filename = op.join(hp.data_dir, ori_filename)
        complete_data = np.load(ori_filename)
        complete_data = complete_data[np.newaxis, :, :, np.newaxis]
        if complete_data.shape[2] > hp.length_5sec:
            complete_data = complete_data[:, :, :hp.length_5sec, :]
        complete_data[0, :, hp.seg_start:hp.seg_start+hp.image_width, :] = test_data
        mel = mag_to_mel(complete_data, hp.sr)
        subbands = mb_melgan(mel, training=False)
        audios = pqmf.synthesis(subbands)

        wav_name = op.join(hp.wave_dir, basename) + hp.wav_ext
        wav = tf.audio.encode_wav(audios[0], hp.sr)
        tf.io.write_file(wav_name, wav)
