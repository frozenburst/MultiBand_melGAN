import sys
sys.path.append(".")

from tqdm import tqdm

import tensorflow as tf
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
    output_pth = './predicts'
    ckpt = 'libs/mb_melgan/ckpt/ljs/generator-800000.h5'
    config = 'libs/mb_melgan/configs/multiband_melgan.ljs_v1.yaml'
    sr = 22050
    hop_size = 256
    length_5sec = int((sr / hop_size) * 5)
    isMag = True


if __name__ == "__main__":

    filename = 'LJ007-0173-mag-raw-feats.npy'
    root = '/work/r08922a13/datasets/LJSpeech-1.1/test/preprocess'
    file_pth = op.join(root, filename)
    test_data = np.load(file_pth)
    print(test_data.shape)

    if hp.isMag:
        test_data = mag_to_mel(test_data, hp.sr)
    else:
        test_data = test_data[tf.newaxis, :, :]

    mel_lengths = []
    mel_lengths.append(test_data.shape[1])

    # [T] -> [B, T]
    # test_data = test_data[np.newaxis, :, :]
    print(test_data.shape)

    # trim with custom
    # test_data = test_data[:, 100:100+hp.length_5sec, :]


    if not op.exists(hp.output_pth):
        os.mkdir(hp.output_pth)

    with open(hp.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    mb_melgan = TFMelGANGenerator(
        config=MultiBandMelGANGeneratorConfig(**config["multiband_melgan_generator_params"]),
        name="multiband_melgan_generator",
    )
    mb_melgan._build()
    mb_melgan.load_weights(hp.ckpt)

    pqmf = TFPQMF(
        config=MultiBandMelGANGeneratorConfig(**config["multiband_melgan_generator_params"]), name="pqmf"
    )

    generated_subbands = mb_melgan(test_data)
    generated_audios = pqmf.synthesis(generated_subbands)

    generated_audios = generated_audios.numpy()

    for i, audio in enumerate(generated_audios):
        sf.write(
            op.join(hp.output_pth, '2-117625-A-10_mbmelgan.wav'),
            audio[: mel_lengths[i] * config["hop_size"]],
            config["sampling_rate"],
            "PCM_16",
        )
