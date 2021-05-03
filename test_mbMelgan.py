from tqdm import tqdm

import tensorflow as tf
import os.path as op
import os
import numpy as np
import soundfile as sf
import yaml

from configs.mb_melgan import MultiBandMelGANGeneratorConfig
from datasets.mel_dataset import MelDataset
from models.mb_melgan import TFPQMF, TFMelGANGenerator


print(tf.__version__)

class hp:
    # Training setting
    output_pth = './predicts'
    ckpt = 'ckpt/generator-800000.h5'
    config = 'configs/multiband_melgan.v1.yaml'


if __name__ == "__main__":

    filename = '2-117625-A-10-mel-raw-feats.npy'
    root = '/work/r08922a13/datasets/ESC-50-master/split/test/preprocess'
    file_pth = op.join(root, filename)
    test_data = np.load(file_pth)
    print(test_data.shape)
    mel_lengths = []
    mel_lengths.append(len(test_data))

    # [T] -> [B, T]
    test_data = test_data[np.newaxis, :, :]
    print(test_data.shape)


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
