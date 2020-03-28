#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import os
import yaml
import sys
import glob
import gzip
import random
import multiprocessing as mp
import tensorflow as tf
from lib.tfprocess import TFProcess
# from lib.chunkparser import ChunkParser

SKIP = 32
SKIP_MULTIPLE = 1024


def get_all_chunks(path):
    chunks = []
    for dir in glob.glob(path):
        chunks += glob.glob(f"{dir}*.gz")
    return chunks


def get_latest_chunks(path, num_chunks, allow_less):
    chunks = get_all_chunks(path)
    if not allow_less and len(chunks) < num_chunks:
        print(f"Not enough chunks {len(chunks)}")
        sys.exit(1)

    print(f"sorting {len(chunks)} chunks...", end='')
    chunks.sort(key=os.path.getmtime, reverse=True)
    print("[done]")
    chunks = chunks[:num_chunks]
    print("{} - {}".format(os.path.basename(chunks[-1]),
                           os.path.basename(chunks[0])))
    random.shuffle(chunks)
    return chunks


def extract_inputs_outputs(raw):
    def log(s):
        tf.print(s, output_stream=sys.stdout)
        return s

    def logv(s):
        tf.print(s, output_stream=sys.stdout, summarize=-1)
        return s

    def log_(s):
        return s

    def read(pos, len, type):
        return tf.io.decode_raw(tf.strings.substr(raw, pos, len), type)
    log("HELLO WORLD\n\n\n")

    # First 4 bytes in each batch entry are boring.

    # Next 4 change how we construct some of the unit planes.
    log("\n\ninput_format")
    input_format = log_(read(4, 4, tf.int32))
    input_format = log_(tf.reshape(input_format, [-1, 1, 1, 1]))

    # Next 7432 are easy, policy extraction.
    tf.print("\n\npolicy", output_stream=sys.stdout)
    policy = logv(read(8, 7432, tf.float32))
    # tf.print(policy, output_stream=sys.stdout)

    # Next are 104 bit packed chess boards, they have to be expanded.
    tf.print("\n\nbit_planes", output_stream=sys.stdout)
    bit_planes = log_(read(7440, 832, tf.uint8))
    bit_planes = log_(tf.reshape            (bit_planes, [-1, 104, 8]))
    bit_planes = log_(tf.expand_dims        (bit_planes, -1))
    bit_planes = log_(tf.tile               (bit_planes, [1, 1, 1, 8]))
    bit_planes = log_(tf.bitwise.bitwise_and(bit_planes, [128, 64, 32, 16, 8, 4, 2, 1]))
    bit_planes = log_(tf.cast               (bit_planes, tf.float32))
    bit_planes = log_(tf.minimum            (bit_planes, 1.))
    # tf.print("\n\nbit_planes", output_stream=sys.stdout)
    # tf.print(bit_planes, output_stream=sys.stdout)

    # Next 5 bytes are 1 or 0 to indicate 8x8 planes of 1 or 0.
    unit_planes = read(8272, 5, tf.uint8)
    unit_planes = tf.expand_dims(unit_planes, -1)
    unit_planes = tf.expand_dims(unit_planes, -1)
    unit_planes = tf.tile       (unit_planes, [1, 1, 8, 8])
    # In order to do the conditional for frc we need to make bit unpacked versions.
    # Note little endian for these fields so the bitwise_and array is reversed.
    bitsplat_unit_planes = tf.bitwise.bitwise_and(unit_planes, [1, 2, 4, 8, 16, 32, 64, 128])
    bitsplat_unit_planes = tf.cast               (bitsplat_unit_planes, tf.float32)
    bitsplat_unit_planes = tf.minimum            (bitsplat_unit_planes, 1.)
    unit_planes = tf.cast(unit_planes, tf.float32)

    # Fifty-move rule count plane.
    rule50_plane = read(8277, 1, tf.uint8)
    rule50_plane = tf.expand_dims(rule50_plane, -1)
    rule50_plane = tf.expand_dims(rule50_plane, -1)
    rule50_plane = tf.tile       (rule50_plane, [1, 1, 8, 8])
    rule50_plane = tf.cast       (rule50_plane, tf.float32)
    rule50_plane = tf.divide     (rule50_plane, 99.)

    # Zero plane and one plane.
    zero_plane = tf.zeros_like(rule50_plane)
    one_plane = tf.ones_like(rule50_plane)

    # For FRC unit planes must be replaced with 0 and 2 merged, 1 and 3 merged, two zero planes and then 4.
    queenside = tf.concat([
        bitsplat_unit_planes[:, :1, :1], zero_plane[:, :, :6],
        bitsplat_unit_planes[:, 2:3, :1]
    ], 2)
    kingside = tf.concat([
        bitsplat_unit_planes[:, 1:2, :1], zero_plane[:, :, :6],
        bitsplat_unit_planes[:, 3:4, :1]
    ], 2)
    unit_planes = tf.where(
        input_format == 2,
        tf.concat(
            [queenside, kingside, zero_plane, zero_plane, unit_planes[:, 4:]],
            1),
        unit_planes)

    tf.print(bit_planes.shape, output_stream=sys.stdout)

    inputs = tf.concat([bit_planes, unit_planes, rule50_plane, zero_plane, one_plane], 1)
    inputs = tf.reshape(inputs, [-1, 112, 64])

    # Winner is stored in one signed byte and needs to be converted to one hot.
    winner = read(8279, 1, tf.int8)
    winner = tf.cast (winner, tf.float32)
    winner = tf.tile (winner, [1, 3])
    winner = tf.equal(winner, [1., 0., -1.])
    z      = tf.cast (winner, tf.float32)

    # Outcome distribution needs to be calculated from q and d.
    best_q = read(8284, 4, tf.float32)
    best_d = read(8292, 4, tf.float32)
    best_q_w = 0.5 * (1.0 - best_d + best_q)
    best_q_l = 0.5 * (1.0 - best_d - best_q)

    q = tf.concat([best_q_w, best_d, best_q_l], 1)

    ply_count = read(8304, 4, tf.float32)
    tf.print("\n\nGoodbye WORLD", output_stream=sys.stdout)

    return (inputs, policy, z, q, ply_count)


def semi_sample(x):
    return tf.slice(tf.random.shuffle(x), [0], [SKIP_MULTIPLE])


def main(cmd):
    cfg = yaml.safe_load(cmd.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))

    train_ratio = cfg['dataset']['train_ratio']
    num_chunks = cfg['dataset']['num_chunks']
    allow_less = cfg['dataset'].get('allow_less_chunks', False)
    if 'input' in cfg['dataset']:
        chunks = get_latest_chunks(cfg['dataset']['input'], num_chunks, allow_less)
        num_train = int(len(chunks) * train_ratio)
        num_test = len(chunks) - num_train
        train_chunks = chunks[:num_train]
        test_chunks = chunks[num_train:]
    else:
        num_train = int(num_chunks * train_ratio)
        num_test = num_chunks - num_train
        train_chunks = get_latest_chunks(cfg['dataset']['input_train'], num_train, allow_less)
        test_chunks = get_latest_chunks(cfg['dataset']['input_test'], num_test, allow_less)

    total_batch_size = cfg['training']['batch_size']
    batch_splits = cfg['training'].get('num_batch_splits', 1)
    if total_batch_size % batch_splits != 0:
        raise ValueError('num_batch_splits must divide batch_size evenly')
    split_batch_size = total_batch_size // batch_splits

    root_dir = os.path.join(cfg['training']['path'], cfg['name'])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    tfprocess = TFProcess(cfg)

    def read(chunks):
        return tf.data.FixedLengthRecordDataset(
            chunks,
            8308,
            compression_type='GZIP',
            num_parallel_reads=max(2, mp.cpu_count() - 2) // 2)

    def parse(chunks, shuffle_size):
        return tf.data.Dataset\
            .from_tensor_slices(chunks)\
            .shuffle(len(chunks))\
            .repeat()\
            .batch(256)\
            .interleave(read, num_parallel_calls=2)\
            .batch(SKIP_MULTIPLE*SKIP)\
            .map(semi_sample)\
            .unbatch()\
            .shuffle(shuffle_size)\
            .batch(split_batch_size)\
            .map(extract_inputs_outputs)\
            .prefetch(4)

    train_shuffle_size = cfg['training']['shuffle_size']
    test_shuffle_size = int(train_shuffle_size * (1.0 - train_ratio))
    train_dataset = parse(train_chunks, train_shuffle_size)
    test_dataset = parse(test_chunks, test_shuffle_size)

    validation_dataset = None
    if 'input_validation' in cfg['dataset']:
        validation_dataset = \
            read(get_all_chunks(cfg['dataset']['input_validation']))\
            .batch(split_batch_size, drop_remainder=True)\
            .map(extract_inputs_outputs)\
            .prefetch(4)

    tfprocess.init_v2(train_dataset, test_dataset, validation_dataset)
    tfprocess.restore_v2()

    # If number of test positions is not given, sweeps through all test chunks
    # statistically. Assumes average of 10 samples per test game. For
    # simplicity, testing can use the split batch size instead of total batch
    # size. This does not affect results, because test results are simple
    # averages that are independent of batch size.
    num_evals = cfg['training'].get('num_test_positions', len(test_chunks)*10)
    num_evals = max(1, num_evals // split_batch_size)
    print(f"Using {num_evals} evaluation batches")

    tfprocess.process_loop_v2(total_batch_size,
                              num_evals,
                              batch_splits=batch_splits)

    if cmd.output is not None:
        if cfg['training'].get('swa_output', False):
            tfprocess.save_swa_weights_v2(cmd.output)
        else:
            tfprocess.save_leelaz_weights_v2(cmd.output)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
    'Tensorflow pipeline for training Leela Chess.')
    argparser.add_argument('--cfg',
                           type=argparse.FileType('r'),
                           help='yaml configuration with training parameters')
    argparser.add_argument('--output',
                           type=str,
                           help='file to store weights in')

    main(argparser.parse_args())
    mp.freeze_support()
