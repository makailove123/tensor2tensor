# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import numpy as np
from multiprocessing import Pool
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry, decoding

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

def generate_data_one(args):
    problem, data_dir, tmp_dir, input_file, output_file = args
    output_fp = open(output_file, "w")
    for sample in problem.generate_encoded_samples(data_dir, tmp_dir, input_file):
        print("{}\t{}".format(sample["input_id"][0], sample["inputs_d"]), file=output_fp)
    output_fp.close()
    return
    generator_utils.generate_files(
            problem._maybe_pack_examples(
                problem.generate_encoded_samples(
                    data_dir, tmp_dir, input_file)), [output_file])

@registry.register_problem
class TranslateZhzhNew(translate.TranslateProblem):
    """Problem spec for WMT En-De translation, BPE version."""

    @property
    def approx_vocab_size(self):
        return FLAGS.vocab_size

    @property
    def vocab_filename(self):
        return FLAGS.vocab

    def generate_encoded_samples(self, data_dir, tmp_dir, input_file):
        generator = self.generate_samples(data_dir, tmp_dir, input_file)
        encoder = text_encoder.SubwordTextEncoder(self.vocab_filename)
        for sample in generator:
            if self.has_inputs:
                if FLAGS.max_seq_len > 0:
                    sample["inputs"] = encoder.encode(sample["inputs"])[:FLAGS.max_seq_len]
                else:
                    sample["inputs"] = encoder.encode(sample["inputs"])
                sample["inputs"].append(text_encoder.EOS_ID)
            if "targets" in sample:
                if FLAGS.max_seq_len > 0:
                    sample["targets"] = encoder.encode(sample["targets"])[:FLAGS.max_seq_len]
                else:
                    sample["targets"] = encoder.encode(sample["targets"])
                sample["targets"].append(text_encoder.EOS_ID)
            if "inputs" in sample:
              sample["inputs_d"] = encoder.decode(decoding._save_until_eos(np.array(sample["inputs"], dtype=np.int32), False))
            if "outputs" in sample:
              sample["targets_d"] = encoder.decode(decoding._save_until_eos(np.array(sample["targets"], dtype=np.int32), False))
            yield sample

    def feature_encoders(self, data_dir):
        source_token = text_encoder.SubwordTextEncoder(self.vocab_filename)
        target_token = text_encoder.SubwordTextEncoder(self.vocab_filename)
        return {
                "inputs": source_token,
                "targets": target_token,
        }

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        for _ in range(self.num_samples(dataset_split)):
            source = self.generate_random_sequence(dataset_split)
            target = self.transpose_sequence(source)
            yield {
                    "inputs": " ".join(source),
                    "targets": " ".join(target),
            }

    def generate_samples(self, data_dir, tmp_dir, input_file):
        vocab = text_encoder.SubwordTextEncoder(self.vocab_filename)
        data_path = input_file
        with tf.gfile.GFile(data_path, mode="r") as source_file:
            line = source_file.readline()
            while line:
                flds = line.rstrip("\r\n").split("\t")
                source_id = flds[0]
                source = flds[1]
                target_id = flds[2]
                target = flds[3]
                out = {}
                if source_id:
                    out["input_id"] = [int(source_id)]
                if source:
                    out["inputs"] = source
                if target_id:
                    out["target_id"] = [int(target_id)]
                if target:
                    out["targets"] = target
                yield out
                line = source_file.readline()


    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        if FLAGS.input_path:
            input_path = FLAGS.input_path
        else:
            input_path = os.path.join(data_dir, "input")
        if os.path.isdir(input_path):
            input_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if not f.startswith(".")]
        else:
            input_files = [input_path]

        output_files = generator_utils.train_data_filenames(FLAGS.problem, data_dir,len(input_files))

        if not FLAGS.parallel > 0:
            for input_file, output_file in zip(input_files, output_files):
                generate_data_one((self, data_dir, tmp_dir, input_file, output_file))
        else:
            pool = Pool(FLAGS.parallel)
            pool.map(generate_data_one, [(self, data_dir, tmp_dir, input_file, output_file)
                for input_file, output_file in zip(input_files, output_files)])

        #generator_utils.shuffle_dataset(all_paths)

