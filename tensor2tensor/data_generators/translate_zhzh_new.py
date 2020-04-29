# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf
@registry.register_problem
class TranslateZhzhNew(translate.TranslateProblem):
  """Problem spec for WMT En-De translation, BPE version."""

  @property
  def approx_vocab_size(self):
    return 65668

  @property
  def vocab_filename(self):
    return "/home/makai/tensor2tensor/test.vocab"

  def generate_encoded_samples(self, data_dir, tmp_dir, input_file):
    generator = self.generate_samples(data_dir, tmp_dir, input_file)
    encoder = text_encoder.SubwordTextEncoder(self.vocab_filename)
    for sample in generator:
      if self.has_inputs:
        sample["inputs"] = encoder.encode(sample["inputs"])
        sample["inputs"].append(text_encoder.EOS_ID)
      if "targets" in sample:
        sample["targets"] = encoder.encode(sample["targets"])
        sample["targets"].append(text_encoder.EOS_ID)
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
          flds = line.strip().split("\t")
          source = flds[1]
          target = flds[3]
          out = {}
          if source:
            out["inputs"] = source
          if target:
            out["targets"] = target
          yield out
          line = source_file.readline()

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    input_file = os.path.join(data_dir, "input")

    all_paths = [data_dir + "translate_zhzhnew" + generator_utils.UNSHUFFLED_SUFFIX]

    generator_utils.generate_files(
        self._maybe_pack_examples(
            self.generate_encoded_samples(
                data_dir, tmp_dir, input_file)), all_paths)

    generator_utils.shuffle_dataset(all_paths)

