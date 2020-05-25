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
from multiprocessing import Pool
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS




def serailize_example(inputs, targets):
    case = {"inputs": inputs, "targets": targets}
    example = generator_utils.to_example(case)
    return example.SerializeToString()


@registry.register_problem
class TranslateZhzhNewSpark(translate.TranslateProblem):
    """Problem spec for WMT En-De translation, BPE version."""

    @property
    def approx_vocab_size(self):
        return FLAGS.vocab_size

    @property
    def vocab_filename(self):
        return FLAGS.vocab

    def generate_data(self, table_name, spark, output_path, encode_udf):
        sc = spark.sparkContext
        rdf = spark.sql("select norm_note_text, r_norm_note_text from %s" % table_name)
        df = rdf.select(encode_udf(rdf.norm_note_text).alias("inputs"),
                encode_udf(rdf.r_norm_note_text).alias("targets"))
        df.write.format("tfrecords").option("recordType", "Example").save(output_path)
