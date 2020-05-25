#coding=utf-8

import sys
from tensor2tensor.data_generators import text_encoder

def text_generator():
    linecnt = 0
    for line in sys.stdin:
        linecnt += 1
        if linecnt % 100000 == 0:
            print("{} readed".format(linecnt), file=sys.stderr, flush=True)
        for pic in line.split("\t", 2)[1].split(" "):
            if pic:
                yield pic

encoder = text_encoder.SubwordTextEncoder.build_from_generator(text_generator(), 65536, max_subtoken_length=4)
#encoder = text_encoder.SubwordTextEncoder("single_char.vocab")

s = encoder.encode("该公式就成了一个指数为负数的幂函数")
print(s)
encoder.store_to_file("test.vocab")
