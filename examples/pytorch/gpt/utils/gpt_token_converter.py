# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../..")
from examples.tensorflow.gpt.utils import gpt_token_encoder as encoder
import fire
import numpy as np

def convert_token(
    vocab_file="../models/gpt2-vocab.json",
    bpe_file="../models/gpt2-merges.txt",
    out_file="out",
    max_input_length=-1,
    text_out_file=None,
):
    enc = encoder.get_encoder(vocab_file, bpe_file)
    tokens_batch = np.loadtxt(out_file, dtype=np.int32)
    end_id = 50256
    outputs = []
    print(f"[INFO] tokens_batch dim {tokens_batch.ndim}")
    print(f"[INFO] tokens_batch {tokens_batch}")
    if(tokens_batch.ndim == 1): 
        tokens_batch = tokens_batch.reshape([1, -1])
    for batch_num, tokens in enumerate(tokens_batch):
        print(f"[INFO] batch_num {batch_num}")
        if max_input_length > -1:
            end_index = np.where(tokens[max_input_length:] == end_id)[0]
        else:
            end_index = []
        end_pos = len(tokens)
        if len(end_index) > 0:
            end_pos = end_index[0]
        print(f"[INFO] batch {batch_num}: \n[input]:\n{tokens[:16]}\n{enc.decode(tokens[:16])}\n[output]:\n{tokens[16:end_pos]}\n{enc.decode(tokens[16:end_pos])}")
        outputs.append(enc.decode(tokens[:end_pos]))

        #print(f"\n[ids]:{tokens[0]} {enc.decode(tokens[0:1])}")
        #print(f"\n[ids]:{tokens[1]}{enc.decode(tokens[1:2])}")
        for idx_i in range(0,end_pos-1):
            print(f"\n{idx_i}: [ids][{tokens[idx_i]}]----[{enc.decode(tokens[idx_i:idx_i+1])}]")
        
        if text_out_file != None:
            with open(text_out_file, "w+") as f:
                f.writelines("\n".join(outputs))
    # return tokens_batch

if __name__ == "__main__":
    fire.Fire(convert_token)