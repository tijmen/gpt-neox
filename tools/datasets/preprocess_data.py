# Copyright (c) 2024, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Processing data for pretraining."""

import argparse
import multiprocessing
import os
import sys

import lm_dataformat as lmd
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
)
import time
import tqdm
import torch
import ftfy

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
from threading import Semaphore


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, text):
        if self.args.ftfy:
            text = ftfy.fix_text(text)
        ids = {}
        max_seq_length = self.args.max_seq_length

        for key in self.args.jsonl_keys:
            doc_ids = []
            text_ids = Encoder.tokenizer.tokenize(text)
            
            # Filter out None values and ensure we have valid tokens
            if text_ids:
                text_ids = [t for t in text_ids if t is not None]
                
                if len(text_ids) > 0:
                    # Split into chunks of max_seq_length
                    for i in range(0, len(text_ids), max_seq_length):
                        chunk = text_ids[i:i + max_seq_length]
                        
                        # Leave room for EOD token if needed
                        if self.args.append_eod and hasattr(Encoder.tokenizer, 'eod'):
                            chunk = chunk[:max_seq_length - 1]
                            chunk.append(Encoder.tokenizer.eod)
                        
                        # Only pad if not doing sample packing
                        if self.args.pad_to_max_seq_length and not self.args.sample_packing:
                            pad_id = Encoder.tokenizer.pad_id
                            chunk.extend([pad_id] * (max_seq_length - len(chunk)))
                        
                        doc_ids.append(chunk)
                        
            ids[key] = doc_ids
        return ids, len(text)


def get_args(input_args=None):
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input jsonl files or lmd archive(s) - if using multiple archives, put them in a comma separated "
        "list",
    )
    group.add_argument(
        "--jsonl-keys",
        nargs="+",
        default=["text"],
        help="space separate listed of keys to extract from jsonl. Default: text",
    )
    group.add_argument(
        "--num-docs",
        default=None,
        help="Optional: Number of documents in the input data (if known) for an accurate progress bar.",
        type=int,
    )
    group.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Maximum sequence length. Longer documents will be split into multiple sequences.",
    )
    group.add_argument(
        "--pad-to-max-seq-length",
        action="store_true",
        help="Pad the sequences to the maximum sequence length.",
    )
    group.add_argument(
        "--sample-packing",
        action="store_true",
        help="Enable sample packing to maximize sequence utilization.",
    )
    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        required=True,
        choices=[
            "HFGPT2Tokenizer",
            "HFTokenizer",
            "GPT2BPETokenizer",
            "CharLevelTokenizer",
            "TiktokenTokenizer",
            "SPMTokenizer",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument(
        "--vocab-file", type=str, default=None, help="Path to the vocab file"
    )
    group.add_argument(
        "--merge-file",
        type=str,
        default=None,
        help="Path to the BPE merge file (if necessary).",
    )
    group.add_argument(
        "--append-eod",
        action="store_true",
        help="Append an <eod> token to the end of a document.",
    )
    group.add_argument("--ftfy", action="store_true", help="Use ftfy to clean text")
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )
    group.add_argument(
        "--dataset-impl",
        type=str,
        default="mmap",
        choices=["lazy", "cached", "mmap"],
        help="Dataset implementation to use. Default: mmap",
    )

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes to launch"
    )
    group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Interval between progress updates",
    )
    args = parser.parse_args(input_args)
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args


def yield_from_files(fnames: list, semaphore):
    """
    Iterator over input documents using lm_dataformat. Should be able to handle jsons / texts /
    other compressed formats. Also filters out empty documents.

    :param fnames: list of filenames
    """

    def yielder(fname, semaphore):
        for f in filter(lambda x: x, lmd.Reader(fname).stream_data()):
            semaphore.acquire()
            yield f

    for fname in fnames:
        semaphore.acquire()

        yield from yielder(fname, semaphore)


def pack_sequences(sequences, max_seq_length, pad_id, window_size=10):
    """Pack sequences together to maximize utilization.
    
    Args:
        sequences: List of token sequences
        max_seq_length: Maximum allowed sequence length
        pad_id: Token ID to use for padding
        window_size: How far to look ahead for potential matches
    
    Returns:
        List of packed sequences, each of length max_seq_length
    """
    packed_sequences = []
    used = set()
    
    for i in range(len(sequences)):
        if i in used:
            continue
            
        current_seq = sequences[i]
        used.add(i)
        
        # Look ahead up to window_size sequences
        for j in range(i + 1, min(i + window_size + 1, len(sequences))):
            if j in used:
                continue
                
            next_seq = sequences[j]
            # Check if we can fit the next sequence
            if len(current_seq) + len(next_seq) <= max_seq_length:
                current_seq.extend(next_seq)
                used.add(j)
            else:
                break
                
        # Pad to max_seq_length
        current_seq.extend([pad_id] * (max_seq_length - len(current_seq)))
        packed_sequences.append(current_seq)
        
    return packed_sequences


def main(input_args=None):
    args = get_args(input_args)
    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")

    # build a semaphore object to stop `yield_from_files` from getting ahead of encoder.encode and
    # hence building up memory
    semaphore = Semaphore(10000 + args.workers)

    # use multiprocessing to iterate over input documents
    fin = yield_from_files(args.input.split(","), semaphore)

    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, chunksize=25)
    else:
        encoder.initializer()
        encoded_docs = (encoder.encode(doc) for doc in fin)

    # make a dataset builder for each key in args.jsonl_keys
    # each key will output to a different file beginning with args.output_prefix
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.jsonl_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(
            args.output_prefix, key, "document"
        )
        output_idx_files[key] = "{}_{}_{}.idx".format(
            args.output_prefix, key, "document"
        )
        builders[key] = indexed_dataset.make_builder(
            output_bin_files[key],
            impl=args.dataset_impl,
            vocab_size=tokenizer.vocab_size,
        )

    # actually do tokenization
    proc_start = time.time()
    total_bytes_processed = 0
    pbar = tqdm.tqdm()
    
    # Buffer for sample packing
    sequence_buffer = {key: [] for key in args.jsonl_keys}
    buffer_docs = 0
    PACK_INTERVAL = 100  # Pack sequences every N documents
    
    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        semaphore.release()

        for key, sentences in doc.items():
            if args.sample_packing:
                sequence_buffer[key].extend(sentences)
                buffer_docs += 1
                
                # Pack and flush buffer periodically
                if buffer_docs >= PACK_INTERVAL:
                    packed_seqs = pack_sequences(
                        sequence_buffer[key],
                        args.max_seq_length,
                        tokenizer.pad_id
                    )
                    for seq in packed_seqs:
                        builders[key].add_item(np.array(seq, dtype=builders[key].dtype))
                    builders[key].end_document()
                    sequence_buffer[key] = []
                    buffer_docs = 0
            else:
                for sentence in sentences:
                    builders[key].add_item(np.array(sentence, dtype=builders[key].dtype))
                builders[key].end_document()

        # log progress
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            pbar.set_description(
                f"Processed {i}{'' if args.num_docs is None else '/' + str(args.num_docs)} documents ({i / elapsed :.2f} docs/s, {mbs:.2f} MB/s)."
            )
            if i != 0:
                pbar.update(args.log_interval)

    # Handle remaining sequences in buffer
    if args.sample_packing:
        for key in args.jsonl_keys:
            if sequence_buffer[key]:
                packed_seqs = pack_sequences(
                    sequence_buffer[key],
                    args.max_seq_length,
                    tokenizer.pad_id
                )
                for seq in packed_seqs:
                    builders[key].add_item(np.array(seq, dtype=builders[key].dtype))
                builders[key].end_document()

    # save output file
    for key in args.jsonl_keys:
        builders[key].finalize(output_idx_files[key])


if __name__ == "__main__":
    main()
