import multiprocessing
import os
import pickle
import sysconfig
import time
from os import listdir, makedirs, popen
from os.path import isfile, isdir
from random import sample, randrange, choice, shuffle, seed, getstate, setstate, Random
from sys import stdout

import numpy as np
from pybind11.__main__ import print_includes
from io import StringIO
import torch
from torch import nn, LongTensor
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader

from Sophia import SophiaG
from gpt2 import Transformer, TransformerLayer, ToeplitzMode, AblationMode, PositionEmbedding


def build_module(name):
    import sys
    old_stdout = sys.stdout
    try:
        sys.stdout = StringIO()
        print_includes()
        includes = sys.stdout.getvalue().strip()
        sys.stdout.close()
        sys.stdout = old_stdout
    except Exception as e:
        raise e
    finally:
        sys.stdout = old_stdout

    python_extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if sys.platform == "darwin":
        # macOS command
        command = (
            f"g++ -std=c++11 -Ofast -DNDEBUG -fno-stack-protector "
            f"-Wall -Wpedantic -undefined dynamic_lookup -shared -fPIC "
            f"{includes} -I. {name}.cpp -o {name}{python_extension_suffix}"
        )
    else:
        # Non-macOS command
        command = (
            f"g++ -Ofast -std=c++11 -DNDEBUG -fno-stack-protector "
            f"-Wall -Wpedantic -shared -fPIC "
            f"{includes} -I. {name}.cpp -o {name}{python_extension_suffix}"
        )
    print(command)
    if os.system(command) != 0:
        print(f"ERROR: Unable to compile `{name}.cpp`.")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    try:
        from os.path import getmtime
        from importlib.util import find_spec
        generator_spec = find_spec('generator')
        if generator_spec == None:
            raise ModuleNotFoundError
        if getmtime(generator_spec.origin) < getmtime('generator.cpp'):
            print("C++ module `generator` is out-of-date. Compiling from source...")
            build_module("generator")
        import generator
    except ModuleNotFoundError:
        print("C++ module `generator` not found. Compiling from source...")
        build_module("generator")
        import generator
    except ImportError:
        print("Error loading C++ module `generator`. Compiling from source...")
        build_module("generator")
        import generator
    print("C++ module `generator` loaded.")

RESERVED_INDICES = (0,)

class Node(object):
    def __init__(self, id):
        self.id = id
        self.children = []
        self.parents = []

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return 'n(' + str(self.id) + ')'

    def __repr__(self):
        return 'n(' + str(self.id) + ')'

def binomial_confidence_int(p, n):
    return 1.96 * np.sqrt(p * (1.0 - p) / n)

def evaluate_model(model, inputs, outputs):
    device = next(model.parameters()).device
    inputs = torch.tensor(inputs)
    outputs = torch.tensor(outputs)
    inputs = inputs.to(device)
    outputs = outputs.to(device)
    max_input_size = inputs.shape[1]

    if outputs.dim() == 2:
        loss_func = BCEWithLogitsLoss(reduction='mean')
    else:
        loss_func = CrossEntropyLoss(reduction='mean')
    logits, _ = model(inputs)
    loss = loss_func(logits[:, -1, :], outputs).item()

    predictions = torch.argmax(logits[:, -1, :], 1)
    if outputs.dim() == 2:
        acc = torch.sum(torch.gather(outputs, 1, torch.argmax(logits[:,-1,:],dim=1).unsqueeze(1))).item() / outputs.size(0)
    else:
        acc = sum(predictions == outputs).item() / len(predictions)
    return acc, loss, predictions

def unique(x):
    y = []
    for e in x:
        if e not in y:
            y.append(e)
    return y

def train(max_input_size, dataset_size, distribution, max_lookahead, seed_value, nlayers, nhead, hidden_dim, 
          bidirectional, pos_emb, learnable_token_emb, toeplitz_attn, toeplitz_reg, toeplitz_pos_only, 
          add_padding, ablate, pre_ln, curriculum_mode, looped, task, warm_up, batch_size, learning_rate, 
          update_rate, grad_accumulation_steps, max_edges, distance_from_start, max_prefix_vertices, loss):
    
    generator.set_seed(seed_value)
    seed(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    PADDING_TOKEN = (max_input_size-5) // 3 + 3
    BATCH_SIZE = batch_size // grad_accumulation_steps
    print('Number of available CPUs: {}'.format(os.cpu_count()))
    stdout.flush()

    if curriculum_mode == 'y':
        print("Using curriculum learning")

    if loss == "bce":
        print("Using BCE loss")

    if curriculum_mode != 'n' and dataset_size != -1:
        print('ERROR: Curriculum learning is only supported with streaming training (i.e. dataset_size = -1).')
        stdout.flush()
        return
    
    if distribution in ("crafted", "crafted_no_prefix", "star") and max_lookahead == None:
        print('ERROR: Crafted or star training distribution is selected but `max_lookahead` argument is missing.')
        stdout.flush()
        return
    
    if distribution == "simple" and max_lookahead != None:
        print('ERROR: `max_lookahead` is not supported with the simple training distribution.')
        stdout.flush()
        return
    
    if task != "search":
        print('ERROR: This script is configured for search task only.')
        stdout.flush()
        return
    
    if max_lookahead == None:
        max_lookahead = -1

    # Calculate max_edges if not provided
    if max_edges == None:
        max_edges = (max_input_size - 5) // 3

    # first reserve some data for OOD testing
    random_state = getstate()
    np_random_state = np.random.get_state()
    torch_random_state = torch.get_rng_state()

    reserved_inputs = set()
    NUM_TEST_SAMPLES = 10000
    
    # Generate reserved test data for search task
    print('Reserving OOD test data for search task')
    stdout.flush()
    gen_eval_start_time = time.perf_counter()
    
    # Generate test data with full complexity (alpha=1.0)
    eval_inputs, eval_outputs, eval_labels, _ = generator.generate_training_set(
        max_input_size, NUM_TEST_SAMPLES, max_lookahead, max_edges, reserved_inputs, 
        distance_from_start, max_prefix_vertices if max_prefix_vertices != None else -1, True, 1.0)
    
    print('Done. Throughput: {} examples/s'.format(NUM_TEST_SAMPLES / (time.perf_counter() - gen_eval_start_time)))
    for i in range(eval_inputs.shape[0]):
        reserved_inputs.add(tuple([x for x in eval_inputs[i,:] if x != PADDING_TOKEN]))

    if BATCH_SIZE < eval_inputs.shape[0]:
        eval_inputs = eval_inputs[:BATCH_SIZE]
        eval_outputs = eval_outputs[:BATCH_SIZE]

    train_filename = 'train{}_search_inputsize{}_maxlookahead{}_{}seed{}.pkl'.format(
        dataset_size, max_input_size, max_lookahead, 'padded_' if add_padding else '', seed_value)

    prefix = 'search_results/'

    if not torch.cuda.is_available():
        print("WARNING: CUDA device is not available, using CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # compute the checkpoint filenames
    filename = prefix + 'checkpoints_search_{}layer_inputsize{}_maxlookahead{}_seed{}_train{}'.format(
        nlayers, max_input_size, max_lookahead, seed_value, dataset_size if dataset_size != -1 else 'streaming')
    if hidden_dim != 16:
        filename += '_hiddendim{}'.format(hidden_dim)
    if bidirectional:
        filename += '_nomask'
    if pos_emb == 'none':
        filename += '_NoPE'
    elif pos_emb == 'rotary':
        filename += '_RoPE'
    if learnable_token_emb:
        filename += '_learntokemb'
    if ablate != "none":
        filename += '_ablate' + ablate
    if toeplitz_attn:
        filename += '_toeplitz'
        if toeplitz_pos_only:
            filename += 'pos'
    if toeplitz_reg != 0.0:
        filename += '_toeplitz'
        if toeplitz_pos_only:
            filename += 'pos'
        filename += str(toeplitz_reg)
    if not pre_ln:
        filename += '_postLN'
    if add_padding:
        filename += '_padded'
    if curriculum_mode == 'y':
        filename += '_curriculum'
    if looped:
        filename += '_looped'
    if distribution != 'crafted':
        filename += '_' + distribution.replace('_', '-')
    if nhead != 1:
        filename += '_nhead' + str(nhead)
    if warm_up != 0:
        filename += '_warmup' + str(warm_up)
    if batch_size != 2**8:
        filename += '_batchsize' + str(batch_size)
    if learning_rate != 1.0e-5:
        filename += '_lr' + str(learning_rate)
    if update_rate != 2 ** 18:
        filename += '_update' + str(update_rate)
    if loss == "bce":
        filename += '_bce'

    if isdir(filename):
        existing_epochs = [int(ckpt[(ckpt.rfind('epoch') + len('epoch')):-len('.pt')]) for ckpt in listdir(filename) if ckpt.startswith('epoch')]
    else:
        existing_epochs = []
        makedirs(filename)

    ntoken = (max_input_size-5) // 3 + 5
    d_hid = ntoken + hidden_dim
    dropout = 0
    if ablate == "none":
        ablation_mode = AblationMode.NO_ABLATION
    elif ablate == "attn_linear":
        ablation_mode = AblationMode.ABLATE_ATTN_LINEAR
    elif ablate == "attn_linear_projv":
        ablation_mode = AblationMode.ABLATE_ATTN_LINEAR_PROJV
    if toeplitz_attn and toeplitz_pos_only:
        toeplitz = ToeplitzMode.LOWER_RIGHT
    elif toeplitz_attn and not toeplitz_pos_only:
        toeplitz = ToeplitzMode.BLOCK
    else:
        toeplitz = ToeplitzMode.NONE
    if pos_emb == "absolute":
        pos_emb_mode = PositionEmbedding.ABSOLUTE
    elif pos_emb == "rotary":
        pos_emb_mode = PositionEmbedding.ROTARY
    else:
        pos_emb_mode = PositionEmbedding.NONE

    if len(existing_epochs) == 0:
        print("Building search model.")
        model = Transformer(
            layers=nlayers,
            pad_idx=PADDING_TOKEN,
            words=ntoken,
            seq_len=max_input_size,
            heads=nhead,
            dims=max(ntoken, d_hid),
            rate=1,
            dropout=dropout,
            bidirectional=bidirectional,
            pos_emb=pos_emb_mode,
            learn_token_emb=learnable_token_emb,
            ablate=ablation_mode,
            toeplitz=toeplitz,
            pre_ln=pre_ln,
            looped=looped
        )
        model.to(device)
        epoch = 0
    else:
        # Resume from checkpoint
        last_epoch = max(existing_epochs)
        epoch = last_epoch + 1
        print("Loading model from '{}/epoch{}.pt'...".format(filename, last_epoch))
        stdout.flush()
        loaded_obj = torch.load(filename + '/epoch{}.pt'.format(last_epoch), map_location=device)
        model, random_state, np_random_state, torch_random_state = loaded_obj
        setstate(random_state)
        np.random.set_state(np_random_state)
        torch.set_rng_state(torch_random_state.cpu())

    if loss == "bce":
        loss_func = BCEWithLogitsLoss(reduction='mean')
    else:
        loss_func = CrossEntropyLoss(ignore_index=PADDING_TOKEN, reduction='mean')
    
    INITIAL_LR = 1.0e-4
    TARGET_LR = learning_rate

    optimizer = SophiaG((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=0.1)

    log_interval = 1
    eval_interval = 1
    save_interval = 1

    # Initialize curriculum parameters
    if curriculum_mode == 'n':
        curriculum_alpha = 1.0
    elif curriculum_mode == 'y':
        curriculum_alpha = 0.4  # Start with 10% complexity like SI

    if hasattr(model, 'alpha'):
        curriculum_alpha = model.alpha
    else:
        model.alpha = curriculum_alpha

    if hasattr(model, 'max_lookahead'):
        max_lookahead = model.max_lookahead
    else:
        model.max_lookahead = max_lookahead

    if hasattr(model, 'max_edges'):
        max_edges = model.max_edges
    else:
        model.max_edges = max_edges

    if dataset_size == -1:
        # Streaming training setup
        from itertools import cycle
        from threading import Lock
        STREAMING_BLOCK_SIZE = update_rate
        NUM_DATA_WORKERS = 2
        seed_generator = Random(seed_value)
        seed_generator_lock = Lock()
        seed_values = []

        def get_seed(index):
            if index < len(seed_values):
                return seed_values[index]
            seed_generator_lock.acquire()
            while index >= len(seed_values):
                seed_values.append(seed_generator.randrange(2 ** 32))
            seed_generator_lock.release()
            return seed_values[index]

        class StreamingDatasetSearch(torch.utils.data.IterableDataset):
            def __init__(self, offset, alpha, max_lookahead, max_edges, distance_from_start, max_prefix_vertices):
                super(StreamingDatasetSearch).__init__()
                self.offset = offset
                self.alpha = alpha
                self.max_lookahead = max_lookahead
                self.max_edges = max_edges
                self.distance_from_start = distance_from_start
                self.max_prefix_vertices = max_prefix_vertices
                self.multiprocessing_manager = multiprocessing.Manager()
                self.total_collisions = self.multiprocessing_manager.Value(int, 0)
                self.collisions_lock = self.multiprocessing_manager.Lock()

            def process_data(self, start):
                current = start
                worker_info = torch.utils.data.get_worker_info()
                worker_id = worker_info.id
                max_prefix_verts = (0 if distribution == 'crafted_no_prefix' else 
                                   (max_input_size if self.max_prefix_vertices == -1 else self.max_prefix_vertices))
                while True:
                    worker_start_time = time.perf_counter()
                    new_seed = get_seed(current)
                    generator.set_seed(new_seed)
                    seed(new_seed)
                    torch.manual_seed(new_seed)
                    np.random.seed(new_seed)

                    generate_start_time = time.perf_counter()
                    # Generate search training data with curriculum learning support
                    batch = generator.generate_training_set(
                        max_input_size, BATCH_SIZE, self.max_lookahead, self.max_edges,
                        reserved_inputs, self.distance_from_start, max_prefix_verts, 
                        True, self.alpha)
                    
                    if batch[3] != 0:  # num_collisions
                        with self.collisions_lock:
                            self.total_collisions.value += batch[3]
                        stdout.flush()

                    worker_end_time = time.perf_counter()
                    yield batch[:-1]  # Return inputs, outputs, labels (exclude num_collisions)
                    current += NUM_DATA_WORKERS

            def __iter__(self):
                worker_info = torch.utils.data.get_worker_info()
                worker_id = worker_info.id
                return self.process_data(self.offset + worker_id)

        dataset = StreamingDatasetSearch(
            epoch * STREAMING_BLOCK_SIZE // BATCH_SIZE, model.alpha, 
            model.max_lookahead, model.max_edges, distance_from_start,
            max_prefix_vertices if max_prefix_vertices != None else -1)
        loader = DataLoader(dataset, batch_size=None, num_workers=NUM_DATA_WORKERS, 
                           pin_memory=True, prefetch_factor=8)

    examples_seen = epoch * STREAMING_BLOCK_SIZE
    LR_DECAY_TIME = 2**24  # examples seen

    while True:
        start_time = time.perf_counter()
        transfer_time = 0.0
        train_time = 0.0
        log_time = 0.0
        epoch_loss = 0.0
        num_batches = 0
        effective_dataset_size = (STREAMING_BLOCK_SIZE if dataset_size == -1 else dataset_size)
        reinit_data_loader = False
        
        for batch in loader:
            batch_start_time = time.perf_counter()

            # Learning rate scheduling
            if warm_up != 0:
                if examples_seen < warm_up:
                    lr = examples_seen * INITIAL_LR / warm_up
                elif examples_seen < warm_up + LR_DECAY_TIME:
                    lr = (0.5 * np.cos(np.pi * (examples_seen - warm_up) / LR_DECAY_TIME) + 0.5) * (
                                INITIAL_LR - TARGET_LR) + TARGET_LR
                else:
                    lr = TARGET_LR
            else:
                lr = TARGET_LR
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            model.train()

            inputs, outputs, labels = batch
            inputs = inputs.to(device, non_blocking=True)
            outputs = outputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            examples_seen += BATCH_SIZE

            train_start_time = time.perf_counter()
            transfer_time += train_start_time - batch_start_time

            # Forward pass
            logits = model(inputs)

            if loss == "bce":
                loss_val = loss_func(logits[:, -1, :], outputs)
            else:
                loss_val = loss_func(logits[:, -1, :], labels)

            epoch_loss += loss_val.item()
            loss_val.backward()

            if examples_seen % (BATCH_SIZE * grad_accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()

            log_start_time = time.perf_counter()
            train_time += log_start_time - train_start_time
            num_batches += 1

            if num_batches == effective_dataset_size // BATCH_SIZE:
                if epoch % log_interval == 0:
                    elapsed_time = time.perf_counter() - start_time
                    avg_loss = epoch_loss / num_batches
                    print("epoch = {}, training loss = {:.6f}".format(epoch, avg_loss))
                    if device.type == 'cuda':
                        utilization = popen('nvidia-smi --query-gpu=utilization.gpu --format=csv').read().split('\n')[1]
                        print("throughput = {} examples/s, GPU utilization = {}".format(
                            effective_dataset_size / elapsed_time, utilization))
                    else:
                        print("throughput = {} examples/s".format(effective_dataset_size / elapsed_time))
                    print('Total number of training examples in test set: {}'.format(dataset.total_collisions.value))
                    print('Learning rate: {}'.format(lr))
                    if curriculum_mode == 'y':
                        print('Curriculum alpha: {:.2f}'.format(model.alpha))
                    print("[PROFILE] Total batch time: {}s".format(elapsed_time))
                    print("[PROFILE] Time to transfer data to GPU: {}s".format(transfer_time))
                    print("[PROFILE] Time to train: {}s".format(train_time))
                    print("[PROFILE] Time to log/save/validate: {}s".format(log_time))
                    stdout.flush()
                    start_time = time.perf_counter()
                    transfer_time = 0.0
                    train_time = 0.0
                    log_time = 0.0

                if epoch % eval_interval == 0:
                    model.eval()

                    # Training accuracy
                    logits, _ = model(inputs)
                    if loss == "bce":
                        training_acc = torch.sum(torch.gather(outputs, 1, 
                            torch.argmax(logits[:,-1,:],dim=1).unsqueeze(1))).item() / outputs.size(0)
                    else:
                        predictions = torch.argmax(logits[:, -1, :], 1)
                        training_acc = sum(predictions == labels).item() / len(predictions)

                    print("training accuracy: %.2f±%.2f" % (training_acc, 
                          binomial_confidence_int(training_acc, inputs.size(0))))
                    del inputs, outputs, labels
                    stdout.flush()

                    # Test accuracy
                    test_acc, test_loss, _ = evaluate_model(model, eval_inputs, eval_outputs)
                    print("Epoch {}: Test Acc = {:.2f}±{:.2f}, Loss = {:.6f}".format(
                        epoch, test_acc, binomial_confidence_int(test_acc, eval_inputs.shape[0]), test_loss))
                    stdout.flush()

                    # Curriculum learning update
                    if curriculum_mode == 'y' and training_acc > 0.99:
                        if model.alpha < 1.0:
                            model.alpha = min(1.0, model.alpha + 0.1)
                            print("Curriculum update: alpha increased to {:.2f}".format(model.alpha))
                            reinit_data_loader = True
                            break

                if epoch % save_interval == 0:
                    ckpt_filename = filename + '/epoch{}.pt'.format(epoch)
                    print('Saving model to "{}".'.format(ckpt_filename))
                    # torch.save((model, getstate(), np.random.get_state(), torch.get_rng_state()), ckpt_filename)
                    print('Done saving model.')
                    stdout.flush()

                epoch += 1
                num_batches = 0
                epoch_loss = 0.0
                if reinit_data_loader:
                    break

            log_end_time = time.perf_counter()
            log_time += log_end_time - log_start_time

        if reinit_data_loader:
            dataset = StreamingDatasetSearch(
                epoch * STREAMING_BLOCK_SIZE // BATCH_SIZE, model.alpha,
                model.max_lookahead, model.max_edges, distance_from_start,
                max_prefix_vertices if max_prefix_vertices != None else -1)
            loader = DataLoader(dataset, batch_size=None, num_workers=NUM_DATA_WORKERS,
                              pin_memory=True, prefetch_factor=8)
            reinit_data_loader = False


if __name__ == "__main__":
    import argparse
    def parse_bool_arg(v):
        if isinstance(v, bool):
            return v
        elif v.lower() in ('yes', 'true', 'y', 't', '1'):
            return True
        elif v.lower() in ('no', 'false', 'n', 'f', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-input-size", type=int)
    parser.add_argument("--dataset-size", type=int)
    parser.add_argument("--max-lookahead", type=int, required=False)
    parser.add_argument("--max-edges", type=int, required=False)
    parser.add_argument("--distance-from-start", type=int, default=-1, required=False)
    parser.add_argument("--max-prefix-vertices", type=int, required=False)
    parser.add_argument("--nlayers", type=int)
    parser.add_argument("--nhead", type=int, default=1, required=False)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bidirectional", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--pos-emb", type=str, required=True, choices=["absolute", "rotary", "none"])
    parser.add_argument("--learn-tok-emb", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--toeplitz-attn", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--toeplitz-reg", type=float, required=True, default=0.0)
    parser.add_argument("--toeplitz-pos-only", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--add-padding", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--ablate", type=str, default="none", choices=["none", "attn_linear", "attn_linear_projv"])
    parser.add_argument("--preLN", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--curriculum", type=str, required=True, choices=["y", "n"])
    parser.add_argument("--looped", type=parse_bool_arg, default=False)
    parser.add_argument("--task", type=str, default="search", choices=["search"])
    parser.add_argument("--distribution", type=str, default="crafted", choices=["simple", "crafted", "crafted_no_prefix", "star"])
    parser.add_argument("--warm-up", type=int, default=0, required=False)
    parser.add_argument("--batch-size", type=int, default=2**8, required=False)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5, required=False)
    parser.add_argument("--update-rate", type=int, default=2**18, required=False)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1, required=False)
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'bce'], required=False)
    args = parser.parse_args()

    train(
        max_input_size=args.max_input_size,
        dataset_size=args.dataset_size,
        distribution=args.distribution,
        max_lookahead=args.max_lookahead,
        seed_value=args.seed,
        nhead=args.nhead,
        nlayers=args.nlayers,
        hidden_dim=args.hidden_dim,
        bidirectional=args.bidirectional,
        pos_emb=args.pos_emb,
        learnable_token_emb=args.learn_tok_emb,
        toeplitz_attn=args.toeplitz_attn,
        toeplitz_reg=args.toeplitz_reg,
        toeplitz_pos_only=args.toeplitz_pos_only,
        add_padding=args.add_padding,
        ablate=args.ablate,
        pre_ln=args.preLN,
        curriculum_mode=args.curriculum,
        looped=args.looped,
        task=args.task,
        warm_up=args.warm_up,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        update_rate=args.update_rate,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_edges=args.max_edges,
        distance_from_start=args.distance_from_start,
        max_prefix_vertices=args.max_prefix_vertices,
        loss=args.loss)