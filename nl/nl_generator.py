"""
Natural Language Generator for Graph Tasks
"""

import os
import sys
import sysconfig
from io import StringIO
import json
import pickle
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Iterator, Set
from dataclasses import dataclass
import multiprocessing
from threading import Lock
from random import Random
import torch
from torch.utils.data import IterableDataset, DataLoader

# For name and attribute generation
from faker import Faker

# Shared predicate templates for all tasks
PREDICATE_TEMPLATES = [
    "If {name} is {adj_a}, then {name} is {adj_b}.",
    "If {name} is {adj_a}, then they are {adj_b}.",
    "If a person is {adj_a}, they are {adj_b}.",
    "Everyone that is {adj_a} is {adj_b}.",
    "If someone is {adj_a}, then they are {adj_b}."
]


def build_module(name):
    """Build the C++ module if needed"""
    from pybind11.__main__ import print_includes

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
        sys.exit(1)


# Try to import or compile the C++ generator module
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


@dataclass
class NLExample:
    """Container for a natural language example"""
    input_text: str
    output_texts: List[str]  # Changed to list of outputs
    labels: List[Any]  # Changed to list of labels
    output_vector: Any = None  # Store the original one-hot vector
    metadata: Dict = None


class NameAttributeGenerator:
    """Generates consistent names and attributes for nodes"""

    def __init__(self, seed=None):
        self.fake = Faker()
        if seed:
            Faker.seed(seed)
            random.seed(seed)
        self.fake.unique.clear()

    def generate_names(self, n: int) -> List[str]:
        """Generate n unique names"""
        self.fake.unique.clear()
        return [self.fake.unique.first_name() for _ in range(n)]

    def random_syllable(self) -> str:
        """Generate a random syllable"""
        vowels = "aeiou"
        consonants = "bcdfghjklmnpqrstvwxyz"
        patterns = ["CV", "VC", "CVC", "CVV", "CCV", "VCV", "VCC"]
        pattern = random.choice(patterns)

        syllable = ""
        for char in pattern:
            if char == "C":
                syllable += random.choice(consonants)
            elif char == "V":
                syllable += random.choice(vowels)
        return syllable

    def generate_attribute(self) -> str:
        """Generate a fake 2-syllable attribute"""
        return self.random_syllable() + self.random_syllable()

    def generate_attributes(self, n: int) -> List[str]:
        """Generate n unique fake attributes"""
        attributes = set()
        while len(attributes) < n:
            attributes.add(self.generate_attribute())
        attributes = sorted(list(attributes))
        random.shuffle(attributes)
        return attributes


def convert_to_int(value):
    """Safely convert any numeric type to Python int"""
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    return int(value)


def generate_predicate_lines(edges, id_to_attr, name):
    """Generate varied predicate lines for all tasks using shared templates"""
    lines = ["Suppose we have the following facts:"]

    for (A, B) in edges:
        adj_A = id_to_attr[A]
        adj_B = id_to_attr[B]

        # Randomly choose a template
        template = random.choice(PREDICATE_TEMPLATES)

        # Fill in the template
        line = template.format(name=name, adj_a=adj_A, adj_b=adj_B)
        lines.append(line)

    return lines


class NaturalLanguageGraphGenerator:
    """Converts symbolic graph tasks to natural language"""

    def __init__(self, max_input_size: int, seed: int = None, debug: bool = False):
        self.max_input_size = max_input_size
        self.seed = seed
        self.debug = debug
        if seed:
            generator.set_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        # Token definitions
        self.QUERY_PREFIX_TOKEN = (max_input_size - 5) // 3 + 4
        self.PADDING_TOKEN = (max_input_size - 5) // 3 + 3
        self.EDGE_PREFIX_TOKEN = (max_input_size - 5) // 3 + 2
        self.PATH_PREFIX_TOKEN = (max_input_size - 5) // 3 + 1

        # Maximum valid node ID
        self.MAX_NODE_ID = (max_input_size - 5) // 3 - 1

        self.name_gen = NameAttributeGenerator(seed)

    def _parse_symbolic_output(self, inputs: np.ndarray, outputs: np.ndarray,
                               labels: np.ndarray = None) -> List[Dict]:
        """Parse symbolic outputs into structured format - now handles multiple outputs"""
        examples = []

        for i in range(inputs.shape[0]):
            # Remove padding tokens
            input_seq = [x for x in inputs[i] if x != self.PADDING_TOKEN]

            # Parse edges and query
            edges = []
            query = None
            path = []

            j = 0
            while j < len(input_seq):
                if input_seq[j] == self.EDGE_PREFIX_TOKEN:
                    edges.append((convert_to_int(input_seq[j + 1]),
                                  convert_to_int(input_seq[j + 2])))
                    j += 3
                elif input_seq[j] == self.QUERY_PREFIX_TOKEN:
                    query = (convert_to_int(input_seq[j + 1]),
                             convert_to_int(input_seq[j + 2]))
                    j += 3
                elif input_seq[j] == self.PATH_PREFIX_TOKEN:
                    j += 1
                    # Rest is path - convert each element to int
                    path = [convert_to_int(x) for x in input_seq[j:]]
                    break
                else:
                    j += 1

            # Handle multiple outputs
            if outputs.ndim == 1:
                # Single output value
                output_vals = [convert_to_int(outputs[i])]
                output_vector = None
            else:
                # Multi-dimensional output - find all correct answers
                output_row = outputs[i]
                output_vector = output_row.copy()  # Save the original vector

                # Find all indices where output is 1 (or close to 1 for float arrays)
                if output_row.dtype == np.float32 or output_row.dtype == np.float64:
                    output_vals = [convert_to_int(idx) for idx, val in enumerate(output_row) if val > 0.5]
                else:
                    output_vals = [convert_to_int(idx) for idx, val in enumerate(output_row) if val == 1]

                # If no outputs found (all zeros), fall back to argmax
                if len(output_vals) == 0:
                    output_vals = [convert_to_int(output_row.argmax())]

            example = {
                'edges': edges,
                'query': query,
                'path': path,
                'outputs': output_vals,  # Now a list
                'output_vector': output_vector,  # Store original vector
                'label': convert_to_int(labels[i]) if labels is not None else None
            }

            if self.debug:
                if sum(output_vector) > 1:
                    print(f"\nParsed example {i}:")
                    print(f"  Edges: {edges}")
                    print(f"  Query: {query}")
                    print(f"  Path: {path}")
                    print(f"  Outputs: {output_vals}")
                    if output_vector is not None:
                        print(f"  Output vector: {output_vector}")

            examples.append(example)

        return examples

    def _generate_search_nl(self, graph_data: Dict,
                            use_different_names: bool = False) -> NLExample:
        """
        Convert a SEARCH task to natural language.

        Changes:
          - Fix A: filter the path by membership in the actual graph node set (no ID cap).
          - Sanity checks (printed when self.debug is True):
              * Warn if any path nodes were dropped by membership filtering.
              * Warn if label is not a one-hop neighbor of the current node.
              * Warn if any outputs are not one-hop neighbors of the current node.
        """
        edges = graph_data['edges']
        query = graph_data['query']
        path = graph_data['path']
        outputs = graph_data['outputs']  # list[int]
        output_vector = graph_data.get('output_vector')
        label_id = graph_data.get('label')  # scalar int or None

        # --- Build the actual node set from the sample ---
        node_ids: Set[int] = set()
        for a, b in edges:
            node_ids.add(int(a))
            node_ids.add(int(b))
        if query:
            node_ids.add(int(query[0]))
            node_ids.add(int(query[1]))
        for o in outputs:
            node_ids.add(int(o))

        # --- Fix A: membership-based path filtering (no arbitrary max) ---
        path = [int(n) for n in path]  # normalize to int
        filtered_path = [n for n in path if n in node_ids]

        if self.debug and len(filtered_path) != len(path):
            dropped = [n for n in path if n not in node_ids]
            print(f"[SEARCH][WARN] Dropped {len(dropped)} path node(s) "
                  f"not present in graph node set: {dropped}")

        # --- Prepare names/attributes ---
        # Use one name for all nodes unless use_different_names is True
        num_nodes = len(node_ids)
        all_names = self.name_gen.generate_names(num_nodes if use_different_names else 1)
        all_attributes = self.name_gen.generate_attributes(num_nodes)
        sorted_nodes = sorted(node_ids)
        id_to_attr = {node_id: all_attributes[i] for i, node_id in enumerate(sorted_nodes)}
        name = all_names[0]

        # Facts in NL
        lines = generate_predicate_lines(edges, id_to_attr, name)

        # --- Build the question text ---
        X, Y = query
        start_attr = id_to_attr[X]
        goal_attr = id_to_attr[Y]

        # If we have any prefix path tokens, show the exact visited chain
        if filtered_path:
            # Start from the first actual path node's attribute (this equals X in well-formed data)
            path_text = f"{name} is {id_to_attr[filtered_path[0]]}"
            for node_id in filtered_path[1:]:
                path_text += f". {name} is {id_to_attr[node_id]}"
            question = (f"Given that {name} is {start_attr}, and we want to prove "
                        f"{name} is {goal_attr}.\n\nProof: {path_text}. "
                        f"{name} is")
        else:
            question = (f"Given that {name} is {start_attr}, and we want to prove "
                        f"{name} is {goal_attr}.\n\n"
                        f"Proof: {name} is")

        # --- Sanity checks: neighbors vs outputs/label ---
        current_node = filtered_path[-1] if filtered_path else X
        one_hop_neighbors: Set[int] = {b for (a, b) in edges if a == current_node}

        if self.debug:
            # Label should be one-hop from current
            if label_id is not None and label_id not in one_hop_neighbors:
                print(f"[SEARCH][WARN] Label {label_id} is not a one-hop neighbor of "
                      f"current node {current_node}. Neighbors: {sorted(one_hop_neighbors)}")

            # Every output should be one-hop next step (C++ generator guarantees this)
            invalid_outputs = [o for o in outputs if o not in one_hop_neighbors]
            if invalid_outputs:
                print(f"[SEARCH][WARN] Outputs not one-hop from current node {current_node}: "
                      f"{invalid_outputs}. Neighbors: {sorted(one_hop_neighbors)}")

        # Map outputs (IDs) to their attribute strings
        answer_attrs = [id_to_attr[o] for o in outputs]

        # Compose final NL input
        facts = " ".join(lines)
        input_text = facts + " " + question

        return NLExample(
            input_text=input_text,
            output_texts=answer_attrs,
            labels=outputs,
            output_vector=output_vector,
            metadata={'node_mapping': id_to_attr}
        )

    def _generate_dfs_nl(self, graph_data: Dict, show_backtracking: bool = False) -> NLExample:
        """Convert DFS task to natural language - properly handles backtracking

        Args:
            graph_data: Dictionary containing edges, query, path, outputs, etc.
            show_backtracking: If True, explicitly mention backtracking steps in the trace
        """
        edges = graph_data['edges']
        query = graph_data['query']
        path = graph_data['path']
        outputs = graph_data['outputs']  # Now a list
        output_vector = graph_data.get('output_vector')

        if self.debug:
            print("\n_generate_dfs_nl input:")
            print(f"  Edges: {edges}")
            print(f"  Query: {query}")
            print(f"  Path: {path}")
            print(f"  Outputs: {outputs}")

        # Filter out special tokens from path (anything > MAX_NODE_ID)
        filtered_path = [node for node in path if node <= self.MAX_NODE_ID]

        # Collect all unique valid node IDs
        node_ids = set()
        for (a, b) in edges:
            node_ids.add(a)
            node_ids.add(b)
        if query:
            node_ids.add(query[0])
            node_ids.add(query[1])
        for node in filtered_path:
            node_ids.add(node)
        for output_node in outputs:
            node_ids.add(output_node)

        # Generate attributes
        name = self.name_gen.generate_names(1)[0]
        attributes = self.name_gen.generate_attributes(len(node_ids))
        sorted_nodes = sorted(node_ids)
        id_to_attr = {node_id: attributes[i] for i, node_id in enumerate(sorted_nodes)}

        # Generate facts using shared template function
        lines = generate_predicate_lines(edges, id_to_attr, name)

        # Build logical exploration text
        X, Y = query
        start_attr = id_to_attr[X]
        goal_attr = id_to_attr[Y]
        base_text = f"Given {name} is {start_attr}, we want to prove {name} is {goal_attr}."

        if len(filtered_path) > 1:
            # Build edges lookup for quick checking
            edges_set = set((a, b) for a, b in edges)

            # Build the reasoning chain accounting for backtracking
            reasoning_chain = []
            last_parent = None  # Track the last parent we transitioned from

            for i in range(1, len(filtered_path)):
                current_node = filtered_path[i]

                # Find the actual parent node (accounting for backtracking)
                parent_node = None

                # First check if previous node in path has an edge to current
                if (filtered_path[i - 1], current_node) in edges_set:
                    parent_node = filtered_path[i - 1]
                else:
                    # Backtracking occurred - find an earlier node with edge to current
                    for j in range(i - 2, -1, -1):
                        if (filtered_path[j], current_node) in edges_set:
                            parent_node = filtered_path[j]
                            break

                if parent_node is not None:
                    # Add backtracking statement if we backtracked and show_backtracking is True
                    if show_backtracking and last_parent is not None and parent_node != filtered_path[i - 1]:
                        # We backtracked from the previous node to parent_node
                        backtrack_attr = id_to_attr[parent_node]
                        reasoning_chain.append(f"Backtrack to {name} is {backtrack_attr}.")

                    parent_attr = id_to_attr[parent_node]
                    current_attr = id_to_attr[current_node]
                    reasoning_chain.append(
                        f"Since {name} is {parent_attr}, then {name} is {current_attr}.")
                    last_parent = parent_node
                else:
                    # This shouldn't happen in valid DFS, but handle gracefully
                    if self.debug:
                        print(f"Warning: No parent found for node {current_node} in DFS path")

            # Join all reasoning steps
            if reasoning_chain:
                chain_text = " ".join(reasoning_chain)
                # Ask for the next step from the last visited node
                current_attr = id_to_attr[filtered_path[-1]]
                question = f"\n\nProof: {chain_text} Since {name} is {current_attr}, then {name} is"
            else:
                # No valid chain could be built
                question = f"\n\nProof: Since {name} is {start_attr}, then {name} is"
        else:
            # First step - path only contains start node or is empty
            question = f"\n\nProof: Since {name} is {start_attr}, then {name} is"

        # Create as single paragraph
        facts = " ".join(lines)
        input_text = facts + " " + base_text + " " + question

        # Get all answer attributes
        output_texts = [id_to_attr[output_id] for output_id in outputs]

        return NLExample(
            input_text=input_text,
            output_texts=output_texts,
            labels=outputs,
            output_vector=output_vector,
            metadata={'node_mapping': {str(k): v for k, v in id_to_attr.items()}}
        )

    def _generate_si_nl(self, graph_data: Dict, is_selection: bool = None) -> NLExample:
        """Convert SI task to natural language - direct translation of symbolic format"""
        edges = graph_data['edges']
        query = graph_data['query']
        path = graph_data['path']
        outputs = graph_data['outputs']
        output_vector = graph_data.get('output_vector')

        # Filter out special tokens from path (anything > MAX_NODE_ID)
        filtered_path = [node for node in path if node <= self.MAX_NODE_ID]

        # Determine task type based on path structure if not specified
        if is_selection is None:
            if len(path) > 0 and path[-1] > self.MAX_NODE_ID:
                is_selection = True
            else:
                is_selection = False

        # Collect all unique valid node IDs
        node_ids = set()
        for (a, b) in edges:
            node_ids.add(a)
            node_ids.add(b)
        if query:
            node_ids.add(query[0])
            node_ids.add(query[1])
        for node in filtered_path:
            node_ids.add(node)
        for output_node in outputs:
            node_ids.add(output_node)

        # Generate attributes
        name = self.name_gen.generate_names(1)[0]
        attributes = self.name_gen.generate_attributes(len(node_ids))
        sorted_nodes = sorted(node_ids)
        id_to_attr = {node_id: attributes[i] for i, node_id in enumerate(sorted_nodes)}

        # Generate facts
        lines = generate_predicate_lines(edges, id_to_attr, name)

        # Build base text
        X, Y = query
        start_attr = id_to_attr[X]
        goal_attr = id_to_attr[Y]
        base_text = f"Given {name} is {start_attr}, we want to prove {name} is {goal_attr}."

        # Directly translate the path pairs into reasoning steps
        proof_text = "\n\nProof: "
        if len(filtered_path) >= 2:
            reasoning_steps = []
            for i in range(0, len(filtered_path) - 1, 2):
                from_node = filtered_path[i]
                to_node = filtered_path[i + 1]
                from_attr = id_to_attr[from_node]
                to_attr = id_to_attr[to_node]
                reasoning_steps.append(f"Since {name} is {from_attr}, then {name} is {to_attr}")

            if reasoning_steps:
                trace_text = ". ".join(reasoning_steps) + "."
                proof_text += trace_text

        # Build the question
        facts = " ".join(lines)
        if is_selection:
            question = f" Since {name} is"
            input_text = facts + " " + base_text + proof_text + question
        else:
            current_node = X if len(filtered_path) == 0 else filtered_path[-1]
            current_attr = id_to_attr[current_node]
            question = f" Since {name} is {current_attr}, then {name} is"
            input_text = facts + " " + base_text + proof_text + question

        # Get answer attributes
        output_texts = [id_to_attr[output_id] for output_id in outputs]

        return NLExample(
            input_text=input_text,
            output_texts=output_texts,
            labels=outputs,
            output_vector=output_vector,
            metadata={
                'node_mapping': {str(k): v for k, v in id_to_attr.items()},
                'is_selection': is_selection
            }
        )

    def generate_batch(self, task: str, batch_size: int, **kwargs) -> List[NLExample]:
        """Generate a batch of natural language examples"""

        show_backtracking = kwargs.pop('show_backtracking', False)

        # Generate symbolic data first
        if task == 'search':
            max_lookahead = kwargs.get('max_lookahead', 5)
            max_edges = kwargs.get('max_edges', (self.max_input_size - 5) // 3)
            reserved_inputs = kwargs.get('reserved_inputs', set())
            distance_from_start = kwargs.get('distance_from_start', -1)
            max_prefix_vertices = kwargs.get('max_prefix_vertices', self.max_input_size)
            alpha = kwargs.get('alpha', 1.0)

            inputs, outputs, labels, _ = generator.generate_training_set(
                self.max_input_size, batch_size, max_lookahead,
                max_edges, reserved_inputs, distance_from_start,
                max_prefix_vertices, True, alpha
            )

        elif task == 'dfs':
            requested_backtrack = kwargs.get('requested_backtrack', -1)
            reserved_inputs = kwargs.get('reserved_inputs', set())
            alpha = kwargs.get('alpha', 1.0)

            try:
                inputs, outputs, labels, _ = generator.generate_dfs_training_set(
                    self.max_input_size, batch_size, reserved_inputs,
                    requested_backtrack, False, True, True, alpha
                )
            except TypeError:
                # Fallback without alpha
                print("Note: Your compiled generator doesn't support alpha for DFS. Recompile generator.cpp.")
                inputs, outputs, labels, _ = generator.generate_dfs_training_set(
                    self.max_input_size, batch_size, reserved_inputs,
                    requested_backtrack, False, True, True
                )

        elif task == 'si':
            max_frontier_size = kwargs.get('max_frontier_size', 5)
            max_branch_size = kwargs.get('max_branch_size', 5)
            reserved_inputs = kwargs.get('reserved_inputs', set())
            alpha = kwargs.get('alpha', 1.0)
            sample_type = kwargs.get('sample_type', 0)

            inputs, outputs, labels, _ = generator.generate_si_training_set(
                self.max_input_size, batch_size, reserved_inputs,
                max_frontier_size, max_branch_size, True, True,
                alpha, sample_type
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        # Parse symbolic outputs
        parsed_examples = self._parse_symbolic_output(inputs, outputs, labels)

        # Convert to natural language
        nl_examples = []
        for ex in parsed_examples:
            if task == 'search':
                nl_ex = self._generate_search_nl(ex, kwargs.get('use_different_names', False))
            elif task == 'dfs':
                nl_ex = self._generate_dfs_nl(ex, show_backtracking)
            elif task == 'si':
                if kwargs.get('sample_type') == 1:
                    is_selection = True
                elif kwargs.get('sample_type') == 2:
                    is_selection = False
                else:
                    is_selection = None
                nl_ex = self._generate_si_nl(ex, is_selection)

            nl_examples.append(nl_ex)

        return nl_examples

    def generate_batch_with_symbolic(self, task: str, batch_size: int, **kwargs) -> Tuple[List[NLExample], np.ndarray]:
        """Generate batch and also return symbolic representations for reservation"""

        # Generate symbolic data first
        if task == 'search':
            max_lookahead = kwargs.get('max_lookahead', 5)
            max_edges = kwargs.get('max_edges', (self.max_input_size - 5) // 3)
            reserved_inputs = kwargs.get('reserved_inputs', set())
            distance_from_start = kwargs.get('distance_from_start', -1)
            max_prefix_vertices = kwargs.get('max_prefix_vertices', self.max_input_size)
            alpha = kwargs.get('alpha', 1.0)

            inputs, outputs, labels, _ = generator.generate_training_set(
                self.max_input_size, batch_size, max_lookahead,
                max_edges, reserved_inputs, distance_from_start,
                max_prefix_vertices, True, alpha
            )

        elif task == 'dfs':
            requested_backtrack = kwargs.get('requested_backtrack', -1)
            reserved_inputs = kwargs.get('reserved_inputs', set())
            alpha = kwargs.get('alpha', 1.0)

            try:
                inputs, outputs, labels, _ = generator.generate_dfs_training_set(
                    self.max_input_size, batch_size, reserved_inputs,
                    requested_backtrack, False, True, True, alpha
                )
            except TypeError:
                inputs, outputs, labels, _ = generator.generate_dfs_training_set(
                    self.max_input_size, batch_size, reserved_inputs,
                    requested_backtrack, False, True, True
                )

        elif task == 'si':
            max_frontier_size = kwargs.get('max_frontier_size', 5)
            max_branch_size = kwargs.get('max_branch_size', 5)
            reserved_inputs = kwargs.get('reserved_inputs', set())
            alpha = kwargs.get('alpha', 1.0)
            sample_type = kwargs.get('sample_type', 0)

            inputs, outputs, labels, _ = generator.generate_si_training_set(
                self.max_input_size, batch_size, reserved_inputs,
                max_frontier_size, max_branch_size, True, True,
                alpha, sample_type
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        # Parse and convert to NL
        parsed_examples = self._parse_symbolic_output(inputs, outputs, labels)
        nl_examples = []

        for ex in parsed_examples:
            if task == 'search':
                nl_ex = self._generate_search_nl(ex, kwargs.get('use_different_names', False))
            elif task == 'dfs':
                nl_ex = self._generate_dfs_nl(ex)
            elif task == 'si':
                if kwargs.get('sample_type') == 1:
                    is_selection = True
                elif kwargs.get('sample_type') == 2:
                    is_selection = False
                else:
                    is_selection = None
                nl_ex = self._generate_si_nl(ex, is_selection)
            nl_examples.append(nl_ex)

        return nl_examples, inputs


# Modified dataset and generation functions to handle multiple outputs
class StreamingNLDataset(IterableDataset):
    """Streaming dataset for natural language graph tasks"""

    def __init__(self, task: str, max_input_size: int, batch_size: int,
                 seed_offset: int = 0, reserved_inputs: Set = None, **task_kwargs):
        super().__init__()
        self.task = task
        self.max_input_size = max_input_size
        self.batch_size = batch_size
        self.seed_offset = seed_offset
        self.reserved_inputs = reserved_inputs if reserved_inputs is not None else set()
        self.task_kwargs = task_kwargs

        # For managing seeds across workers
        self.seed_generator = Random(seed_offset)
        self.seed_generator_lock = Lock()
        self.seed_values = []

    def get_seed(self, index):
        if index < len(self.seed_values):
            return self.seed_values[index]

        self.seed_generator_lock.acquire()
        while index >= len(self.seed_values):
            self.seed_values.append(self.seed_generator.randrange(2 ** 32))
        self.seed_generator_lock.release()

        return self.seed_values[index]

    def process_data(self, start_idx):
        current = start_idx
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        while True:
            # Get seed for this batch
            seed = self.get_seed(current)

            # Create generator with this seed
            gen = NaturalLanguageGraphGenerator(self.max_input_size, seed)

            # Generate batch with reserved inputs
            nl_examples = gen.generate_batch(
                self.task, self.batch_size,
                reserved_inputs=self.reserved_inputs,
                **self.task_kwargs
            )

            # Convert to format expected by training code
            inputs = [ex.input_text.replace('\n', ' ') for ex in nl_examples]
            outputs = [ex.output_texts for ex in nl_examples]  # Now list of lists
            labels = [ex.labels for ex in nl_examples]  # Now list of lists

            yield inputs, outputs, labels

            current += num_workers

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        return self.process_data(self.seed_offset + worker_id)


def generate_reserved_eval_set(task: str, max_input_size: int, n_eval_samples: int,
                               seed: int = 42, **task_kwargs) -> Tuple[
    List[str], List[List[str]], List[List[Any]], Set]:
    """Generate eval set and return both the data and the reserved symbolic inputs"""

    gen = NaturalLanguageGraphGenerator(max_input_size, seed=seed)

    # Generate eval samples without any reservations
    nl_examples, symbolic_inputs = gen.generate_batch_with_symbolic(
        task, n_eval_samples, reserved_inputs=set(), **task_kwargs
    )

    # Extract the symbolic inputs to reserve
    reserved_inputs = set()
    for i in range(symbolic_inputs.shape[0]):
        # Convert input array to tuple so it's hashable
        input_tuple = tuple(int(x) for x in symbolic_inputs[i] if x != gen.PADDING_TOKEN)
        reserved_inputs.add(input_tuple)

    eval_inputs = [ex.input_text for ex in nl_examples]
    eval_outputs = [ex.output_texts for ex in nl_examples]  # List of lists
    eval_labels = [ex.labels for ex in nl_examples]  # List of lists
    eval_vectors = [ex.output_vector for ex in nl_examples]  # List of output vectors

    return eval_inputs, eval_outputs, eval_labels, reserved_inputs, eval_vectors


def generate_nl_dataset(task: str, max_input_size: int, num_samples: int,
                        seed: int = None, debug: bool = False,
                        reserved_inputs: Set = None,
                        show_backtracking: bool = False,  # New parameter
                        **task_kwargs) -> Tuple[
    List[str], List[List[str]], List[List[Any]]]:
    """Generate a fixed-size natural language dataset with optional reserved inputs

    Args:
        task: The task type ('search', 'dfs', or 'si')
        max_input_size: Maximum input size
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        debug: Whether to print debug information
        reserved_inputs: Set of reserved inputs to avoid
        show_backtracking: For DFS task, whether to explicitly show backtracking steps
        **task_kwargs: Additional task-specific parameters
    """
    gen = NaturalLanguageGraphGenerator(max_input_size, seed, debug)

    all_inputs = []
    all_outputs = []
    all_labels = []

    batch_size = min(1000, num_samples)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        nl_examples = gen.generate_batch(
            task, current_batch_size,
            reserved_inputs=reserved_inputs if reserved_inputs else set(),
            show_backtracking=show_backtracking,  # Pass it through
            **task_kwargs
        )

        all_inputs.extend([ex.input_text for ex in nl_examples])
        all_outputs.extend([ex.output_texts for ex in nl_examples])  # List of lists
        all_labels.extend([ex.labels for ex in nl_examples])  # List of lists

        if (i + 1) % 10 == 0:
            print(f"Generated {(i + 1) * batch_size} / {num_samples} examples")

    return all_inputs, all_outputs, all_labels


if __name__ == "__main__":
    print("Natural Language Graph Generator - Testing Multiple Outputs")
    print("=" * 60)

    # Test with debug mode to see symbolic output
    print("\n1. SEARCH TASK:")
    print("-" * 40)
    inputs, outputs, labels = generate_nl_dataset('search', 100, 9, max_lookahead=10, seed=42, debug=True)
    for i in range(len(inputs)):
        # if not len(labels[i]) > 1:
        #     continue
        print(f"\nExample {i + 1}:")
        print(f"Input: {inputs[i]}")
        print(f"Correct outputs: {outputs[i]}")  # Now shows list of correct answers
        print(f"Labels (node IDs): {labels[i]}")

    # print("\n" + "=" * 60)
    # print("\n2. DFS TASK:")
    # print("-" * 40)
    # inputs, outputs, labels = generate_nl_dataset('dfs', 256,20, requested_backtrack=3, seed=42, debug=True)
    # for i in range(len(inputs)):
    #     if not len(labels[i]) > 1:
    #         continue
    #     print(f"\nExample {i + 1}:")
    #     print(f"Input: {inputs[i]}")
    #     print(f"Correct outputs: {outputs[i]}")  # Now shows list of correct answers
    #     print(f"Labels (node IDs): {labels[i]}")

    # print("\n" + "=" * 60)
    # print("\n3. SI TASK:")
    # print("-" * 40)
    # inputs, outputs, labels = generate_nl_dataset('si', 256, 20, max_frontier_size=12, max_branch_size=12, seed=42,
    #                                               debug=True)
    # for i in range(len(inputs)):
    #     # if not len(labels[i]) > 1:
    #     #     continue
    #     print(f"\nExample {i + 1}:")
    #     print(f"Input: {inputs[i]}")
    #     print(f"Correct outputs: {outputs[i]}")  # Now shows list of correct answers
    #     print(f"Labels (node IDs): {labels[i]}")