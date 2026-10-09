"""
Natural Language Generator for Graph Tasks
"""

import os
import sys
import sysconfig
from io import StringIO
import random
import numpy as np
from typing import Dict, List, Any, Set
from dataclasses import dataclass

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


class NameAttributeGenerator:
    """Generates consistent names and attributes for nodes"""

    def __init__(self, seed=None):
        self.fake = Faker()
        if seed is not None:
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

    # ---- fixed vertex-ID -> attribute-name dictionary (entity-vocabulary curriculum) ----
    # One deterministic list of attribute names (seed POOL_SEED, generated sequentially with
    # duplicates skipped, so entry i is the same in every worker and rank). When id-mapping is
    # active, symbolic vertex ID i always renders as name_for_id(i). The symbolic generator draws
    # each instance's IDs uniformly from 1..max_vertex_id (ID 0 is reserved), and max_vertex_id =
    # max_edges + 1 grows with the curriculum stage (2(L+1) names at stage L) or, with vocab_pool=fixed,
    # stays at the context maximum ((n-5)//3 + 1 names, 255 for n=768). The active vocabulary is
    # therefore exactly that ID range.
    POOL_SEED = 20260907
    _pool = None

    def _attribute_with(self, rng) -> str:
        vowels = "aeiou"; consonants = "bcdfghjklmnpqrstvwxyz"
        patterns = ["CV", "VC", "CVC", "CVV", "CCV", "VCV", "VCC"]
        out = ""
        for _ in range(2):
            for ch in rng.choice(patterns):
                out += rng.choice(consonants) if ch == "C" else rng.choice(vowels)
        return out

    def _ensure_pool(self, n: int):
        if self._pool is not None and len(self._pool) >= n:
            return
        rng = random.Random(self.POOL_SEED)
        pool, seen = [], set()
        while len(pool) < n:
            a = self._attribute_with(rng)
            if a not in seen:
                seen.add(a); pool.append(a)
        self._pool = pool

    def name_for_id(self, vertex_id: int) -> str:
        self._ensure_pool(int(vertex_id) + 1)
        return self._pool[int(vertex_id)]


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
        if seed is not None:
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

    def _generate_search_nl(self, graph_data: Dict) -> NLExample:
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
        # One person name for the whole instance
        num_nodes = len(node_ids)
        all_names = self.name_gen.generate_names(1)
        if getattr(self, "id_name_map", False):
            # entity-vocabulary mode: vertex ID -> fixed attribute name
            id_to_attr = {node_id: self.name_gen.name_for_id(node_id) for node_id in node_ids}
        else:
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
        )

    def generate_batch(self, task: str, batch_size: int, **kwargs) -> List[NLExample]:
        """Generate a batch of natural language search examples (search is the only task; the dfs/si paths
        were removed 2026-10-09)."""
        if task != 'search':
            raise ValueError(f"Unknown task: {task}")

        max_lookahead = kwargs.get('max_lookahead', 5)
        reserved_inputs = kwargs.get('reserved_inputs', set())
        alpha = kwargs.get('alpha', 1.0)
        # Former generate_batch kwargs that no caller ever passed (removed 2026-10-09); the C++ call keeps
        # their old default values.
        max_edges = (self.max_input_size - 5) // 3
        distance_from_start = -1
        max_prefix_vertices = self.max_input_size

        vocab_pool = kwargs.get('vocab_pool', 'none') or 'none'
        if vocab_pool not in ('none', 'grow', 'fixed'):
            raise ValueError(f"unknown vocab_pool mode {vocab_pool!r}")
        self.id_name_map = (vocab_pool != 'none')          # fixed ID->name dictionary
        # Pin the ID range at its maximum only under vocab_pool=fixed (generator.cpp's last argument, fixed_vocab).
        pin_vocab = (vocab_pool == 'fixed')
        try:
            inputs, outputs, labels, _ = generator.generate_training_set(
                self.max_input_size, batch_size, max_lookahead,
                max_edges, reserved_inputs, distance_from_start,
                max_prefix_vertices, True, alpha, pin_vocab
            )
        except TypeError:
            # Older compiled generator without the pinned-ID-range argument.
            if pin_vocab:
                print("WARNING: compiled generator lacks the pinned-ID-range argument (vocab_pool=fixed); falling back to default. Recompile generator.cpp.")
            inputs, outputs, labels, _ = generator.generate_training_set(
                self.max_input_size, batch_size, max_lookahead,
                max_edges, reserved_inputs, distance_from_start,
                max_prefix_vertices, True, alpha
            )

        # Parse symbolic outputs, then convert to natural language
        parsed_examples = self._parse_symbolic_output(inputs, outputs, labels)
        return [self._generate_search_nl(ex) for ex in parsed_examples]


if __name__ == "__main__":
    # Smoke test: a few search examples at a small lookahead.
    gen = NaturalLanguageGraphGenerator(256, seed=42)
    for ex in gen.generate_batch("search", batch_size=3, max_lookahead=4, alpha=0.5):
        print(f"\nInput: {ex.input_text}\nCorrect outputs: {ex.output_texts}\nLabels (node IDs): {ex.labels}")