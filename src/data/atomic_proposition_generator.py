"""Generate atomic propositions using OpenAI API with pooling."""

import os
import json
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

try:
    import openai
except ImportError:
    openai = None


@dataclass
class AtomicProposition:
    """A single atomic proposition."""
    text: str
    topic: str
    is_predicate: bool = False  # True if it's a predicate like "is tall", False if full statement


@dataclass
class EntityPool:
    """Pool of entity names for first-order logic."""
    names: List[str] = field(default_factory=list)

    def sample(self, n: int = 1, exclude: Optional[List[str]] = None) -> List[str]:
        """Sample n distinct entities, excluding specified ones."""
        available = [e for e in self.names if not exclude or e not in exclude]
        return random.sample(available, min(n, len(available)))


@dataclass
class PropositionPool:
    """Pool of atomic propositions for efficient generation."""
    # Full propositions for propositional logic (e.g., "it is raining")
    propositions: List[AtomicProposition] = field(default_factory=list)
    # Predicates for first-order logic (e.g., "tall", "red", "made of metal")
    predicates: List[AtomicProposition] = field(default_factory=list)
    # Relations for first-order logic (e.g., "loves", "is taller than")
    relations: List[AtomicProposition] = field(default_factory=list)
    # Categories/types (e.g., "mammals", "vehicles", "fruits")
    categories: List[AtomicProposition] = field(default_factory=list)
    # Entity names
    entities: EntityPool = field(default_factory=EntityPool)

    def sample_propositions(self, n: int, exclude: Optional[List[str]] = None) -> List[AtomicProposition]:
        """Sample n distinct propositions."""
        available = [p for p in self.propositions if not exclude or p.text not in exclude]
        return random.sample(available, min(n, len(available)))

    def sample_predicates(self, n: int, exclude: Optional[List[str]] = None) -> List[AtomicProposition]:
        """Sample n distinct predicates."""
        available = [p for p in self.predicates if not exclude or p.text not in exclude]
        return random.sample(available, min(n, len(available)))

    def sample_relations(self, n: int = 1) -> List[AtomicProposition]:
        """Sample n distinct relations."""
        return random.sample(self.relations, min(n, len(self.relations)))

    def sample_categories(self, n: int = 1) -> List[AtomicProposition]:
        """Sample n distinct categories."""
        return random.sample(self.categories, min(n, len(self.categories)))

    # -- Serialization for caching and cross-process transfer ----------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pool to a plain dict (JSON-safe)."""
        def _ap_list(items: List[AtomicProposition]) -> List[Dict[str, Any]]:
            return [{"text": a.text, "topic": a.topic, "is_predicate": a.is_predicate}
                    for a in items]
        return {
            "propositions": _ap_list(self.propositions),
            "predicates": _ap_list(self.predicates),
            "relations": _ap_list(self.relations),
            "categories": _ap_list(self.categories),
            "entities": self.entities.names,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PropositionPool":
        """Reconstruct a pool from a dict produced by ``to_dict``."""
        def _parse_ap(items: List[Dict[str, Any]]) -> List[AtomicProposition]:
            return [AtomicProposition(text=i["text"], topic=i.get("topic", ""),
                                     is_predicate=i.get("is_predicate", False))
                    for i in items]
        return cls(
            propositions=_parse_ap(d.get("propositions", [])),
            predicates=_parse_ap(d.get("predicates", [])),
            relations=_parse_ap(d.get("relations", [])),
            categories=_parse_ap(d.get("categories", [])),
            entities=EntityPool(names=d.get("entities", [])),
        )

    def save(self, path: str) -> None:
        """Save pool to a JSON file."""
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "PropositionPool":
        """Load pool from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


@dataclass
class GeneratorConfig:
    """Configuration for atomic proposition generation."""
    api_key: Optional[str] = None
    model: str = "gpt-5.2-2025-12-11"
    max_tokens: int = 16384
    temperature: float = 0.9
    retry_attempts: int = 3
    retry_delay: float = 1.0
    # Pool sizes
    propositions_per_topic: int = 20
    predicates_per_topic: int = 15
    relations_count: int = 30
    categories_count: int = 20
    entities_count: int = 50


# Topic categories for diverse generation
TOPIC_CATEGORIES = [
    "everyday life and routines",
    "science and nature",
    "technology and computing",
    "sports and fitness",
    "food and cooking",
    "travel and geography",
    "history and culture",
    "business and economics",
    "health and medicine",
    "arts and entertainment",
    "education and learning",
    "animals and wildlife",
    "weather and climate",
    "transportation and vehicles",
    "relationships and social dynamics",
]


class AtomicPropositionGenerator:
    """Generate pools of atomic propositions using OpenAI API."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.client = None
        self.pool: Optional[PropositionPool] = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client."""
        if openai is None:
            raise ImportError("openai package not installed. Run: pip install openai")

        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Set via config or environment variable.")

        self.client = openai.OpenAI(api_key=api_key)

    def generate_pool(
        self,
        topics: Optional[List[str]] = None,
        verbose: bool = True,
        skip_categories: bool = False,
    ) -> PropositionPool:
        """
        Generate a pool of atomic propositions with a single API call per category.

        This makes only a few API calls total, then all examples are generated
        by sampling from this pool.

        Args:
            topics: Topic categories to generate propositions for.
            verbose: Print progress messages.
            skip_categories: If True, skip generating categories (saves one
                API call).  Categories are only used by template-based
                inference generation, not by the chain generator.
        """
        topics = topics or TOPIC_CATEGORIES
        pool = PropositionPool()

        if verbose:
            print("Generating proposition pool...")

        # All generation calls are independent — run them concurrently.
        tasks = {
            "propositions": ("Generating propositions...", lambda: self._generate_propositions(topics)),
            "predicates": ("Generating predicates...", lambda: self._generate_predicates(topics)),
            "relations": ("Generating relations...", lambda: self._generate_relations()),
            "entities": ("Generating entity names...", lambda: self._generate_entities()),
        }
        if not skip_categories:
            tasks["categories"] = ("Generating categories...", lambda: self._generate_categories())

        try:
            results: Dict[str, Any] = {}
            if verbose:
                print("  Submitting API calls in parallel...")
            with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                future_to_key = {
                    executor.submit(fn): key
                    for key, (_desc, fn) in tasks.items()
                }
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    results[key] = future.result()
                    desc = tasks[key][0]
                    count = len(results[key])
                    if verbose:
                        print(f"    {desc} done ({count} items)")
        except Exception as exc:
            # Fall back to sequential on any thread-pool failure.
            if verbose:
                print(f"  Parallel generation failed ({exc}), falling back to sequential...")
            results = {}
            for key, (desc, fn) in tasks.items():
                if verbose:
                    print(f"  {desc}")
                results[key] = fn()
                if verbose:
                    print(f"    Generated {len(results[key])} items")

        pool.propositions = results["propositions"]
        pool.predicates = results["predicates"]
        pool.relations = results["relations"]
        pool.entities = EntityPool(names=results["entities"])
        if "categories" in results:
            pool.categories = results["categories"]

        self.pool = pool

        # Validate that the pool has content.
        total = (len(pool.propositions) + len(pool.predicates)
                 + len(pool.relations) + len(pool.entities.names))
        if total == 0:
            raise RuntimeError(
                "Pool generation produced no content. Check the API key, "
                "model name, and network connectivity."
            )

        if verbose:
            print("Pool generation complete!")

        return pool

    def _generate_propositions(self, topics: List[str]) -> List[AtomicProposition]:
        """Generate full propositions for propositional logic."""
        prompt = f"""Generate a diverse list of simple, atomic propositions that can be true or false.
These will be used for logic reasoning examples.

Topics to cover: {', '.join(topics[:5])}

Requirements:
- Each proposition should be a simple declarative statement
- Start each proposition with a lowercase letter — do not capitalise the first word unless it is a proper noun (e.g., "the sun is shining" not "The sun is shining", but "Paris is beautiful" is fine)
- Propositions should be concrete and specific, not abstract
- Include cause-effect pairs (e.g., "it is raining" and "the ground is wet")
- Keep each proposition under 10 words
- Generate at least {self.config.propositions_per_topic * len(topics[:5])} propositions

Respond with a JSON array of strings only:
["it is raining", "the ground is wet", ...]"""

        content = self._call_api(prompt)
        propositions = self._parse_string_array(content)

        return [
            AtomicProposition(text=p, topic="mixed", is_predicate=False)
            for p in propositions
        ]

    def _generate_predicates(self, topics: List[str]) -> List[AtomicProposition]:
        """Generate predicates for first-order logic (e.g., 'tall', 'red', 'edible')."""
        prompt = f"""Generate a diverse list of predicates (adjectives, past participles, or noun phrases) that can describe things.
These will be used in first-order logic statements like "X is [predicate]", so they MUST be grammatically correct after "is".

Topics to cover: {', '.join(topics[:5])}

Requirements:
- ONLY use adjectives, past participles, or noun phrases — these must sound natural after "is"
- Do NOT include "is" as a prefix — write "tall" not "is tall"
- Do NOT use present-tense or base-form verbs — "absorbs water", "requires a helmet", "eats food", "runs fast" are ALL invalid
- Valid forms: adjectives ("tall", "flexible", "edible"), past participles ("made of metal", "broken", "frozen"), noun phrases ("a mammal", "a solid"), ability phrases ("able to fly")
- Start each predicate with a lowercase letter
- Generate predicates that can form logical chains (e.g., "made of metal" implies "a conductor")
- Keep each predicate under 5 words
- Generate at least {self.config.predicates_per_topic * len(topics[:5])} predicates

Respond with a JSON array of strings only:
["tall", "made of metal", "edible", "flexible", "able to fly", ...]"""

        content = self._call_api(prompt)
        predicates = self._parse_string_array(content)

        # Strip leading "is " to avoid double-"is" when rendered with the
        # template "{entity} is {predicate}".
        cleaned = []
        for p in predicates:
            if p.lower().startswith("is "):
                p = p[3:]
            cleaned.append(p)

        # Filter out verb phrases: present-tense verbs (e.g. "absorbs water",
        # "requires a helmet") are invalid as "{entity} is {predicate}".
        # Heuristic: first word ends in -s/-es but NOT in adjective suffixes.
        _ADJ_SUFFIXES = ("ous", "ious", "eous", "ness", "less", "ful", "ble",
                         "ive", "ent", "ant", "ous", "ss")
        valid = []
        for p in cleaned:
            first = p.split()[0].lower() if p else ""
            if (first.endswith("s")
                    and not any(first.endswith(sfx) for sfx in _ADJ_SUFFIXES)):
                continue  # likely a present-tense verb phrase — skip
            valid.append(p)

        return [
            AtomicProposition(text=p, topic="mixed", is_predicate=True)
            for p in valid
        ]

    def _generate_relations(self) -> List[AtomicProposition]:
        """Generate relations for relational reasoning (e.g., 'loves', 'is taller than')."""
        prompt = f"""Generate a list of binary relations that can hold between two entities.
These will be used in statements like "A [relation] B".

Requirements:
- Include various types: social, physical, temporal, causal
- Each relation should be a verb or verb phrase
- Some should imply properties (e.g., "teaches" implies the first entity is knowledgeable)
- Generate at least {self.config.relations_count} relations

Respond with a JSON array of strings only:
["loves", "is taller than", "teaches", ...]"""

        content = self._call_api(prompt)
        relations = self._parse_string_array(content)

        return [
            AtomicProposition(text=r, topic="relations", is_predicate=False)
            for r in relations
        ]

    def _generate_categories(self) -> List[AtomicProposition]:
        """Generate categories for universal statements (e.g., 'mammals', 'vehicles')."""
        prompt = f"""Generate a list of category/type nouns for universal statements.
These will be used in statements like "All [category] are..." or "X is a [category]".

Requirements:
- Include natural kinds (animals, plants), artifacts (vehicles, tools), abstract categories
- Each should be a plural noun or noun phrase
- Generate at least {self.config.categories_count} categories

Respond with a JSON array of strings only:
["mammals", "vehicles", "students", ...]"""

        content = self._call_api(prompt)
        categories = self._parse_string_array(content)

        return [
            AtomicProposition(text=c, topic="categories", is_predicate=False)
            for c in categories
        ]

    def _generate_entities(self) -> List[str]:
        """Generate entity names for first-order logic."""
        prompt = f"""Generate a list of specific entity names for logic examples.
These will be used as specific instances in statements.

Requirements:
- Include person names (diverse), place names, object descriptions
- Mix of proper nouns and definite descriptions
- Use lowercase for common nouns and articles (e.g., "the red car", "the old house") — only capitalise proper nouns (e.g., "John", "Mount Everest", "Paris")
- Generate at least {self.config.entities_count} entity names

Respond with a JSON array of strings only:
["John", "the red car", "Paris", ...]"""

        content = self._call_api(prompt)
        return self._parse_string_array(content)

    def _call_api(self, prompt: str) -> str:
        """Make API call with retries."""
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    max_completion_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content
                if not content:
                    print(f"    Warning: API returned empty content "
                          f"(finish_reason={response.choices[0].finish_reason})")
                    return "[]"
                return content
            except Exception as e:
                if attempt < self.config.retry_attempts - 1:
                    print(f"    Warning: API call failed (attempt {attempt + 1}/"
                          f"{self.config.retry_attempts}): {e}")
                    time.sleep(self.config.retry_delay)
                else:
                    raise
        return "[]"

    def _parse_string_array(self, content: str) -> List[str]:
        """Parse JSON array of strings from API response.

        Strips trailing periods from each string so that propositions
        don't include sentence-ending punctuation (e.g. "it is raining."
        becomes "it is raining").
        """
        try:
            content = content.strip()
            # Find JSON array in content
            start_idx = content.find("[")
            end_idx = content.rfind("]") + 1
            if start_idx >= 0 and end_idx > start_idx:
                content = content[start_idx:end_idx]
            items = json.loads(content)
            return [s.rstrip(".") for s in items if isinstance(s, str)]
        except (json.JSONDecodeError, AttributeError):
            preview = repr(content[:200]) if content else repr(content)
            print(f"    Warning: failed to parse API response as JSON array: {preview}")
            return []

    def get_propositions_for_template(
        self,
        template_name: str,
        premise_slots: List[str],
        entity_slots: List[str],
        logic_type: str
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Get propositions for a specific template by sampling from the pool.

        Returns:
            (propositions_dict, entities_dict)
        """
        if self.pool is None:
            raise ValueError("Pool not generated. Call generate_pool() first.")

        propositions = {}
        entities = {}

        if logic_type == "propositional":
            # For propositional logic, sample full propositions
            sampled = self.pool.sample_propositions(len(premise_slots))
            for slot, prop in zip(premise_slots, sampled):
                propositions[slot] = prop.text
        else:
            # For first-order logic, handle based on template type
            if template_name == "universal_instantiation":
                # Need: X (category), P (predicate), a (entity)
                cat = self.pool.sample_categories(1)[0] if self.pool.categories else None
                pred = self.pool.sample_predicates(1)[0] if self.pool.predicates else None
                propositions["X"] = cat.text if cat else "things"
                propositions["P"] = pred.text if pred else "special"
            elif template_name == "relational_inference":
                # Need: R (relation), S (predicate)
                rel = self.pool.sample_relations(1)[0] if self.pool.relations else None
                pred = self.pool.sample_predicates(1)[0] if self.pool.predicates else None
                propositions["R"] = rel.text if rel else "knows"
                propositions["S"] = pred.text if pred else "wise"
            else:
                # Default: sample predicates for P, Q, R, etc.
                sampled = self.pool.sample_predicates(len(premise_slots))
                for slot, prop in zip(premise_slots, sampled):
                    propositions[slot] = prop.text

            # Sample entities
            if entity_slots:
                entity_names = self.pool.entities.sample(len(entity_slots))
                for slot, name in zip(entity_slots, entity_names):
                    entities[slot] = name

        return propositions, entities


# Fallback pool for when API is not available
def create_fallback_pool() -> PropositionPool:
    """Create a basic fallback pool without API calls."""
    pool = PropositionPool()

    # Basic propositions
    pool.propositions = [
        AtomicProposition("it is raining", "weather", False),
        AtomicProposition("the ground is wet", "weather", False),
        AtomicProposition("the sun is shining", "weather", False),
        AtomicProposition("it is cold outside", "weather", False),
        AtomicProposition("the alarm is ringing", "everyday", False),
        AtomicProposition("I wake up early", "everyday", False),
        AtomicProposition("the coffee is hot", "everyday", False),
        AtomicProposition("the door is open", "everyday", False),
        AtomicProposition("the light is on", "everyday", False),
        AtomicProposition("the car is running", "transport", False),
        AtomicProposition("the road is clear", "transport", False),
        AtomicProposition("traffic is heavy", "transport", False),
        AtomicProposition("the store is open", "everyday", False),
        AtomicProposition("the phone is charging", "tech", False),
        AtomicProposition("the battery is full", "tech", False),
        AtomicProposition("the wifi is connected", "tech", False),
        AtomicProposition("the water is boiling", "science", False),
        AtomicProposition("the ice is melting", "science", False),
        AtomicProposition("the plant is growing", "nature", False),
        AtomicProposition("the bird is singing", "nature", False),
    ]

    # Basic predicates
    pool.predicates = [
        AtomicProposition("tall", "physical", True),
        AtomicProposition("short", "physical", True),
        AtomicProposition("heavy", "physical", True),
        AtomicProposition("light", "physical", True),
        AtomicProposition("red", "color", True),
        AtomicProposition("blue", "color", True),
        AtomicProposition("made of metal", "material", True),
        AtomicProposition("made of wood", "material", True),
        AtomicProposition("edible", "property", True),
        AtomicProposition("dangerous", "property", True),
        AtomicProposition("valuable", "property", True),
        AtomicProposition("fragile", "property", True),
        AtomicProposition("warm-blooded", "biology", True),
        AtomicProposition("able to fly", "ability", True),
        AtomicProposition("able to swim", "ability", True),
    ]

    # Basic relations
    pool.relations = [
        AtomicProposition("loves", "social", False),
        AtomicProposition("knows", "social", False),
        AtomicProposition("teaches", "social", False),
        AtomicProposition("is taller than", "physical", False),
        AtomicProposition("is older than", "temporal", False),
        AtomicProposition("works with", "social", False),
        AtomicProposition("lives near", "spatial", False),
        AtomicProposition("owns", "possession", False),
    ]

    # Basic categories
    pool.categories = [
        AtomicProposition("mammals", "biology", False),
        AtomicProposition("birds", "biology", False),
        AtomicProposition("vehicles", "artifacts", False),
        AtomicProposition("fruits", "food", False),
        AtomicProposition("students", "social", False),
        AtomicProposition("teachers", "social", False),
        AtomicProposition("buildings", "artifacts", False),
        AtomicProposition("metals", "material", False),
    ]

    # Basic entities
    pool.entities = EntityPool(names=[
        "John", "Mary", "Alice", "Bob", "Charlie",
        "the red car", "the old house", "the big dog",
        "Paris", "London", "Tokyo", "New York",
        "Mount Everest", "the Pacific Ocean",
        "the morning sun", "the evening star",
    ])

    return pool
