"""Data schema definitions for annotations and model outputs."""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid
import jsonschema


@dataclass
class EvidenceSpan:
    """Evidence span linking a premise to a document."""
    doc_id: str
    start: int
    end: int
    text: Optional[str] = None


@dataclass
class Premise:
    """A single premise."""
    id: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text
        }


@dataclass
class Annotation:
    """Complete annotation with premises, content, and metadata.

    The ``content`` field stores the training target:

    * **Stage 0**: the final conclusion text.
    * **Stage 1**: the full proof trace with ``<PREMISE>``/``<CONCLUSION>``
      tags.
    * **Legacy / non-chain data**: a plain conclusion string.
    """
    id: str
    premises: List[Premise]
    content: str
    verifier_notes: Optional[str] = None
    annotator_id: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "premises": [p.to_dict() for p in self.premises],
            "content": self.content,
            "verifier_notes": self.verifier_notes,
            "annotator_id": self.annotator_id,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Annotation":
        """Create Annotation from dictionary.

        Accepts both ``"content"`` (preferred) and ``"conclusion"``
        (backward-compatible) as the training-target field.
        """
        premises = [
            Premise(
                id=p["id"],
                text=p["text"]
            ) for p in data["premises"]
        ]

        # Accept both "content" and legacy "conclusion".
        content_data = data.get("content") or data.get("conclusion", "")
        if isinstance(content_data, dict):
            content_text = content_data.get("text", "")
        else:
            content_text = content_data

        return cls(
            id=data["id"],
            premises=premises,
            content=content_text,
            verifier_notes=data.get("verifier_notes"),
            annotator_id=data.get("annotator_id"),
            timestamp=data.get("timestamp")
        )
    
    def to_jsonl(self) -> str:
        """Convert to JSONL string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_jsonl(cls, line: str) -> "Annotation":
        """Create Annotation from JSONL line."""
        return cls.from_dict(json.loads(line))


# JSON Schema for validation
ANNOTATION_SCHEMA = {
    "type": "object",
    "required": ["id", "premises"],
    "properties": {
        "id": {"type": "string"},
        "premises": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "text"],
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"}
                }
            }
        },
        "content": {"type": "string"},
        "conclusion": {"type": "string"},  # backward compat
        "verifier_notes": {"type": "string"},
        "annotator_id": {"type": "string"},
        "timestamp": {"type": "string"}
    }
}


def validate_annotation(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate annotation against schema."""
    try:
        jsonschema.validate(instance=data, schema=ANNOTATION_SCHEMA)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e)


def safe_parse_model_output(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Safely parse model output JSON, attempting repairs if needed."""
    import re
    
    # First attempt: direct JSON parsing
    try:
        obj = json.loads(text)
        jsonschema.validate(obj, ANNOTATION_SCHEMA)
        return obj, None
    except json.JSONDecodeError as e:
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                obj = json.loads(json_match.group(1))
                jsonschema.validate(obj, ANNOTATION_SCHEMA)
                return obj, None
            except:
                pass
        
        # Try single-pass repair: fix common issues
        repaired = text
        # Fix unescaped quotes in strings
        repaired = re.sub(r'(?<!\\)"(?![,\]\}])', '\\"', repaired)
        # Try parsing again
        try:
            obj = json.loads(repaired)
            jsonschema.validate(obj, ANNOTATION_SCHEMA)
            return obj, None
        except:
            pass
    
    except jsonschema.ValidationError as e:
        return None, f"Schema validation error: {str(e)}"
    
    return None, f"Failed to parse JSON: {str(e)}"


def format_prompt(context: str, question: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
    """Format prompt for training or inference."""
    if system_prompt is None:
        system_prompt = """You must respond only in valid JSON format. Your response must contain exactly two fields:
1. "premises": an array of concise factual statements (premises) that support your content
2. "content": the conclusion or proof trace that follows logically from the premises

Each premise should be a factual statement that can be verified. Link premises to evidence when available.
Ensure the content follows logically from all the premises provided."""
    
    user_part = f"Question: {question}\n" if question else ""
    user_part += f"Context: {context}"
    
    return f"System: {system_prompt}\n\nUser: {user_part}\n\nAssistant:"

