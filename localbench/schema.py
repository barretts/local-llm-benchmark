"""Strict JSON/tool contracts shared by every engine and the safe executor.

Schema validation is not path authorization or a code sandbox.  Those checks
belong to the controller-owned executor, and can be supplied to measurements.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any


class ValidationError(ValueError):
    """An input does not satisfy a controller-owned contract."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValidationError(f"nonfinite JSON value: {value}")


def loads_json(value: str) -> Any:
    """Decode JSON, rejecting duplicate keys and JavaScript NaN/Infinity."""
    if not isinstance(value, str):
        raise ValidationError("JSON input must be text")
    result = json.loads(value, object_pairs_hook=_pairs, parse_constant=_constant)
    _json_value(result)
    return result


def _json_value(value: Any) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise ValidationError("value is not finite JSON data")


def validate(value: Any, contract: dict[str, Any], path: str = "arguments") -> Any:
    """Validate the small JSON Schema subset used by benchmark tools/answers.

    Unsupported keywords fail closed rather than pretending to validate them.
    Boolean is never accepted as an integer/number.  Unknown object fields are
    rejected by default; arbitrary answer objects explicitly opt in.
    """
    supported = {"type", "properties", "required", "additionalProperties", "items",
                 "enum", "const", "minimum", "maximum", "minLength", "maxLength",
                 "minItems", "maxItems", "description", "title"}
    unknown = set(contract) - supported
    if unknown:
        raise ValidationError(f"unsupported schema keywords: {sorted(unknown)}")
    _json_value(value)
    kind = contract.get("type")
    types = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: type(item) is int,
        "number": lambda item: type(item) is int or (type(item) is float and math.isfinite(item)),
        "boolean": lambda item: type(item) is bool,
        "null": lambda item: item is None,
    }
    if kind is not None:
        if kind not in types:
            raise ValidationError(f"unsupported schema type: {kind}")
        if not types[kind](value):
            raise ValidationError(f"{path} must be {kind}")
    if "enum" in contract and not any(type(value) is type(item) and value == item
                                      for item in contract["enum"]):
        raise ValidationError(f"{path} is outside enum")
    if "const" in contract and (type(value) is not type(contract["const"])
                                or value != contract["const"]):
        raise ValidationError(f"{path} does not match const")
    if isinstance(value, dict):
        properties = contract.get("properties", {})
        required = contract.get("required", [])
        missing = set(required) - set(value)
        if missing:
            raise ValidationError(f"{path} missing fields: {sorted(missing)}")
        extras = set(value) - set(properties)
        additional = contract.get("additionalProperties", False)
        if extras and additional is False:
            raise ValidationError(f"{path} unexpected fields: {sorted(extras)}")
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key], f"{path}.{key}")
            elif isinstance(additional, dict):
                validate(item, additional, f"{path}.{key}")
    if isinstance(value, list):
        for item in value:
            if "items" in contract:
                validate(item, contract["items"], f"{path}[]")
        for key, operator in (("minItems", lambda n, bound: n < bound),
                              ("maxItems", lambda n, bound: n > bound)):
            if key in contract and operator(len(value), contract[key]):
                raise ValidationError(f"{path} violates {key}")
    if isinstance(value, str):
        for key, operator in (("minLength", lambda n, bound: n < bound),
                              ("maxLength", lambda n, bound: n > bound)):
            if key in contract and operator(len(value), contract[key]):
                raise ValidationError(f"{path} violates {key}")
    if type(value) in (int, float):
        if "minimum" in contract and value < contract["minimum"]:
            raise ValidationError(f"{path} below minimum")
        if "maximum" in contract and value > contract["maximum"]:
            raise ValidationError(f"{path} above maximum")
    return value


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


_PATH = {"type": "string", "minLength": 1}
TOOL_SCHEMAS = {
    "list_files": _object({"path": _PATH}, []),
    "read_file": _object({"path": _PATH}, ["path"]),
    "write_file": _object({"path": _PATH, "content": {"type": "string"}},
                          ["path", "content"]),
    "apply_patch": _object({"path": _PATH, "patch": {"type": "string", "minLength": 1}},
                           ["path", "patch"]),
    "run_tests": _object({}, []),
    "submit_answer": _object({
        "answer": {"type": "object", "additionalProperties": True},
        "evidence_ids": {"type": "array", "items": {"type": "string", "minLength": 1}},
    }, ["answer", "evidence_ids"]),
}

_DESCRIPTIONS = {
    "list_files": "List permitted fixture files beneath a relative path.",
    "read_file": "Read a permitted fixture file using its relative path.",
    "write_file": "Replace a permitted source file with complete content.",
    "apply_patch": "Apply a unified diff to one permitted source file.",
    "run_tests": "Run the fixed public tests in the isolated CPU grader.",
    "submit_answer": "Submit the exact answer object and evidence ids in source order.",
}


def validate_tool(name: str, arguments: Any,
                  answer_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(name, str) or name not in TOOL_SCHEMAS:
        raise ValidationError("unexpected tool name")
    contract = copy.deepcopy(TOOL_SCHEMAS[name])
    if name == "submit_answer" and answer_schema is not None:
        contract["properties"]["answer"] = answer_schema
    return validate(arguments, contract)


def tool_definitions(include_answer: bool = True,
                     answer_schema: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return independent OpenAI function schemas; no engine-specific changes."""
    result = []
    for name, contract in TOOL_SCHEMAS.items():
        if name == "submit_answer" and not include_answer:
            continue
        parameters = copy.deepcopy(contract)
        if name == "submit_answer" and answer_schema is not None:
            parameters["properties"]["answer"] = copy.deepcopy(answer_schema)
        result.append({"type": "function", "function": {
            "name": name, "description": _DESCRIPTIONS[name], "parameters": parameters}})
    return result
