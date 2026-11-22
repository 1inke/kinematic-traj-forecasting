"""Reevaluate the original decision rules from the retained, sealed tables."""
import ast
import gzip
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType


SNAPSHOT = Path(__file__).resolve().parents[1] / "snapshot"


def original(name):
    path = SNAPSHOT / name
    if path.is_file():
        return path.read_bytes()
    with gzip.open(path.with_suffix(path.suffix + ".gz"), "rb") as stream:
        return stream.read()


def main():
    done = json.loads(original("DONE.json"))
    data = {}
    for name, expected in done["files"].items():
        raw = original(name)
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f"Original result checksum differs: {name}")
        if name.endswith(".json"):
            data[name] = json.loads(raw)
    tables = {name.removeprefix("bias_tables_").removesuffix(".json"): value
              for name, value in data.items() if name.startswith("bias_tables_")}

    # The decision module imports only the OBS labels from the data module.
    # Read that literal so the aggregate replay needs no private prediction inputs.
    source = ast.parse(original("bias_inputs.py"))
    names = next(node.value for node in source.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "OBS" for t in node.targets))
    inputs = ModuleType("bias_inputs")
    inputs.OBS = ast.literal_eval(names)
    sys.modules["bias_inputs"] = inputs
    evaluate = runpy.run_path(str(SNAPSHOT / "bias_decisions.py"))["evaluate"]
    replay = evaluate(tables, data["geom_correction.json"], data["prob_correction.json"])
    reference = data["RESULT.json"]
    for key, value in replay.items():
        if reference[key] != value:
            raise ValueError(f"Decision replay differs: {key}")
    print(json.dumps(dict(status="PASS", groups=len(tables), sealed_files=len(done["files"]),
                          markers=replay["summary"])))


if __name__ == "__main__":
    main()
