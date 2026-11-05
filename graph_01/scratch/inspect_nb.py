import json, sys
sys.stdout.reconfigure(encoding="utf-8")

with open("notebook/01_diabetes_kg.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb.get('cells', []))}\n")

for i, cell in enumerate(nb.get("cells", [])):
    cell_type = cell.get("cell_type")
    source = "".join(cell.get("source", []))
    lines = source.split("\n")
    if cell_type == "markdown":
        headers = [l for l in lines if l.strip().startswith("#")]
        if headers:
            print(f"Cell {i} [MD]: {headers[0]}")
        else:
            preview = lines[0][:80] if lines else ""
            print(f"Cell {i} [MD preview]: {preview}")
    elif cell_type == "code":
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
        preview = code_lines[0][:80] if code_lines else (lines[0][:80] if lines else "empty")
        print(f"Cell {i} [Code]: {preview}")
