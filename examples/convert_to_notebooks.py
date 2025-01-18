"""
Convert Python scripts to Jupyter notebooks.
This script takes Python files and converts them to Jupyter notebooks,
preserving markdown comments and code cells.
"""

import nbformat as nbf
import re
from pathlib import Path

def create_notebook_from_py(py_file):
    nb = nbf.v4.new_notebook()
    cells = []
    
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by triple quotes for markdown cells
    parts = re.split(r'"""(.*?)"""', content, flags=re.DOTALL)
    
    for i, part in enumerate(parts):
        if i % 2 == 1:  # Markdown content (inside triple quotes)
            cells.append(nbf.v4.new_markdown_cell(part.strip()))
        else:  # Code content
            code = part.strip()
            if code:  # Only add non-empty code cells
                cells.append(nbf.v4.new_code_cell(code))
    
    nb.cells = cells
    
    # Create output filename
    py_path = Path(py_file)
    nb_path = py_path.parent / f"{py_path.stem}.ipynb"
    
    # Write the notebook
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Created notebook: {nb_path}")

def main():
    # Basic usage examples
    create_notebook_from_py('examples/01_basic_usage/getting_started.py')
    
    # Parameter sweeps
    create_notebook_from_py('examples/02_parameter_sweeps/temperature_coupling_sweep.py')
    create_notebook_from_py('examples/02_parameter_sweeps/information_planck_sweep.py')
    create_notebook_from_py('examples/02_parameter_sweeps/field_strength_sweep.py')
    
    # Phase transitions
    create_notebook_from_py('examples/03_phase_transitions/01_critical_behavior.py')
    
    # Classical vs quantum comparisons
    create_notebook_from_py('examples/04_classical_vs_quantum/01_fidelity_comparison.py')

if __name__ == '__main__':
    main() 