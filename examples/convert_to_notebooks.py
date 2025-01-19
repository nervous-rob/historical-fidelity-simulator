"""Convert Python Examples to Jupyter Notebooks

This script creates a master notebook that organizes and documents all example scripts,
providing a comprehensive guide to the Historical Fidelity Simulator.

Features:
1. Creates individual notebooks for each example
2. Generates a master notebook with proper cross-references
3. Includes theoretical background and implementation details
4. Provides clear navigation between related examples
"""

import nbformat as nbf
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

def extract_module_docstring(content: str) -> Optional[str]:
    """Extract the module-level docstring from Python content.
    
    Args:
        content: Python file content
        
    Returns:
        Extracted module docstring if found, None otherwise
    """
    # Look for module docstring at the start of the file
    match = re.match(r'^"""(.*?)"""', content, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_function_docstrings(content: str) -> List[Tuple[str, str, str]]:
    """Extract function definitions and their docstrings.
    
    Args:
        content: Python file content
        
    Returns:
        List of tuples (function_def, docstring, function_body)
    """
    # Pattern to match function definitions with docstrings
    pattern = r'(def\s+[^:]+:)\s*"""(.*?)"""(.*?)(?=\n\s*(?:def|\Z))'
    matches = re.finditer(pattern, content, re.DOTALL)
    return [(m.group(1), m.group(2), m.group(3)) for m in matches]

def extract_main_function(content: str) -> Optional[str]:
    """Extract the main function definition if it exists.
    
    Args:
        content: Python file content
        
    Returns:
        Main function code block if found, None otherwise
    """
    # Pattern to match main function definition and body
    pattern = r'def\s+main\s*\([^)]*\)\s*->\s*None:\s*""".*?"""(.*?)(?=\n\s*(?:def|\s*if\s+__name__|\Z))'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return f"def main() -> None:{match.group(1)}"
    return None

def split_code_blocks(content: str) -> List[str]:
    """Split content into code blocks, preserving imports and setup code.
    
    Args:
        content: Python file content
        
    Returns:
        List of code blocks
    """
    # Remove module docstring first
    content = re.sub(r'^""".*?"""', '', content, flags=re.DOTALL)
    
    # Split by function definitions but preserve imports and setup
    blocks = []
    
    # Extract imports and initial setup
    setup_pattern = r'^.*?(?=\n\s*def|\Z)'
    setup_match = re.match(setup_pattern, content, re.DOTALL)
    if setup_match:
        setup_code = setup_match.group(0).strip()
        if setup_code:
            blocks.append(setup_code)
    
    # Extract function blocks (excluding main function)
    func_pattern = r'(def\s+(?!main\s*\([^)]*\)\s*->\s*None:).*?:.*?(?=\n\s*(?:def|\Z)))'
    functions = re.finditer(func_pattern, content, re.DOTALL)
    for func in functions:
        blocks.append(func.group(1).strip())
    
    return [b for b in blocks if b.strip()]

def ensure_directory_exists(path: Path) -> None:
    """Ensure the specified directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
    """
    path.mkdir(parents=True, exist_ok=True)

def create_notebook_from_py(py_file: Path, notebooks_dir: Path) -> Path:
    """Convert a Python file to a Jupyter notebook.
    
    Args:
        py_file: Path to the Python file
        notebooks_dir: Directory to save the notebook
        
    Returns:
        Path to the created notebook
    """
    nb = nbf.v4.new_notebook()
    cells = []
    
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add module docstring as first markdown cell
    module_docstring = extract_module_docstring(content)
    if module_docstring:
        cells.append(nbf.v4.new_markdown_cell(module_docstring))
    
    # Process code blocks
    code_blocks = split_code_blocks(content)
    
    for block in code_blocks:
        # Check if this block is a function definition
        if block.startswith('def '):
            # Extract function docstring
            func_matches = re.match(r'(def\s+[^:]+:)\s*"""(.*?)"""(.*)', block, re.DOTALL)
            if func_matches:
                # Add function docstring as markdown
                func_def = func_matches.group(1)
                docstring = func_matches.group(2)
                func_body = func_matches.group(3)
                
                cells.append(nbf.v4.new_markdown_cell(f"### {func_def}\n\n{docstring}"))
                if func_body.strip():
                    cells.append(nbf.v4.new_code_cell(f"{func_def}{func_body}"))
            else:
                cells.append(nbf.v4.new_code_cell(block))
        else:
            # Non-function code block
            cells.append(nbf.v4.new_code_cell(block))
    
    # Add main function if it exists
    main_func = extract_main_function(content)
    if main_func:
        # Add main function definition
        cells.append(nbf.v4.new_code_cell(main_func))
        # Add cell to run main
        cells.append(nbf.v4.new_code_cell("# Run the main function\nmain()"))
    
    nb.cells = cells
    
    # Create output filename in notebooks directory, preserving category structure
    relative_path = py_file.relative_to(Path('examples/scripts'))
    nb_path = notebooks_dir / relative_path.parent / f"{py_file.stem}.ipynb"
    ensure_directory_exists(nb_path.parent)
    
    # Write the notebook
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Created notebook: {nb_path}")
    return nb_path

def create_master_notebook(example_files: Dict[str, List[Path]], notebooks_dir: Path) -> None:
    """Create a master notebook that organizes all examples.
    
    Args:
        example_files: Dictionary mapping categories to lists of example files
        notebooks_dir: Directory to save the master notebook
    """
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Title and introduction
    cells.append(nbf.v4.new_markdown_cell("""# Historical Fidelity Simulator Examples
    
This notebook provides a comprehensive guide to using the Historical Fidelity Simulator,
organized by topic and complexity. Each section includes theoretical background,
implementation details, and links to specific example notebooks.

For more details, see the [research proposal](../docs/research-proposal.md) and
[development standards](../docs/development_standards_and_project_goals.md).
"""))
    
    # Table of contents
    toc = ["## Table of Contents\n"]
    for category in example_files:
        toc.append(f"- [{category}](#{category.lower().replace(' ', '-')})")
    cells.append(nbf.v4.new_markdown_cell("\n".join(toc)))
    
    # Add each category
    for category, files in example_files.items():
        # Category header
        cells.append(nbf.v4.new_markdown_cell(f"## {category}"))
        
        # Process each example in the category
        for py_file in files:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract docstring
            docstring = extract_module_docstring(content)
            if docstring:
                # Create relative link to the notebook
                relative_path = py_file.relative_to(Path('examples/scripts'))
                nb_path = notebooks_dir / relative_path.parent / f"{py_file.stem}.ipynb"
                rel_path = nb_path.relative_to(notebooks_dir.parent)
                
                cells.append(nbf.v4.new_markdown_cell(f"""### [{py_file.stem}]({rel_path})

{docstring}
"""))
    
    nb.cells = cells
    
    # Save master notebook
    master_path = notebooks_dir / "master_guide.ipynb"
    
    # Write the notebook
    with open(master_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Created master notebook: {master_path}")

def main() -> None:
    """Convert example scripts to notebooks and create master guide."""
    # Define example categories and files
    example_files = {
        "Basic Usage": [
            Path('examples/scripts/01_basic_usage/getting_started.py')
        ],
        "Parameter Sweeps": [
            Path('examples/scripts/02_parameter_sweeps/temperature_coupling_sweep.py'),
            Path('examples/scripts/02_parameter_sweeps/information_planck_sweep.py'),
            Path('examples/scripts/02_parameter_sweeps/field_strength_sweep.py')
        ],
        "Phase Transitions": [
            Path('examples/scripts/03_phase_transitions/critical_behavior.py')
        ],
        "Classical vs Quantum": [
            Path('examples/scripts/04_classical_vs_quantum/fidelity_comparison.py')
        ]
    }
    
    # Set up output directories
    notebooks_dir = Path('examples/notebooks')
    ensure_directory_exists(notebooks_dir)
    
    # Convert individual files
    for category_files in example_files.values():
        for py_file in category_files:
            create_notebook_from_py(py_file, notebooks_dir)
    
    # Create master notebook
    create_master_notebook(example_files, notebooks_dir)

if __name__ == '__main__':
    main() 