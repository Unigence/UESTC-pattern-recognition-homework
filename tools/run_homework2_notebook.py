"""Execute homework 2 with this interpreter and retain every inline output."""
from pathlib import Path
import json
import os
import sys
import tempfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "mlp_svm_gender.ipynb"
notebook = nbformat.read(PATH, as_version=4)

# The temporary specification uses the invoking .venv and changes no global kernel.
with tempfile.TemporaryDirectory(prefix="homework2_kernel_") as directory:
    spec_dir = Path(directory) / "homework2-local"
    spec_dir.mkdir()
    (spec_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Pattern Recognition Homework 2", "language": "python",
    }), encoding="utf-8")
    manager = KernelManager(kernel_name="homework2-local",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]))
    client = NotebookClient(notebook, km=manager, timeout=1800,
        resources={"metadata": {"path": str(ROOT)}}, store_widget_state=True)
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        # Retain diagnostic outputs too if a future rerun fails.
        nbformat.write(notebook, PATH)

code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
print(f"Executed {len(code_cells)} code cells; inline outputs saved in {PATH.name}.")
