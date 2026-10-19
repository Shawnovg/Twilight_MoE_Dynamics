"""Execute every notebook in a fresh kernel and save outputs plus HTML previews."""
import argparse
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/smoke.yaml')
    parser.add_argument('--pattern', default='*.ipynb')
    parser.add_argument('--output-dir', default=None, help='Optional separate executed-notebook directory')
    args = parser.parse_args()
    os.environ['TWILIGHT_CONFIG'] = str((ROOT / args.config).resolve())
    runtime = ROOT / '.runtime'
    runtime.mkdir(exist_ok=True)
    for name, subdir in [('JUPYTER_RUNTIME_DIR', 'jupyter'), ('IPYTHONDIR', 'ipython'),
                         ('MPLCONFIGDIR', 'matplotlib'), ('JUPYTER_DATA_DIR', 'jupyter_data'),
                         ('JUPYTER_CONFIG_DIR', 'jupyter_config')]:
        os.environ[name] = str(runtime / subdir)
        (runtime / subdir).mkdir(exist_ok=True)
    # Restrict discovery to the running environment; never pick a user's unrelated kernel.
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    manager = KernelSpecManager(kernel_dirs=[str(Path(sys.prefix) / 'share' / 'jupyter' / 'kernels')])
    output = ROOT / args.output_dir if args.output_dir else ROOT / 'notebooks'
    output.mkdir(parents=True, exist_ok=True)
    previews = output / 'html' if args.output_dir else ROOT / 'results' / 'previews'
    previews.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / 'notebooks').glob(args.pattern)):
        print('Executing', path.name, flush=True)
        nb = nbformat.read(path, as_version=4)
        # Remove summaries of prior runs before recomputing any outputs.
        nb.cells = [c for c in nb.cells if c.metadata.get('twilight_summary') is not True]
        nbformat.validate(nb)
        km = KernelManager(kernel_name='python3', kernel_spec_manager=manager)
        km.kernel_spec.argv[0] = sys.executable
        NotebookClient(nb, timeout=1800, startup_timeout=60, kernel_name='python3', km=km,
                       resources={'metadata': {'path': str(ROOT)}}).execute(cleanup_kc=True)
        nb.metadata['twilight_execution'] = {'config': args.config, 'status': 'completed'}
        nbformat.write(nb, output / path.name)
        html, _ = HTMLExporter().from_notebook_node(nb)
        (previews / (path.stem + '.html')).write_text(html, encoding='utf-8')
    print('All notebooks executed and saved.', flush=True)


if __name__ == '__main__':
    main()
