import os
import json
import numpy as np
import time
import argparse
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from genepro.scikit import GeneProClassifier
from genepro.node_impl import *


def run_regr(n_samples: int = 1_000_000, use_opencl: bool = True, n_jobs: int = 4, max_gens: int = 40, opencl_platform: str = "radeonsi"):
  """Run the regression/classification script.

  Args:
    n_samples: number of samples to bootstrap (with replacement) from the
      sklearn breast cancer dataset.
    use_opencl: whether to enable RUSTICL/OpenCL support via environment var

  Returns:
    tuple: (test_acc, best_tree_repr)
  """
  #os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1' # Enable PyOpenCL compiler output
  if use_opencl:
    os.environ['RUSTICL_ENABLE'] = opencl_platform  # Enable RUSTICL for AMD GPUs
  else:
    # ensure the variable is not set
    os.environ.pop('RUSTICL_ENABLE', None)

  # Let's load the Breast Cancer data set from sklearn
  X, y = load_breast_cancer(return_X_y=True)

  # Create a train and test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Bootstrap the training set to requested number of training samples (with replacement)
  if n_samples is not None and n_samples > 0:
    train_idx = np.random.choice(np.arange(X_train.shape[0]), size=n_samples, replace=True)
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]

  # Apply feature normalization
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # Set up what nodes genepro should use
  internal_nodes = [Plus(), Minus(), Times(), Div(), Log()]
  # As leaf nodes, let's set up the possibility to use each feature, plus a constant
  # (this is the default if leaf_nodes are not provided)
  num_features = X_train.shape[1]
  leaf_nodes = [Feature(i) for i in range(num_features)] + [Constant()]

  # Set up classifier
  gp = GeneProClassifier(
    score=balanced_accuracy_score,
    evo_kwargs={
      'internal_nodes': internal_nodes,
      'leaf_nodes': leaf_nodes,
      'verbose': True,
      'pop_size': 128,
      'max_gens': max_gens,
      'max_tree_size': 35,
      'n_jobs': n_jobs,
      'use_opencl': use_opencl,
    },
  )

  # Run
  gp.fit(X_train, y_train)

  # Get test (balanced) accuracy
  test_acc = balanced_accuracy_score(y_test, gp.predict(X_test))
  print("The balanced accuracy on the test set is {:.3f}".format(test_acc))
  # Get the best-found tree (at the last generation) and simplify it
  best_tree_repr = gp.evo.best_of_gens[-1].get_readable_repr()
  print("Obtained by the model:", best_tree_repr)

  return test_acc, best_tree_repr


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run regression/classification experiments with an optional cooldown between runs.")
  parser.add_argument("--cooldown", "-c", type=float, default=0.0,
                      help="Seconds to sleep between runs to reduce CPU temperature (default: 0).")
  parser.add_argument("--n-jobs", "-j", type=int, default=4,
                      help="Number of jobs/workers to use (passed to evo as 'n_jobs'; default: 4).")
  parser.add_argument("--max-gens", "-g", type=int, default=40,
                      help="Maximum number of generations for the evolutionary run (default: 40).")
  parser.add_argument("--sizes", "-s", type=str, default="100,1000,10000,100000,1000000",
                      help=("Comma-separated list of sample sizes to run. Example: --sizes=100,1000,10000 "
                            "(default: '100,1000,10000,100000,1000000')."))
  parser.add_argument("--opencl-platform", "-p", type=str, default="radeonsi",
                     help="OpenCL platform to use (default: 'radeonsi'), other options include 'iris', 'nouveau', etc.")
  parser.add_argument("--meta-out", type=str, default=None,
                     help="Optional path to write JSON metadata for the run (overwritten)."
                     )
  # mutually exclusive options to restrict runs to only opencl or only cpu
  group = parser.add_mutually_exclusive_group()
  group.add_argument("--opencl-only", action="store_true",
                     help="Run only with OpenCL/RUSTICL enabled.")
  group.add_argument("--cpu-only", action="store_true",
                     help="Run only without OpenCL/RUSTICL (CPU-only).")
  args = parser.parse_args()
  cooldown = float(args.cooldown)
  n_jobs = int(args.n_jobs)
  max_gens = int(args.max_gens)
  # parse comma-separated sizes into ints
  try:
    sizes = [int(s.strip()) for s in args.sizes.split(',') if s.strip()]
  except Exception:
    raise SystemExit("Invalid --sizes value; must be a comma-separated list of integers.")
  # Determine which opencl settings to run
  if args.opencl_only:
    opencl_choices = [True]
  elif args.cpu_only:
    opencl_choices = [False]
  else:
    opencl_choices = [True, False]
  results = []

  for use_opencl in opencl_choices:
    print(f"\nRunning experiments with use_opencl={use_opencl}")
    for i, n in enumerate(sizes):
      print(f"\nStarting run: n_samples={n}, use_opencl={use_opencl}")
      t0 = time.perf_counter()
      try:
        test_acc, best = run_regr(n_samples=n, use_opencl=use_opencl, n_jobs=n_jobs, max_gens=max_gens)
      except Exception as e:
        # record failure
        elapsed = time.perf_counter() - t0
        print(f"Run failed for n={n}, use_opencl={use_opencl}: {e}")
        entry = {
          'n_samples': n,
          'use_opencl': use_opencl,
          'elapsed_s': elapsed,
          'test_acc': None,
          'error': str(e),
        }
        results.append(entry)
        # write metadata if requested
        try:
          if args.meta_out:
            with open(args.meta_out, 'w', encoding='utf-8') as mfh:
              json.dump(entry, mfh)
        except Exception:
          # don't let metadata writing break the run
          pass
        continue

      elapsed = time.perf_counter() - t0
      print(f"Completed run: n_samples={n}, use_opencl={use_opencl}, time={elapsed:.2f}s, acc={test_acc:.3f}")
      results.append({
        'n_samples': n,
        'use_opencl': use_opencl,
        'elapsed_s': elapsed,
        'test_acc': float(test_acc),
        'error': None,
      })
      # write metadata if requested
      try:
        if args.meta_out:
          entry = {
            'n_samples': n,
            'use_opencl': use_opencl,
            'elapsed_s': elapsed,
            'test_acc': float(test_acc),
            'error': None,
          }
          with open(args.meta_out, 'w', encoding='utf-8') as mfh:
            json.dump(entry, mfh)
      except Exception:
        # don't let metadata writing break the run
        pass

      # apply cooldown between runs if requested, but not after the very last run
      is_last_run = (use_opencl is False) and (i == len(sizes) - 1)
      if cooldown > 0 and not is_last_run:
        print(f"Cooling down for {cooldown:.1f}s to reduce CPU temperature...")
        time.sleep(cooldown)

  # Summary
  print("\nSummary of runs:")
  for r in results:
    status = "OK" if r['error'] is None else f"ERR: {r['error']}"
    print(f"n={r['n_samples']:>7}, opencl={r['use_opencl']}, time={r['elapsed_s']:8.2f}s, acc={r['test_acc']}, {status}")
