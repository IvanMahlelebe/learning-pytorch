import os
import glob


def get_latest_output_path():
  """Returns path to the most recent model outputs CSV file"""
  output_dir = "models/modeloutputs"
  files = glob.glob(os.path.join(output_dir, "iteration.outputs.*.csv"))
  if not files:
    raise FileNotFoundError(f"No output files found in {output_dir}")
  latest_file = max(files, key=os.path.getctime)
  return latest_file
