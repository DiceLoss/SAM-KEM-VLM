import hashlib
import json

import numpy as np

from sam_kem.display import render_prompt_png
from sam_kem.noise import array_sha256


def test_prompt_render_is_bound_to_exact_mlem_array(tmp_path):
  source = tmp_path / "mlem10.npy"
  output = tmp_path / "prompt.png"
  array = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
  np.save(source, array)
  render_prompt_png(source, output)
  metadata = json.loads(output.with_suffix(".json").read_text())
  assert metadata["source_mlem10_sha256"] == array_sha256(array)
  assert metadata["png_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
