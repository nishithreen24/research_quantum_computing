cat <<'EOF' > setup.sh
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

echo "==> qanneal setup (CPU)"

if ! command -v cmake >/dev/null 2>&1; then
  echo "CMake not found. Install it first: brew install cmake"
  exit 1
fi

python -V
python -m pip install -U pip setuptools wheel
python -m pip install -U scikit-build-core pybind11 numpy

PYBIND11_DIR="$(python -m pybind11 --cmakedir)"
rm -rf build
cmake -S . -B build -DQANNEAL_BUILD_PYTHON=ON -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build build

python - <<'PY'
import site, glob, shutil, os
site_dir = site.getsitepackages()[0]
dst = os.path.join(site_dir, "qanneal")
os.makedirs(dst, exist_ok=True)
for f in glob.glob("python/qanneal/_qanneal*"):
    shutil.copy2(f, dst)
    print("copied:", f, "->", dst)
PY

python - <<'PY'
import qanneal
import qanneal._qanneal
print("qanneal version:", qanneal.__version__)
PY

echo "==> Setup complete"
EOF

chmod +x setup.sh
./setup.sh
