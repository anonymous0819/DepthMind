from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_impl = import_module('external_tools.depth.extract_o3dvqa_da3_depth')
if __name__ == "__main__":
    raise SystemExit(_impl.main())
else:
    sys.modules[__name__] = _impl
