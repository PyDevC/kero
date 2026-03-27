import os
import lit.formats

config.name = "Kero"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]

test_dir = os.path.join(os.path.dirname(__file__), "lit")
project_root = os.path.dirname(os.path.dirname(__file__))
bindir = os.path.join(project_root, "build", "bin")

config.environment["PATH"] = os.path.pathsep.join([bindir, os.environ.get("PATH", "")])

kero_opt_path = os.path.join(bindir, "kero-opt")
config.substitutions.append(("kero-opt", kero_opt_path))
config.substitutions.append(("FileCheck", os.path.join(os.environ["THIRDPARTY_LLVM_DIR"], "build", "bin", "FileCheck")))
