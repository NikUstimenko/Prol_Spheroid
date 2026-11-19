"""Compile and run the spheroidal-function solver in the current worker directory."""
import os
from pathlib import Path
import subprocess

executable = Path("profcn_II.exe" if os.name == "nt" else "profcn_II").resolve()
subprocess.run(
    ["gfortran", "-std=legacy", "param.f90", "profcn_II.f90", "-o", str(executable)],
    check=True,
)
subprocess.run([str(executable)], check=True)
