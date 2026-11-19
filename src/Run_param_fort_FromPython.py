"""Compile the precision module; propagate compiler failures to the caller."""
import subprocess

subprocess.run(["gfortran", "-c", "param.f90"], check=True)
