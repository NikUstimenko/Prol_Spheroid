"""
Parallel frequency sweep for a penetrable prolate spheroid using
Gauss-Legendre integration for sigma_sca and the optical theorem for sigma_ext.

Place this file in Prol_Spheroid/src and run:
    python cross_section.py

Output:
    spheroid_reference.npz
"""

import multiprocessing as mp
import os
import shutil
import tempfile
import subprocess
from multiprocessing.util import Finalize
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from prol_spheroid import ProlateSpheroid
from IterativeSolvers import IterativeRefinement


# ============================================================
# Physical parameters
# ============================================================

rho0 = 1000.0
c0 = 100.0 * np.sqrt(21.0)

rho1 = 1.5 * rho0
c1 = c0 / 1.1

b = 0.005
aspect_ratio = 1.5
a = aspect_ratio * b

# Polar incidence angle relative to the spheroid symmetry axis z.
theta_i_deg = 0.0

frequencies = np.linspace(10e3, 25e3, 11)

# Gauss-Legendre quadrature order
Nq = 32

precision_fbs = 1e-6
N_EXTRA = -4

N_WORKERS = min(len(frequencies), os.cpu_count() or 1)
OUTFILE = "spheroid_reference.npz"


# ============================================================
# Worker sandbox
# ============================================================

_WORKER_TMP = None


def _cleanup_worker():
    global _WORKER_TMP
    if _WORKER_TMP is not None:
        shutil.rmtree(_WORKER_TMP, ignore_errors=True)
        _WORKER_TMP = None


def _init_worker(original_src):
    global _WORKER_TMP

    _WORKER_TMP = tempfile.mkdtemp(
        prefix=f"prol_spheroid_gauss_{os.getpid()}_",
        dir=Path(original_src).resolve().parent,
    )
    worker_src = Path(_WORKER_TMP) / "src"
    Finalize(None, _cleanup_worker, exitpriority=10)
    shutil.copytree(
        original_src, worker_src,
        ignore=shutil.ignore_patterns("__pycache__", "prol_spheroid_gauss_*"),
    )
    os.chdir(worker_src)


# ============================================================
# Settings
# ============================================================

class Settings:
    def __init__(self, freq):
        self.prefix = f"gauss_{freq:.6f}"
        self.ro_w = rho0
        self.ro_s = rho1
        self.c_w = c0
        self.c_s = c1
        self.a = a
        self.b = b
        self.delta_f = 1.0
        self.min_freq = float(freq)
        self.max_freq = float(freq + 1.0)
        self.theta_i_deg = theta_i_deg
        self.precision_fbs = precision_fbs
        self.n_extra = int(N_EXTRA)


def make_solver():
    return IterativeRefinement(
        solver_type="LU",
        tol=1e-10,
        maxiter=100,
        verbose=False,
    )


# ============================================================
# One frequency
# ============================================================


def compute_one_frequency(index, freq_hz):
    spheroid = ProlateSpheroid(
        Settings(freq_hz),
        make_solver(),
    )
    spheroid.n_extra = int(N_EXTRA)

    try:
        sigma_sca, sigma_ext, details = (
            spheroid.compute_scattering_cross_section_gauss(
                freq_Hz=float(freq_hz),
                Nq=Nq,
                return_details=True,
            )
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr or exc.stdout or str(exc)
        raise RuntimeError(f"PSWF calculation failed at {freq_hz:g} Hz:\n{message}") from exc

    k0 = 2.0*np.pi*float(freq_hz)/c0
    k0a = k0*a

    theta_i = np.deg2rad(theta_i_deg)

    # Orthogonal projected area of ellipsoid (b,b,a).
    A_proj = (
        np.pi*a*b*b
        / np.sqrt(
            b*b*np.sin(theta_i)**2
            + a*a*np.cos(theta_i)**2
        )
    )

    Q_sca = sigma_sca/A_proj
    Q_ext = sigma_ext/A_proj

    if sigma_sca != 0.0:
        optical_error = abs(sigma_ext - sigma_sca)/abs(sigma_sca)
    else:
        optical_error = np.nan

    return {
        "index": int(index),
        "frequency": float(freq_hz),
        "k0": float(k0),
        "k0a": float(k0a),
        "sigma_sca": float(sigma_sca),
        "sigma_ext": float(sigma_ext),
        "Q_sca": float(Q_sca),
        "Q_ext": float(Q_ext),
        "optical_error": float(optical_error),
        "A_proj": float(A_proj),
        "f_forward": complex(details["f_forward"]),
        "M_order": int(details["M_order"]),
        "N_order_Fortran": int(details["N_order_Fortran"]),
        "N_by_m": np.asarray(details["N_by_m"], dtype=int),
        "sigma_by_m": np.asarray(details["sigma_by_m"], dtype=float),
        "f_forward_by_m": np.asarray(
            details["f_forward_by_m"],
            dtype=np.complex128,
        ),
    }


# ============================================================
# Main
# ============================================================


def main():
    original_src = Path(__file__).resolve().parent

    if original_src.name != "src":
        raise RuntimeError(
            "Place this script in the Prol_Spheroid/src directory."
        )

    if shutil.which("gfortran") is None:
        raise RuntimeError("GNU Fortran (gfortran) is required and must be available on PATH.")

    results = [None]*len(frequencies)
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(str(original_src),),
    ) as executor:

        futures = {
            executor.submit(
                compute_one_frequency,
                i,
                float(freq),
            ): i
            for i, freq in enumerate(frequencies)
        }

        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            results[i] = result

    frequency = np.array(
        [r["frequency"] for r in results], dtype=float
    )
    k0 = np.array([r["k0"] for r in results], dtype=float)
    k0a = np.array([r["k0a"] for r in results], dtype=float)

    sigma_sca = np.array(
        [r["sigma_sca"] for r in results], dtype=float
    )
    sigma_ext = np.array(
        [r["sigma_ext"] for r in results], dtype=float
    )
    Q_sca = np.array(
        [r["Q_sca"] for r in results], dtype=float
    )
    Q_ext = np.array(
        [r["Q_ext"] for r in results], dtype=float
    )
    optical_error = np.array(
        [r["optical_error"] for r in results], dtype=float
    )
    A_proj = np.array(
        [r["A_proj"] for r in results], dtype=float
    )
    f_forward = np.array(
        [r["f_forward"] for r in results], dtype=np.complex128
    )

    M_order = np.array(
        [r["M_order"] for r in results], dtype=int
    )
    N_order_Fortran = np.array(
        [r["N_order_Fortran"] for r in results], dtype=int
    )

    N_by_m = np.empty(len(results), dtype=object)
    sigma_by_m = np.empty(len(results), dtype=object)
    f_forward_by_m = np.empty(len(results), dtype=object)

    for i, r in enumerate(results):
        N_by_m[i] = r["N_by_m"]
        sigma_by_m[i] = r["sigma_by_m"]
        f_forward_by_m[i] = r["f_forward_by_m"]

    np.savez(
        OUTFILE,
        frequency_hz=frequency,
        k0=k0,
        k0a=k0a,
        sigma_sca=sigma_sca,
        sigma_ext=sigma_ext,
        Q_sca=Q_sca,
        Q_ext=Q_ext,
        optical_error=optical_error,
        projected_area=A_proj,
        f_forward=f_forward,
        M_order=M_order,
        N_order_Fortran=N_order_Fortran,
        N_by_m=N_by_m,
        sigma_by_m=sigma_by_m,
        f_forward_by_m=f_forward_by_m,
        Nq=Nq,
        N_EXTRA=N_EXTRA,
        rho0=rho0,
        rho1=rho1,
        c0=c0,
        c1=c1,
        a=a,
        b=b,
        aspect_ratio=aspect_ratio,
        theta_i_deg=theta_i_deg,
        precision_fbs=precision_fbs,
    )

    print(f"\nSaved results to: {OUTFILE}")

    plt.figure()
    plt.plot(k0a, Q_sca, "o-", label=r"$Q_{\rm sca}$")
    plt.plot(k0a, Q_ext, "s--", label=r"$Q_{\rm ext}$")
    plt.xlabel(r"$k_0 a$")
    plt.ylabel("Efficiency")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
