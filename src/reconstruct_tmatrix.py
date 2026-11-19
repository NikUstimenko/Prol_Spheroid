"""
Reconstruct an acoustotreams-compatible acoustic T-matrix of a centered
penetrable prolate spheroid from multiple plane-wave illuminations.

Reference for the method: 
M. Fruhnert, et al., Beilstein J. Nanotechnol. 8, pp. 614-626 (2017). 

This is a symmetry-adapted implementation of the multiple-plane-wave
T-matrix reconstruction idea.  It exploits axial symmetry, so the T-matrix
is block diagonal in m:

    T_{l m, l' m'} = delta_{m m'} T^(m)_{l l'}.

Only plane waves with incident azimuth phi_i = 0 are required.  For each
m >= 0, the block is reconstructed from

    A^(m) = T^(m) B^(m)
    T^(m) = A^(m) pinv(B^(m)).

For the centered prolate spheroid, reflection symmetry gives

    T^(-m) = T^(+m),

so the positive-m blocks are copied to negative m.

The final matrix is stored in acoustotreams' default ordering:

    (0,0),
    (1,-1), (1,0), (1,1),
    (2,-2), (2,-1), (2,0), (2,1), (2,2), ...

The normalization is chosen to match acoustotreams:

Regular spherical wave:
    j_l(kr) Y_lm(theta,phi)

Singular spherical wave:
    h_l^(1)(kr) Y_lm(theta,phi)

Plane-wave regular coefficients:
    b_lm = 4*pi * i^l * Y_lm^*(theta_i, phi_i)

Far-field relation:
    h_l^(1)(kr) Y_lm ~ exp(ikr)/r * [(-i)^(l+1)/k] Y_lm

so if
    c_lm = integral f(theta,phi) Y_lm^* dOmega
then the acoustotreams singular-wave coefficient is
    a_lm = k * i^(l+1) * c_lm.

Requirements
------------
Run from the `src` directory of:
    https://github.com/CRIMAC-WP4-Machine-learning/Prol_Spheroid

Requires NumPy, SciPy, h5py, Matplotlib, and the repository's Fortran compiler/setup.

Output
------
spheroid_pswf.hdf5, containing:
    frequency                    frequencies in Hz
    tmatrix                      complex array (frequency, outgoing mode, incoming mode)
    modes/l, modes/m              acoustotreams ordering
    omega, k, k_1, k0             angular frequency and wavenumbers
    materials, geometry           physical parameters, units, full axes and semiaxes
    reconstruction             numerical settings used to compute the T-matrix

The time convention is exp(-i*omega*t), and a = T b, with a scattered and
b incident. Here k = omega/c_b, k_1 = omega/c_1,
and acoustotreams' air-reference wavenumber is k0 = omega/(343 m/s).
The full axis lengths are a_x and a_z; the upstream spheroidal solver uses
semiaxes a_x/2 and a_z/2. Its auxiliary coefficients are separate from a,b.

Install h5py in addition to the solver's Python dependencies. The Fortran
launchers require gfortran on PATH.
Run from Prol_Spheroid/src:
    python reconstruct_tmatrix.py
"""

import os
import sys
import subprocess
from multiprocessing.util import Finalize
import shutil
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import numpy as np
import h5py

from numpy.polynomial.legendre import leggauss
from scipy.special import lpmv, gammaln

from IterativeSolvers import IterativeRefinement

from prol_spheroid import (
    write_Inputfile,
    infunc_content_Rad_SWF,
    infunc_content_dr_values,
)

from funcs.FUNC_ReadFortranOutput import (
    func_Rad_SWF_ArgExp,
    func_dr_values_ArgExp,
    func_Smn_eta_c_from_dr_ArgExp,
    Multi_ArgExp,
    Sum_ArgExp,
    Get_ArgExp,
    func_Integrate_SmnEta1_SmlEta2_ArgExp,
)


# ============================================================
# USER PARAMETERS
# ============================================================

# ---------- Physical parameters ----------
rho_b = 1000.0
c_b = 100.0 * np.sqrt(21.0)

rho_1 = 1.5 * rho_b
c_1 = c_b / 1.1

# Manuscript notation: full transverse and axial lengths (m).
a_x = 0.010
aspect_ratio = 1.5
a_z = aspect_ratio * a_x
C_AIR = 343.0  # acoustotreams uses k0 = omega/C_AIR.

# Frequencies at which the T-matrix is reconstructed
frequencies_hz = np.linspace(10e3, 25e3, 11)

# Set separately if you want to limit CPU usage.
N_WORKERS = min(len(frequencies_hz), os.cpu_count() or 1)

# ---------- Desired spherical T-matrix cutoff ----------
l_max = 11

# Number of plane-wave illuminations.
# Symmetry-adapted minimum is l_max+1 because the largest block is m=0.
# A factor ~2 overdetermines every block and is cheap here because the
# expensive spheroidal operator is built only once per m.
K = 2 * (l_max + 1)

# Polar quadrature order used to project the spheroidal far field onto
# spherical harmonics.  Increase this and verify convergence if needed.
N_q = max(64, 2 * l_max + 16)

# Spheroidal-series accuracy setting corresponding to the upstream code.
PRECISION_FBS = 1e-9

# Extra spheroidal modes beyond the conservative automatic estimate.
# Use your separate N-convergence study to choose this.
N_EXTRA = 0

# SVD cutoff used in the pseudoinverse of every incident coefficient block.
PINV_RCOND = 1e-12

# Output
OUTFILE = "spheroid_pswf.hdf5"


# ============================================================
# BASIC HELPERS
# ============================================================

def ae_value(argexp):
    """Convert repository [argument, base-10 exponent] representation."""
    return complex(argexp[0]) * (10.0 ** float(argexp[1]))


def ylm_norm(l, m):
    """
    Normalization for
        Y_l^m(theta,phi) = N_lm P_l^m(cos theta) exp(i m phi)
    for m >= 0, with scipy.special.lpmv's Condon-Shortley convention.
    """
    log_ratio = gammaln(l - m + 1.0) - gammaln(l + m + 1.0)
    return np.sqrt(
        (2.0 * l + 1.0) / (4.0 * np.pi) * np.exp(log_ratio)
    )


def acoustotreams_modes(lmax):
    """Return l,m arrays in acoustotreams default ordering."""
    ls = []
    ms = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            ls.append(l)
            ms.append(m)
    return np.asarray(ls, dtype=int), np.asarray(ms, dtype=int)


def mode_index_map(lmax):
    ls, ms = acoustotreams_modes(lmax)
    return {(int(l), int(m)): i for i, (l, m) in enumerate(zip(ls, ms))}


def incident_coefficients_positive_m(mu_inc, m, lmax):
    """
    acoustotreams plane-wave coefficients for phi_i=0 and m>=0:

        b_lm = 4*pi i^l Y_lm^*(theta_i,0)

    Shape: (number of l values, number of illuminations)
    """
    mu_inc = np.asarray(mu_inc, dtype=float)
    lvals = np.arange(m, lmax + 1, dtype=int)

    B = np.zeros((len(lvals), len(mu_inc)), dtype=np.complex128)

    for row, l in enumerate(lvals):
        P = lpmv(m, l, mu_inc)
        N = ylm_norm(l, m)

        B[row, :] = (
            4.0
            * np.pi
            * (1j ** l)
            * N
            * P
        )

    return B


# ============================================================
# FORTRAN SPHEROIDAL DATA
# ============================================================

def prepare_spheroidal_data(freq_hz):
    """
    Run the upstream Fortran spheroidal-function calculation only twice:
    once for the interior size parameter hs and once for the exterior hw.

    The resulting radial functions and angular expansion coefficients are
    independent of plane-wave incidence angle, which is why all illuminations
    can reuse them.
    """
    worker_src = Path.cwd()
    parent_dir = worker_src.parent

    # Compile the precision module before generating the spheroidal functions.
    subprocess.run(
        [sys.executable, str(worker_src / "Run_param_fort_FromPython.py")],
        stdout=subprocess.DEVNULL,
        check=True,
    )

    omega = 2.0 * np.pi * freq_hz
    k = omega / c_b
    k_1 = omega / c_1

    interfocal = np.sqrt(a_z * a_z - a_x * a_x)
    hw = interfocal * k / 2.0
    hs = interfocal * k_1 / 2.0

    # Use the largest upstream N estimate (axial-incidence value) as a
    # conservative common spheroidal order for ALL incident directions.
    if PRECISION_FBS < 1e-5:
        n_auto = int(np.ceil(0.75 * hs + 12.0))
    else:
        n_auto = int(np.ceil(0.75 * hs + 8.0))

    # Also keep a few more spheroidal modes than the desired spherical lmax.
    N_sph = max(n_auto + int(N_EXTRA), l_max + 3)

    # Need spheroidal sectors m=0,...,l_max.
    M_sph = l_max + 1

    script_profcn = worker_src / "Run_profcn_II_fort_FromPython.py"
    data = {"k": k, "N_sph": N_sph}

    # The radial functions and angular expansion coefficients are independent of
    # this angle. The Fortran solver still needs a value in its input file.
    theta_dummy_deg = 37.123
    for label, size_parameter in (("hs", hs), ("hw", hw)):
        write_Inputfile(
            size_parameter, aspect_ratio, M_sph, N_sph, theta_dummy_deg, parent_dir
        )
        subprocess.run(
            [sys.executable, str(script_profcn)],
            stdout=subprocess.DEVNULL,
            check=True,
        )
        data[f"Rad_{label}"] = infunc_content_Rad_SWF(parent_dir)
        data[f"dr_{label}"] = infunc_content_dr_values(parent_dir)

    return data


# ============================================================
# INCIDENCE-INDEPENDENT SPHEROIDAL m-BLOCK OPERATOR
# ============================================================

def build_unscaled_spheroidal_operator(data, m):
    """
    Build the incidence-independent linear operator for one spheroidal m sector.

    The upstream code forms matrices whose row n contains the plane-wave
    coefficient

        C_n = i^n E_m S_mn(cos theta_i).

    Writing
        D = diag(C) D0
        F = -diag(C) F0,

    the upstream solve can be transformed by y = diag(C) alpha into

        D0^T y = -F0^T C.

    Therefore
        y = H_m C,
        H_m = solve(D0^T, -F0^T),

    and H_m is independent of illumination direction.

    This removes the singular-row problem that occurs when an individual
    S_mn(cos theta_i) is exactly zero and makes the multi-plane-wave
    reconstruction much faster.
    """
    N = data["N_sph"]
    Nf = data["N_sph"]

    D0_arg = np.zeros((N, N), dtype=np.complex128)
    D0_exp = np.zeros((N, N), dtype=float)

    F0_arg = np.zeros((N, N), dtype=np.complex128)
    F0_exp = np.zeros((N, N), dtype=float)

    for il, l_sph in enumerate(range(m, m + N)):

        R_ml_hs = func_Rad_SWF_ArgExp(
            data["Rad_hs"], m, l_sph, Nf
        )
        R1_ml_hs = R_ml_hs[0]
        dR1_ml_hs = R_ml_hs[1]

        dml_hs = func_dr_values_ArgExp(
            data["dr_hs"], m, l_sph, Nf
        )

        for in_, n_sph in enumerate(range(m, m + N)):

            dmn_hw = func_dr_values_ArgExp(
                data["dr_hw"], m, n_sph, Nf
            )

            R_mn_hw = func_Rad_SWF_ArgExp(
                data["Rad_hw"], m, n_sph, Nf
            )

            R1_mn_hw = R_mn_hw[0]
            dR1_mn_hw = R_mn_hw[1]
            R2_mn_hw = R_mn_hw[2]
            dR2_mn_hw = R_mn_hw[3]

            R3_mn_hw = Sum_ArgExp(
                R1_mn_hw,
                Multi_ArgExp([1j, 0], R2_mn_hw),
            )
            dR3_mn_hw = Sum_ArgExp(
                dR1_mn_hw,
                Multi_ArgExp([1j, 0], dR2_mn_hw),
            )

            # D_nl without the incident C_n factor
            D_nl = Multi_ArgExp(
                Get_ArgExp(rho_1 / rho_b),
                Multi_ArgExp(dR3_mn_hw, R1_ml_hs),
            )
            D_nl = Sum_ArgExp(
                D_nl,
                Multi_ArgExp(
                    [-1, 0],
                    Multi_ArgExp(R3_mn_hw, dR1_ml_hs),
                ),
            )

            # F_nl without the incident -C_n factor
            F_nl = Multi_ArgExp(
                Get_ArgExp(rho_1 / rho_b),
                Multi_ArgExp(dR1_mn_hw, R1_ml_hs),
            )
            F_nl = Sum_ArgExp(
                F_nl,
                Multi_ArgExp(
                    [-1, 0],
                    Multi_ArgExp(R1_mn_hw, dR1_ml_hs),
                ),
            )

            Beta_nl = func_Integrate_SmnEta1_SmlEta2_ArgExp(
                m,
                n_sph,
                l_sph,
                dmn_hw,
                dml_hs,
            )

            Dtmp = Multi_ArgExp(Beta_nl, D_nl)
            Ftmp = Multi_ArgExp(Beta_nl, F_nl)

            D0_arg[in_, il] = Dtmp[0]
            D0_exp[in_, il] = Dtmp[1]

            F0_arg[in_, il] = Ftmp[0]
            F0_exp[in_, il] = Ftmp[1]

    D0 = D0_arg * np.power(10.0, D0_exp)
    F0 = F0_arg * np.power(10.0, F0_exp)

    solver = IterativeRefinement(
        solver_type="LU",
        tol=1e-11,
        maxiter=100,
        verbose=False,
    )

    # D0^T H = -F0^T
    return solver.solve(D0.T, -F0.T)


# ============================================================
# SPHEROIDAL ANGULAR MATRICES
# ============================================================

def spheroidal_angular_matrix(data, m, mu_values):
    """
    S[q,n] = S_{m,n_sph}(hw, mu_q)
    for n_sph = m,...,m+N_sph-1.
    """
    mu_values = np.asarray(mu_values, dtype=float)

    N = data["N_sph"]
    Nf = data["N_sph"]

    S = np.zeros((len(mu_values), N), dtype=np.complex128)

    for col, n_sph in enumerate(range(m, m + N)):
        dr = func_dr_values_ArgExp(
            data["dr_hw"], m, n_sph, Nf
        )

        for row, mu in enumerate(mu_values):
            val = func_Smn_eta_c_from_dr_ArgExp(
                m,
                n_sph,
                float(mu),
                dr,
            )
            S[row, col] = ae_value(val)

    return S


# ============================================================
# ONE m-BLOCK: PLANE-WAVE RESPONSES + SPHERICAL PROJECTION
# ============================================================

def reconstruct_m_block(data, m, mu_inc, mu_q, w_q):
    """Reconstruct the spherical T-matrix block for order m >= 0."""
    k = data["k"]
    N = data["N_sph"]

    H = build_unscaled_spheroidal_operator(data, m)

    # Angular spheroidal functions at incident and scattering directions.
    S_inc = spheroidal_angular_matrix(
        data, m, mu_inc
    )  # (K,N)

    S_scat = spheroidal_angular_matrix(
        data, m, mu_q
    )  # (N_q,N)

    n_sph = np.arange(m, m + N, dtype=int)

    Em = 1.0 if m == 0 else 2.0

    # C_n(theta_i) = i^n E_m S_mn(theta_i)
    C = (
        S_inc
        * (Em * np.power(1j, n_sph))[None, :]
    )  # (K,N)

    # y = H C, all illuminations at once.
    # Y shape (N,K)
    Y = H @ C.T

    # The upstream far-field expression contains a_n E_m S_inc.
    # Since y_n = C_n a_n and C_n=i^n E_m S_inc,
    # a_n E_m S_inc = y_n / i^n.
    coeff_out = (
        np.power(1j, -n_sph)[:, None] * Y
    )  # (N,K)

    # g_m(mu_q) for all K illuminations:
    # shape (N_q,K)
    G = (
        2.0 / (1j * k)
    ) * (S_scat @ coeff_out)

    # Project g_m onto positive-m normalized spherical harmonics.
    lvals = np.arange(m, l_max + 1, dtype=int)
    A = np.zeros(
        (len(lvals), len(mu_inc)),
        dtype=np.complex128,
    )

    phi_projection_factor = (
        2.0 * np.pi if m == 0 else np.pi
    )

    for row, l in enumerate(lvals):
        P_q = lpmv(m, l, mu_q)
        Nlm = ylm_norm(l, m)

        # c_lm = integral f Y_lm^* dOmega
        c_lm = (
            phi_projection_factor
            * Nlm
            * np.sum(
                (w_q * P_q)[:, None] * G,
                axis=0,
            )
        )

        # acoustotreams outgoing coefficient:
        # f_lm = (-i)^(l+1)/k * a_lm
        A[row, :] = (
            k
            * np.power(1j, l + 1)
            * c_lm
        )

    # Exact acoustotreams incident coefficients for phi_i=0.
    B = incident_coefficients_positive_m(
        mu_inc, m, l_max
    )

    return A @ np.linalg.pinv(B, rcond=PINV_RCOND)


# ============================================================
# FULL MATRIX ASSEMBLY
# ============================================================

def assemble_full_matrix(blocks):
    """
    Assemble full T in acoustotreams default ordering.

    For a centered isotropic prolate spheroid:
        T^(-m) = T^(+m).
    """
    index = mode_index_map(l_max)

    dim = (l_max + 1) ** 2
    T = np.zeros((dim, dim), dtype=np.complex128)

    for m, Tm in blocks.items():
        lvals = np.arange(m, l_max + 1, dtype=int)

        for i, l_out in enumerate(lvals):
            for j, l_in in enumerate(lvals):

                T[
                    index[(int(l_out), int(m))],
                    index[(int(l_in), int(m))],
                ] = Tm[i, j]

                if m > 0:
                    T[
                        index[(int(l_out), int(-m))],
                        index[(int(l_in), int(-m))],
                    ] = Tm[i, j]

    return T


# ============================================================
# PARALLEL FREQUENCY SWEEP
# ============================================================

def _cleanup_worker(worker_dir, original_dir):
    os.chdir(original_dir)
    shutil.rmtree(worker_dir, ignore_errors=True)


def _init_worker(original_src):
    """Keep each worker's profcn.dat and fort.* files in a private directory."""
    worker_dir = Path(tempfile.mkdtemp(
        prefix=f"prol_spheroid_tmat_{os.getpid()}_",
        dir=original_src.parent,
    ))
    worker_src = worker_dir / "src"
    Finalize(None, _cleanup_worker, args=(worker_dir, Path.cwd()), exitpriority=10)
    shutil.copytree(
        original_src, worker_src,
        ignore=shutil.ignore_patterns("__pycache__", "prol_spheroid_*"),
    )
    os.chdir(worker_src)


def compute_one_frequency(freq_hz, mu_inc, mu_q, w_q):
    """Reconstruct the T-matrix at one frequency."""
    data = prepare_spheroidal_data(freq_hz)
    blocks = {
        m: reconstruct_m_block(data, m, mu_inc, mu_q, w_q)
        for m in range(l_max + 1)
    }
    return assemble_full_matrix(blocks), data["N_sph"]


def write_hdf5(output, frequencies, results, mu_inc):
    """Save the complex T-matrices, mode order, and calculation metadata."""
    matrices, spheroidal_orders = zip(*results)
    T = np.stack(matrices)
    l_modes, m_modes = acoustotreams_modes(l_max)
    frequencies = np.asarray(frequencies, dtype=float)
    omega = 2.0 * np.pi * frequencies

    with h5py.File(output, "w") as h5:
        h5.attrs.update({
            "name": "Penetrable prolate spheroid",
            "description": "PSWF far fields projected onto SSWs; symmetry-adapted plane-wave reconstruction.",
            "storage_format": "acoustotreams manuscript reproduction, schema 1",
            "time_convention": "exp(-i*omega*t)",
            "coefficient_relation": "a = T b; a scattered, b incident",
            "basis": "regular j_l(k*r) Y_lm; outgoing h_l^(1)(k*r) Y_lm",
            "spherical_harmonics": "complex, orthonormal, Condon-Shortley phase",
            "constraints_enforced": "m=m_prime and equal +m/-m blocks",
        })

        def dataset(group, name, values, unit="1", description=None):
            ds = group.create_dataset(name, data=values)
            ds.attrs["unit"] = unit
            if description is not None:
                ds.attrs["description"] = description
            return ds

        dataset(h5, "frequency", frequencies, "Hz")
        tm = h5.create_dataset("tmatrix", data=T, compression="gzip", shuffle=True)
        tm.attrs["unit"] = "1"
        tm.attrs["axes"] = "frequency, outgoing mode (l,m), incoming mode (l_prime,m_prime)"
        for axis, label in enumerate(("frequency", "outgoing_mode", "incoming_mode")):
            tm.dims[axis].label = label
        dataset(h5, "omega", omega, "rad/s")
        dataset(h5, "k", omega / c_b, "1/m", "background wavenumber")
        dataset(h5, "k_1", omega / c_1, "1/m", "spheroid wavenumber")
        dataset(h5, "k0", omega / C_AIR, "1/m", "acoustotreams reference wavenumber in air")
        h5["k"].attrs["definition"] = "omega/c_b"
        h5["k_1"].attrs["definition"] = "omega/c_1"
        h5["k0"].attrs["definition"] = "omega/C_AIR"

        modes = h5.create_group("modes")
        dataset(modes, "l", l_modes)
        dataset(modes, "m", m_modes)
        modes.attrs["ordering"] = "l=0..l_max, m=-l..l"
        materials = h5.create_group("materials")
        for name, rho, speed, symbols in (("embedding", rho_b, c_b, "rho_b,c_b"),
                                          ("scatterer", rho_1, c_1, "rho_1,c_1")):
            medium = materials.create_group(name)
            medium.attrs["manuscript_symbols"] = symbols
            dataset(medium, "density", rho, "kg/m^3")
            dataset(medium, "speed_of_sound", speed, "m/s")
            dataset(medium, "transverse_speed_of_sound", 0.0, "m/s")

        geometry = h5.create_group("geometry")
        geometry.attrs.update({"shape": "prolate spheroid", "symmetry_axis": "z"})
        dataset(geometry, "a_x", a_x, "m", "full transverse axis length")
        dataset(geometry, "a_z", a_z, "m", "full axial length")
        dataset(geometry, "axis_lengths", [a_x, a_x, a_z], "m")
        dataset(geometry, "semiaxes", [a_x / 2, a_x / 2, a_z / 2], "m")
        dataset(geometry, "center", [0.0, 0.0, 0.0], "m")
        dataset(geometry, "aspect_ratio", aspect_ratio)

        settings = h5.create_group("reconstruction")
        settings.attrs["coefficient_matrices"] = "A scattered, B incident; T = A pinv(B)"
        settings.attrs["PSWF_truncation"] = "N_sph and M_sph are distinct from the SSW cutoff l_max"
        for key, value in (("l_max", l_max), ("K", K), ("N_q", N_q),
                           ("N_EXTRA", N_EXTRA), ("PINV_RCOND", PINV_RCOND),
                           ("PRECISION_FBS", PRECISION_FBS), ("C_AIR", C_AIR)):
            dataset(settings, key, value, "m/s" if key == "C_AIR" else "1")
        dataset(settings, "N_sph", np.asarray(spheroidal_orders, dtype=int))
        dataset(settings, "M_sph", l_max + 1)
        dataset(settings, "incident_pressure_amplitude", 1.0, "Pa")
        dataset(settings, "incident_mu", mu_inc)
        dataset(settings, "incident_theta_deg", np.degrees(np.arccos(mu_inc)), "degree")
        dataset(settings, "incident_phi_deg", 0.0, "degree")


def main():
    original_src = Path(__file__).resolve().parent
    mu_inc, _ = leggauss(K)
    mu_q, w_q = leggauss(N_q)
    reconstruct = partial(compute_one_frequency, mu_inc=mu_inc, mu_q=mu_q, w_q=w_q)

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(original_src,),
    ) as executor:
        # map returns the matrices in the order of the input frequencies.
        results = list(executor.map(reconstruct, frequencies_hz))

    output = Path(OUTFILE)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        temporary_output = Path(directory) / output.name
        write_hdf5(temporary_output, frequencies_hz, results, mu_inc)
        temporary_output.replace(output)
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
