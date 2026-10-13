"""Bound finite-noise population optimizers from independently verified profiles.

This consumes terminal-audit enclosures, never refits the original optimizer,
and never interprets an inconclusive sufficient bound as an absence of bias.
All scientific comparisons are exact rational or outward Arb comparisons.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import signal
import time

from flint import arb, ctx, fmpq
import flint


SIGMA_GRID = (F(1, 20),) + tuple(F(1, 10**k) for k in range(2, 21))
TAIL_ROUNDING_FLOOR = F(1, 2**256)
TERMINAL_VERSION = "terminal-partition-v1"
HISTORICAL_RUN_SEAL = "e14c41e4cba0dacd63607550d51eec53f5edd117224af1f797568f854beb4fcb"
HISTORICAL_SOURCE_MANIFEST = "f042e708377505e907b9ddc183dfb0a1b9bb15050b9ccc7e1e5f26f7a17967b8"


def exact_fraction(value):
    if isinstance(value, (float, bool)):
        raise ValueError("Scientific inputs must be exact integers or rational strings")
    return F(value)


def ball(value):
    value = exact_fraction(value)
    return arb(fmpq(value.numerator, value.denominator))


def endpoint(value, *, upper=True):
    """Return an exact finite dyadic endpoint, without a binary64 conversion."""
    out = value.upper() if upper else value.lower()
    if not out.is_finite() or not out.is_exact():
        raise ArithmeticError("Finite exact interval endpoint required")
    return F(str(out.fmpq()))


def nonnegative_upper(value):
    """Round astronomically small positive tails upward, never to zero."""
    if not value.is_finite() or value.lower() < 0:
        raise ArithmeticError("Expected a finite nonnegative bound")
    if value < ball(TAIL_ROUNDING_FLOOR):
        return TAIL_ROUNDING_FLOOR
    return endpoint(value)


def binary_entropy(p):
    if not (p > 0 and p < 1):
        raise ValueError("Entropy argument must be certified inside (0,1)")
    return -p*p.log() - (1-p)*(-p).log1p()


def validate_spec(spec):
    q0, lam = exact_fraction(spec["q0"]), exact_fraction(spec["lambda"])
    alpha, beta = map(exact_fraction, spec["I"])
    d = spec["d"]
    if type(d) is not int or d < 2 or d % 2:
        raise ValueError("The explicit chi-square formula requires even d >= 2")
    r = q0*(1-q0)
    if not (0 < alpha <= r < q0 <= beta < F(1, 2)) or lam < 0:
        raise ValueError("Invalid sparse activity interval or crossover")
    return q0, d, lam, alpha, beta


def variance_domain(spec):
    """A priori domain for *prelimit* optima, from a four-center small ball."""
    q0, d, _, alpha, _ = validate_spec(spec)
    m = d//2
    # Largest t on this deterministic dyadic approximation, with the density
    # union bound checked in exact arithmetic. t is squared ball radius.
    rhs = F(2**m * math.factorial(m), 8)
    low, high = F(0), F(1)
    while high**m <= rhs:
        high *= 2
    for _ in range(24):
        mid = (low+high)/2
        if mid**m <= rhs:
            low = mid
        else:
            high = mid
    rho_squared = low
    if rho_squared <= 0 or 4*rho_squared**m / (2**m*math.factorial(m)) > F(1, 2):
        raise ArithmeticError("Small-ball distortion floor failed")
    eta = rho_squared/2
    h0 = binary_entropy(ball(q0))
    reference = -arb(m)-2*h0
    raw_plus = (1+2*h0/m).exp()
    plus = F(math.ceil(endpoint(raw_plus)*256), 256)
    minus = eta/(2*d)
    for _ in range(64):
        value = -m*ball(minus).log()-ball(eta)/(2*ball(minus))
        if value.upper() < reference.lower():
            break
        minus /= 2
    else:
        raise ArithmeticError("Unable to certify lower variance cutoff")
    if not (0 < minus < eta/d < plus) or ball(plus) < raw_plus:
        raise ArithmeticError("Variance cutoffs failed verification")
    return {
        "rho_squared": str(rho_squared), "eta": str(eta),
        "v_minus": str(minus), "v_plus": str(plus),
        "reference_lower": str(endpoint(reference, upper=False)),
        "lower_cutoff_envelope_upper": str(endpoint(value)),
        "prior_cost_upper": str(endpoint(-2*ball(alpha).log())),
    }


def remainder(spec, sigma, domain=None):
    q0, d, lam, alpha, _ = validate_spec(spec)
    del q0
    sigma = exact_fraction(sigma)
    if sigma <= 0:
        raise ValueError("Finite-noise certificate requires sigma > 0")
    domain = variance_domain(spec) if domain is None else domain
    vm, vp = exact_fraction(domain["v_minus"]), exact_fraction(domain["v_plus"])
    C = -2*ball(alpha).log()
    A = ball(sigma*sigma)*(d+2*ball(vp)*C)
    conditions = {
        "cross_group_separation_at_least_one": lam*lam*sigma*sigma <= 3,
        "entropy_argument_at_most_half": bool(A.upper() <= ball(F(1, 2))),
        "wrong_group_prior_margin": bool(ball(1/(4*vp*sigma*sigma)) > C),
    }
    output = {"sigma": str(sigma), "conditions": conditions,
              "A_upper": str(endpoint(A)), "applicable": all(conditions.values())}
    if not output["applicable"]:
        return output
    entropy = 2*binary_entropy(A)
    x = F(1, 32*sigma*sigma)
    m = d//2
    # Exact polynomial coefficients avoid cancellation in tiny Gaussian tails.
    term, first = F(1), F(1)
    for j in range(1, m):
        term *= x/j
        first += term
    second = first+term*x/m
    tail = (-ball(x)).exp()*(C*ball(first)+ball(F(d, 2)/vm)*ball(second))
    entropy_upper, tail_upper = nonnegative_upper(entropy), nonnegative_upper(tail)
    output.update(entropy_upper=str(entropy_upper), tail_upper=str(tail_upper),
                  epsilon_upper=str(entropy_upper+tail_upper),
                  tail_rounding_floor=str(TAIL_ROUNDING_FLOOR))
    return output


def excludes(feasible_lower, outside_upper, epsilon_upper):
    if outside_upper is None:
        return False
    return (exact_fraction(feasible_lower)
            > exact_fraction(outside_upper)+exact_fraction(epsilon_upper))


def certify_cell(cell):
    spec = cell["spec"]
    validate_spec(spec)
    result = {"spec": spec, "terminal_audit_status": cell.get("status"),
              "original_status": cell.get("original_status"),
              "status": "audit_unresolved", "sigma_grid": [],
              "sigma0_q_lt_q0": None, "sigma0_q_gt_r": None,
              "sigma0_strict_bracket": None}
    if cell.get("status") != "verified":
        result["reason"] = "Independent terminal certificate not verified"
        return result
    lower = exact_fraction(cell["feasible_lower"])
    upper = cell["outside_upper"]
    lam = exact_fraction(spec["lambda"])
    if upper.get("q_ge_q0") is None or (lam > 0 and upper.get("q_le_r") is None):
        raise ValueError("Verified cell lacks full excluded-set upper bounds")
    domain = variance_domain(spec)
    result.update(status="bound_inconclusive", variance_domain=domain,
                  feasible_lower=str(lower), outside_upper=upper,
                  limiting_margin={
                      k: None if v is None else str(lower-exact_fraction(v))
                      for k, v in upper.items()})
    for sigma in SIGMA_GRID:
        check = remainder(spec, sigma, domain)
        up = check["applicable"] and excludes(lower, upper["q_ge_q0"], check["epsilon_upper"])
        down = (lam > 0 and check["applicable"]
                and excludes(lower, upper["q_le_r"], check["epsilon_upper"]))
        check.update(q_lt_q0_certified=bool(up), q_gt_r_certified=bool(down),
                     strict_bracket_certified=bool(up and down))
        for key, passed in (("sigma0_q_lt_q0", up), ("sigma0_q_gt_r", down),
                            ("sigma0_strict_bracket", up and down)):
            if passed and result[key] is None:
                result[key] = str(sigma)
        result["sigma_grid"].append(check)
    if result["sigma0_q_lt_q0"] or result["sigma0_q_gt_r"]:
        result["status"] = "positive_noise_exclusion_certified"
    result["at_original_noise_0_05"] = result["sigma_grid"][0]
    result["threshold_scope"] = "all 0 < sigma <= sigma0 on h=(lambda*sigma/2)^2"
    return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_panel():
    return [
        {"id": f"q{q.replace('/', '_')}_d{d}_l{lam.replace('/', '_')}",
         "q0": q, "d": d, "lambda": lam, "I": ["1/100", "2/5"]}
        for q in ("1/10", "3/10") for d in (2, 10)
        for lam in ("0", "1/2", "2", "8", "16")
    ]


def read_verified_terminal(path, seal_path, verifier_sha):
    """Bind a full or partial audit to its reviewed verifier and original run."""
    if not re.fullmatch(r"[0-9a-f]{64}", verifier_sha):
        raise ValueError("Expected verifier SHA-256 is required")
    if path.name != "TERMINAL_AUDIT.json" or seal_path.name != "SEAL.json" or path.parent != seal_path.parent:
        raise ValueError("Audit and SEAL.json must be the original sibling artifacts")
    seal = json.loads(seal_path.read_text())
    if (seal.get("version") != TERMINAL_VERSION
            or seal.get("files", {}).get(path.name) != sha(path)):
        raise ValueError("Terminal audit seal does not match its bytes/version")
    source = json.loads(path.read_text())
    required = {"version": TERMINAL_VERSION, "source_sha256": verifier_sha,
                "historical_run_seal_sha256": HISTORICAL_RUN_SEAL,
                "source_manifest_sha256": HISTORICAL_SOURCE_MANIFEST, "denominator": 20}
    if any(source.get(k) != value for k, value in required.items()):
        raise ValueError("Terminal audit has a different source or historical provenance")
    cells = source.get("cells", [])
    if [c["spec"] for c in cells] != fixed_panel():
        raise ValueError("Input must preserve every original specification and its order")
    verified = [c for c in cells if c.get("status") == "verified"]
    if (source.get("completed_cells") != len(verified)
            or source.get("passed") is not (len(verified) == 20)):
        raise ValueError("Terminal audit completion claims disagree with its rows")
    # Prefix-journal checking also keeps partial audit coverage explicit.
    prefix_path = path.parent/"VERIFIED_PREFIX.jsonl"
    if seal.get("files", {}).get(prefix_path.name) != sha(prefix_path):
        raise ValueError("Verified prefix journal is not bound by the audit seal")
    prefix = [json.loads(line) for line in prefix_path.read_text().splitlines() if line]
    if prefix != verified:
        raise ValueError("Verified cell rows differ from the independently saved prefix")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-audit", required=True, type=Path)
    parser.add_argument("--terminal-audit-seal", required=True, type=Path)
    parser.add_argument("--terminal-verifier-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or Path(os.environ["COLAB_TASK_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    destination = out/"FINITE_NOISE_CERTIFICATES.json"
    if destination.exists():
        raise FileExistsError("Preserve prior output; choose a new task/run directory")
    if os.environ.get("VG_SAE_COLAB_CPU_ONLY") != "1":
        raise RuntimeError("Run through the verified Colab CPU launcher")
    start = time.process_time()
    ctx.prec = 512
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    def timeout(signum, frame):
        raise TimeoutError("Finite-noise evaluation reached its wall limit")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    source = read_verified_terminal(args.terminal_audit, args.terminal_audit_seal,
                                    args.terminal_verifier_sha256)
    cells = source["cells"]
    rows = [certify_cell(cell) for cell in cells]
    result = {
        "version": 1, "status": "completed", "cells": rows,
        "source_sha256": sha(__file__), "terminal_audit_sha256": sha(args.terminal_audit),
        "terminal_audit_seal_sha256": sha(args.terminal_audit_seal),
        "terminal_verifier_sha256": args.terminal_verifier_sha256,
        "environment": {"python": platform.python_version(), "python_flint": flint.__version__,
                        "arb_precision_bits": ctx.prec},
        "cpu_seconds": time.process_time()-start, "cpu_cap_seconds": 60,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "scope": "supplied two-atom population model; exact fixed crossover path",
        "new_samples": 0, "training_runs": 0, "original_results_modified": False,
    }
    destination.write_text(json.dumps(result, indent=2)+"\n")
    signal.alarm(0)
    print(json.dumps({"cells": len(rows), "q_lt_q0_certified": sum(r["sigma0_q_lt_q0"] is not None for r in rows),
                      "strict_bracket_certified": sum(r["sigma0_strict_bracket"] is not None for r in rows),
                      "cpu_seconds": result["cpu_seconds"]}))


if __name__ == "__main__":
    main()
