"""Check a frozen, finite-parameter VG bias witness on exact population atoms.

Execute only through a Colab task with COLAB_TASK_OUTPUT set. This is a
deterministic four-state population calculation, with no sampling, seed,
optimization, training, or LLM activation collection. All 16 frozen cases remain
in the output, including differences unresolved at float64 precision.
"""

from __future__ import annotations

import time

_MODULE_WALL_STARTED = time.monotonic()

import csv
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys


def _process_wall_origin() -> tuple[float, str]:
    """Include interpreter startup when Linux process-age metadata is available.

    proc uptime has limited printed precision. The extra 0.05 s makes the
    reconstructed start conservative for that quantization and these two reads.
    The fallback starts before the expensive Torch import and is disclosed.
    """
    try:
        fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        age = max(0.0, uptime - start_ticks / os.sysconf("SC_CLK_TCK"))
        return time.monotonic() - age - 0.05, "linux_process_age_with_0.05s_guard"
    except (OSError, ValueError, IndexError):
        return _MODULE_WALL_STARTED, "module_entry_before_torch_import"


_PROCESS_WALL_STARTED, _PROCESS_WALL_SCOPE = _process_wall_origin()

TOTAL_CPU_SECONDS = 60
TOTAL_WALL_SECONDS = 120
FINALIZATION_RESERVE_SECONDS = 5
RESULT_FILENAME = "native_bias_witness_results.json"


def _install_import_deadlines() -> None:
    """Protect the executable's imports; importing helpers in tests is inert."""
    if os.environ.get("VG_SAE_COLAB_CPU_ONLY") != "1":
        raise SystemExit("VG_SAE_COLAB_CPU_ONLY must be exactly '1'")
    output = os.environ.get("COLAB_TASK_OUTPUT")
    if not output:
        raise SystemExit("COLAB_TASK_OUTPUT is required; no local output default")
    if (Path(output) / RESULT_FILENAME).exists():
        raise SystemExit(f"Preserve existing attempt: {RESULT_FILENAME} already exists")
    inherited = [int(value) for value in resource.getrlimit(resource.RLIMIT_CPU)
                 if value != resource.RLIM_INFINITY]
    hard_cpu = min([TOTAL_CPU_SECONDS, *inherited])
    resource.setrlimit(resource.RLIMIT_CPU, (hard_cpu, hard_cpu))

    def import_timeout(_signum, _frame):
        raise SystemExit(124)

    signal.signal(signal.SIGALRM, import_timeout)
    remaining = TOTAL_WALL_SECONDS - (time.monotonic() - _PROCESS_WALL_STARTED)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, remaining))


if __name__ == "__main__":
    _install_import_deadlines()

import torch

# Permit both `python -m scripts.check_native_bias_witness` and the script path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sae_model import VGSAEConfig, VariationalGarroteSAE


PROTOCOL_VERSION = "native-finite-beta-bias-v1"
DECIMAL_PRECISION = 240
ROUNDOFF_FACTOR = 8192
Q_VALUES = ("0.1", "0.3")
BETA_VALUES = ("1", "4", "40", "400")
GAMMA_VALUES = ("1", "16")
SUPPORTS = ((0, 0), (0, 1), (1, 0), (1, 1))
DIMENSION = 2


class BudgetExceeded(RuntimeError):
    """A terminal bounded-attempt outcome, not permission to change a case."""


def validate_runtime(output: Path, *, environ=None,
                     cuda_available: bool | None = None) -> None:
    """Check execution authorization and preservation before creating outputs."""
    environment = os.environ if environ is None else environ
    if environment.get("VG_SAE_COLAB_CPU_ONLY") != "1":
        raise RuntimeError("VG_SAE_COLAB_CPU_ONLY must be exactly '1'")
    cuda_present = torch.cuda.is_available() if cuda_available is None else cuda_available
    if cuda_present:
        raise RuntimeError("This frozen task requires a CPU-only runtime; CUDA is available")
    if (output / RESULT_FILENAME).exists():
        raise FileExistsError(f"Preserve existing attempt: {RESULT_FILENAME} already exists")


class ExecutionBudget:
    """Entrypoint-only process limits; never instantiate in numerical unit tests.

    CPU consumption is time.process_time(), measured from process start, so it
    includes imports. SIGXCPU and SIGALRM stop science with five seconds reserved
    for durable output. The hard process CPU limit is 60 s; a final wall alarm
    enforces the 120 s deadline during serialization as well.
    """

    def __init__(self):
        self.wall_origin = _PROCESS_WALL_STARTED
        self.original_cpu_limits = resource.getrlimit(resource.RLIMIT_CPU)
        inherited = [int(value) for value in self.original_cpu_limits
                     if value != resource.RLIM_INFINITY]
        self.hard_cpu = min([TOTAL_CPU_SECONDS, *inherited])
        self.science_cpu = max(0, self.hard_cpu - FINALIZATION_RESERVE_SECONDS)
        self.science_wall = TOTAL_WALL_SECONDS - FINALIZATION_RESERVE_SECONDS
        self.installed = False

    @staticmethod
    def _cpu_signal(_signum, _frame):
        raise BudgetExceeded("Process CPU budget reached; scientific work stopped")

    @staticmethod
    def _wall_signal(_signum, _frame):
        raise BudgetExceeded("Process wall budget reached; scientific work stopped")

    @staticmethod
    def _final_wall_signal(_signum, _frame):
        raise SystemExit(124)

    def install(self) -> None:
        signal.signal(signal.SIGXCPU, self._cpu_signal)
        signal.signal(signal.SIGALRM, self._wall_signal)
        resource.setrlimit(resource.RLIMIT_CPU, (self.science_cpu, self.hard_cpu))
        self.installed = True
        remaining = self.science_wall - (time.monotonic() - self.wall_origin)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, remaining))

    def check(self) -> None:
        if time.process_time() >= self.science_cpu:
            raise BudgetExceeded("Process CPU budget reached; scientific work stopped")
        if time.monotonic() - self.wall_origin >= self.science_wall:
            raise BudgetExceeded("Process wall budget reached; scientific work stopped")

    def begin_finalization(self) -> None:
        if not self.installed:
            return
        signal.setitimer(signal.ITIMER_REAL, 0)
        # Raising the soft limit to the already-installed hard limit suppresses
        # repeated SIGXCPU during the reserved serialization interval.
        resource.setrlimit(resource.RLIMIT_CPU, (self.hard_cpu, self.hard_cpu))
        signal.signal(signal.SIGALRM, self._final_wall_signal)
        remaining = TOTAL_WALL_SECONDS - (time.monotonic() - self.wall_origin)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, remaining))


def retain_budget_remainder(rows: list[dict], reason: str,
                            interrupted_case: bool) -> None:
    """Keep every frozen row after a timeout without invoking more computation."""
    for index, case in enumerate(frozen_cases()[len(rows):]):
        status = ("budget_unresolved_interrupted" if index == 0 and interrupted_case
                  else "budget_unresolved_not_started")
        rows.append({"case_id": case.case_id, **asdict(case),
                     "status": status, "budget_reason": reason,
                     "numeric_result_available": False})


@dataclass(frozen=True)
class WitnessCase:
    q: str
    beta: str
    gamma: str

    @property
    def case_id(self) -> str:
        return f"q{self.q.replace('.', 'p')}_beta{self.beta}_gamma{self.gamma}"


def frozen_cases() -> tuple[WitnessCase, ...]:
    return tuple(WitnessCase(*row) for row in itertools.product(
        Q_VALUES, BETA_VALUES, GAMMA_VALUES
    ))


def _d(value: Decimal | str | int | float) -> Decimal:
    """Preserve actual float bits for the saved-tensor reference."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal.from_float(value)
    return Decimal(value)


def _softplus(value: Decimal) -> Decimal:
    # Stable for both signs; precision is fixed by the caller.
    return max(value, Decimal(0)) + (1 + (-abs(value)).exp()).ln()


def _sigmoid(value: Decimal) -> Decimal:
    if value >= 0:
        return 1 / (1 + (-value).exp())
    positive = value.exp()
    return positive / (1 + positive)


def _inverse_softplus(value: Decimal) -> Decimal:
    if value <= 0:
        raise ValueError("Finite softplus witnesses require positive amplitudes")
    return (value.exp() - 1).ln()


def _validate(q: Decimal, beta: Decimal, gamma: Decimal) -> None:
    if not all(value.is_finite() for value in (q, beta, gamma)):
        raise ValueError("Parameters must be finite")
    if not 0 < q < 1 or beta <= 0 or gamma <= 0:
        raise ValueError("Require 0 < q < 1, beta > 0, gamma > 0")


def profile_value(q: Decimal, beta: Decimal, gamma: Decimal,
                  bias: Decimal) -> Decimal:
    """Scalar closure objective; excludes the bias-independent Gaussian term."""
    q, beta, gamma, bias = map(_d, (q, beta, gamma, bias))
    _validate(q, beta, gamma)
    if not 0 <= bias <= 1:
        raise ValueError("Require 0 <= bias <= 1")
    active_logit = beta * (1 - bias) ** 2 / 2 - gamma
    return (
        beta / 2 * ((1 - q) * bias**2 + q * (1 - bias) ** 2)
        + q * _softplus(-gamma)
        - q * _softplus(active_logit)
    )


def profile_derivative(q: Decimal, beta: Decimal, gamma: Decimal,
                       bias: Decimal) -> Decimal:
    q, beta, gamma, bias = map(_d, (q, beta, gamma, bias))
    _validate(q, beta, gamma)
    if not 0 <= bias <= 1:
        raise ValueError("Require 0 <= bias <= 1")
    gate = _sigmoid(beta * (1 - bias) ** 2 / 2 - gamma)
    return beta * (bias - q + q * (1 - bias) * gate)


def analytic_design(case: WitnessCase) -> dict[str, Decimal | bool]:
    """One prespecified bias and floor, never selected using a computed outcome."""
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        q, beta, gamma = map(Decimal, (case.q, case.beta, case.gamma))
        _validate(q, beta, gamma)
        deficit = _sigmoid(gamma - beta / 2)
        bias = q * deficit
        floor = bias / 16
        prior = _sigmoid(-gamma)
        closure_lower = beta * bias**2 / 2
        baseline = profile_value(q, beta, gamma, Decimal(0))
        candidate = profile_value(q, beta, gamma, bias)
        baseline_defect = (1 - q) * beta * prior * floor**2 / 2
        candidate_defect = (1 - q) * beta * prior * (bias * floor + floor**2 / 2)
        finite_delta = baseline - candidate + baseline_defect - candidate_defect
        return {
            "q": q, "beta": beta, "gamma": gamma,
            "prior_probability": prior,
            "initial_active_gate_deficit": deficit,
            "candidate_bias": bias,
            "inactive_amplitude_floor": floor,
            "closure_baseline_scalar": baseline,
            "closure_candidate_scalar": candidate,
            "closure_delta_scalar": baseline - candidate,
            "closure_delta_lower_scalar": closure_lower,
            "finite_baseline_closure_defect_scalar": baseline_defect,
            "finite_candidate_closure_defect_scalar": candidate_defect,
            "finite_delta_scalar": finite_delta,
            "finite_delta_lower_scalar": Decimal(7) / 8 * closure_lower,
            "candidate_below_closure_baseline_lower_scalar": Decimal(223) / 256 * closure_lower,
            "derivative_at_zero": -beta * q * deficit,
            "all_off_entire_restricted_profile": beta / 2 <= gamma,
        }


def population_weights(q: Decimal) -> tuple[Decimal, ...]:
    return tuple(q**sum(state) * (1 - q) ** (DIMENSION - sum(state))
                 for state in SUPPORTS)


def construct_native_model(case: WitnessCase, bias: Decimal,
                           amplitude_floor: Decimal) -> VariationalGarroteSAE:
    """Materialize every parameter on CPU, without drawing a random initializer.

    `meta` construction executes the existing class initialization symbolically.
    No random parameter is sampled on CPU/GPU; all empty CPU tensors are then
    assigned the explicit witness. No subclass or altered free-energy code is used.
    """
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        beta, gamma = Decimal(case.beta), Decimal(case.gamma)
        if not 0 <= bias < 1:
            raise ValueError("Finite witness requires 0 <= bias < 1")
        if amplitude_floor <= 0:
            raise ValueError("Amplitude floor must be positive")
        gate_slope = beta * (1 - bias) ** 2 / 2
        gate_bias = -gamma + gate_slope * bias
        inactive_pre = _inverse_softplus(amplitude_floor)
        active_pre = _inverse_softplus(1 - bias)
        amplitude_slope = active_pre - inactive_pre
        amplitude_bias = inactive_pre + amplitude_slope * bias
        cfg = VGSAEConfig(
            input_dim=DIMENSION, n_latents=DIMENSION,
            beta=float(beta), lambda_sparsity=float(gamma),
            beta_mode="learned", dtype=torch.float64,
            decoder_bias=True, nonnegative_amplitudes=True,
            use_variance_term=True, use_entropy_term=True,
            entropy_weight=1.0, normalize_decoder=True,
        )
        with torch.device("meta"):
            model = VariationalGarroteSAE(cfg)
        model.to_empty(device="cpu")
        with torch.no_grad():
            eye = torch.eye(DIMENSION, dtype=torch.float64, device="cpu")
            model.decoder.weight.copy_(eye)
            model.pre_bias.fill_(float(bias))
            model.log_beta.fill_(float(beta.ln()))
            model.gate_encoder.weight.copy_(eye * float(gate_slope))
            model.gate_encoder.bias.fill_(float(gate_bias))
            model.amplitude_encoder.weight.copy_(eye * float(amplitude_slope))
            model.amplitude_encoder.bias.fill_(float(amplitude_bias))
        model.eval()
        model.requires_grad_(False)
        return model


def state_as_json(model: VariationalGarroteSAE) -> dict:
    return {name: value.detach().cpu().tolist()
            for name, value in model.state_dict().items()}


def _decimal_array(values: list) -> list:
    return [_decimal_array(value) if isinstance(value, list) else _d(value)
            for value in values]


def tensor_population_reference(parameters: dict, q: Decimal,
                                gamma: Decimal) -> dict:
    """Independent real-arithmetic evaluation of the saved finite float tensors.

    Decimal exp/ln at fixed precision is a high-precision reference, not outward
    interval arithmetic. The Gaussian constant uses the same stored 2*pi float
    constant as the native implementation. Population probabilities use exact q.
    """
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        q, gamma = _d(q), _d(gamma)
        decoder = _decimal_array(parameters["decoder.weight"])
        bias = _decimal_array(parameters["pre_bias"])
        gate_weights = _decimal_array(parameters["gate_encoder.weight"])
        gate_bias = _decimal_array(parameters["gate_encoder.bias"])
        amp_weights = _decimal_array(parameters["amplitude_encoder.weight"])
        amp_bias = _decimal_array(parameters["amplitude_encoder.bias"])
        log_beta = _d(parameters["log_beta"])
        beta = log_beta.exp()
        normalizer = -Decimal(DIMENSION) / 2 * (
            log_beta - _d(2.0 * math.pi).ln()
        )
        weighted_objective = Decimal(0)
        records = []
        for state, probability in zip(SUPPORTS, population_weights(q)):
            centered = [Decimal(state[j]) - bias[j] for j in range(DIMENSION)]
            logits = [sum((gate_weights[i][j] * centered[j]
                           for j in range(DIMENSION)), gate_bias[i])
                      for i in range(DIMENSION)]
            amp_pre = [sum((amp_weights[i][j] * centered[j]
                            for j in range(DIMENSION)), amp_bias[i])
                       for i in range(DIMENSION)]
            gates = [_sigmoid(logit) for logit in logits]
            amplitudes = [_softplus(value) for value in amp_pre]
            coefficients = [gates[i] * amplitudes[i] for i in range(DIMENSION)]
            residuals = [centered[j] - sum(decoder[j][i] * coefficients[i]
                                           for i in range(DIMENSION))
                         for j in range(DIMENSION)]
            recon = sum(value**2 for value in residuals) / 2
            variance = sum(
                gates[i] * (1 - gates[i]) * amplitudes[i] ** 2
                * sum(decoder[j][i] ** 2 for j in range(DIMENSION))
                for i in range(DIMENSION)
            ) / 2
            prior = gamma * sum(gates) + DIMENSION * _softplus(-gamma)
            entropy = sum(_softplus(-abs(value))
                          + abs(value) * _sigmoid(-abs(value)) for value in logits)
            objective = beta * (recon + variance) + normalizer + prior - entropy
            weighted_objective += probability * objective
            records.append({
                "support": list(state), "probability": str(probability),
                "objective": str(objective),
                "gates": list(map(str, gates)),
                "amplitudes": list(map(str, amplitudes)),
            })
        return {
            "objective": weighted_objective,
            "effective_beta": beta,
            "gaussian_normalizer": normalizer,
            "states": records,
        }


def evaluate_native_population(model: VariationalGarroteSAE, q: Decimal) -> dict:
    weights = population_weights(q)
    weighted_objective = torch.zeros((), dtype=torch.float64, device="cpu")
    weighted_scale = torch.zeros_like(weighted_objective)
    soft_density = torch.zeros_like(weighted_objective)
    hard_density = torch.zeros_like(weighted_objective)
    hard_squared_error = torch.zeros_like(weighted_objective)
    rows = []
    with torch.no_grad():
        for state, probability in zip(SUPPORTS, weights):
            inputs = torch.tensor([state], dtype=torch.float64, device="cpu")
            terms = model.free_energy(inputs)
            hard, amplitudes, hard_code = model.encode_inference(inputs)
            hard_reconstruction = model.decode(hard_code)
            probability_float = float(probability)
            weighted_objective += probability_float * terms["loss"]
            gaussian = -DIMENSION / 2 * torch.log(
                terms["beta_eff"] / (2.0 * math.pi)
            )
            scale = (abs(terms["beta_eff"] * terms["energy"])
                     + abs(gaussian) + abs(terms["prior"]) + abs(terms["entropy"]))
            weighted_scale += probability_float * scale
            soft_density += probability_float * terms["m"].mean()
            hard_density += probability_float * hard.mean()
            squared_error = (inputs - hard_reconstruction).pow(2).sum()
            hard_squared_error += probability_float * squared_error
            rows.append({
                "support": list(state), "probability_exact": str(probability),
                "probability_float64": probability_float,
                "objective": terms["loss"].item(),
                "mean_reconstruction_half_squared_error": terms["recon"].item(),
                "bernoulli_variance_half_squared_error": terms["variance"].item(),
                "prior": terms["prior"].item(), "entropy": terms["entropy"].item(),
                "effective_beta": terms["beta_eff"].item(),
                "gates": terms["m"][0].tolist(),
                "amplitudes": amplitudes[0].tolist(),
                "gate_logits": terms["gate_logits"][0].tolist(),
                "hard_mask": hard[0].tolist(),
                "posterior_mean_reconstruction": terms["x_hat"][0].tolist(),
                "hard_reconstruction": hard_reconstruction[0].tolist(),
            })
    return {
        "objective": weighted_objective.item(),
        "absolute_component_scale": weighted_scale.item(),
        "expected_soft_density": soft_density.item(),
        "expected_hard_density": hard_density.item(),
        "hard_reconstruction_squared_error": hard_squared_error.item(),
        "states": rows,
    }


def _stringify_decimals(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_decimals(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_decimals(item) for item in value]
    return value


def evaluate_case(case: WitnessCase) -> dict:
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        design = analytic_design(case)
        q, gamma = design["q"], design["gamma"]
        candidate_bias = design["candidate_bias"]
        amplitude_floor = design["inactive_amplitude_floor"]
        models = {
            "baseline": construct_native_model(case, Decimal(0), amplitude_floor),
            "candidate": construct_native_model(case, candidate_bias, amplitude_floor),
        }
        records = {}
        references = {}
        for role, model in models.items():
            state = state_as_json(model)
            native = evaluate_native_population(model, q)
            reference = tensor_population_reference(state, q, gamma)
            references[role] = reference
            records[role] = {
                "config": {**asdict(model.config), "dtype": str(model.config.dtype)},
                "state_dict": state,
                "native_float64": native,
                "actual_tensor_decimal_reference": _stringify_decimals(reference),
                "decoder_distance_to_teacher": 0.0,
            }

        native_baseline = records["baseline"]["native_float64"]
        native_candidate = records["candidate"]["native_float64"]
        float_delta = native_baseline["objective"] - native_candidate["objective"]
        tensor_reference_delta = references["baseline"]["objective"] - references["candidate"]["objective"]
        guard = ROUNDOFF_FACTOR * torch.finfo(torch.float64).eps * max(
            1.0, native_baseline["absolute_component_scale"],
            native_candidate["absolute_component_scale"],
        )
        discrepancies = {
            role: abs(records[role]["native_float64"]["objective"]
                      - float(references[role]["objective"]))
            for role in models
        }
        parity_pass = all(value <= guard for value in discrepancies.values())
        closure_check = design["closure_delta_scalar"] >= design["closure_delta_lower_scalar"]
        finite_check = design["finite_delta_scalar"] >= design["finite_delta_lower_scalar"]
        analytic_pass = bool(closure_check and finite_check and 0 < candidate_bias < q)
        if not analytic_pass or not parity_pass:
            status = "implementation_check_failed"
        elif float_delta > guard and tensor_reference_delta > 0:
            status = "finite_native_decrease_resolved_float64"
        else:
            status = "unresolved_float64_difference"

        normalizer = -Decimal(DIMENSION) / 2 * (
            design["beta"].ln() - _d(2.0 * math.pi).ln()
        )
        closure_expected = {
            "baseline": DIMENSION * design["closure_baseline_scalar"] + normalizer,
            "candidate": DIMENSION * design["closure_candidate_scalar"] + normalizer,
        }
        for role in models:
            # This signed gap includes both the finite amplitude floor and tensor
            # rounding. It is deliberately not substituted for the ideal formula.
            records[role]["actual_tensor_minus_target_closure_objective"] = str(
                references[role]["objective"] - closure_expected[role]
            )

        return {
            "case_id": case.case_id, **asdict(case), "status": status,
            "analytic_design": _stringify_decimals(design),
            "checks": {
                "analytic_descent_bounds_pass": analytic_pass,
                "native_decimal_parity_pass": parity_pass,
                "native_decimal_objective_discrepancies": discrepancies,
                "float64_resolution_guard": guard,
                "guard_is_formal_error_bound": False,
            },
            "population_objective_difference_baseline_minus_candidate": {
                "ideal_finite_family_decimal": str(DIMENSION * design["finite_delta_scalar"]),
                "ideal_finite_family_analytic_lower": str(DIMENSION * design["finite_delta_lower_scalar"]),
                "saved_tensor_decimal_reference": str(tensor_reference_delta),
                "native_float64": float_delta,
                "native_minus_guard": float_delta - guard,
                "formal_numerical_certificate": False,
            },
            "models": records,
            "scope": "Restricted noiseless family; both decoders equal the teacher. No dictionary-gap or global-class conclusion.",
        }


def protocol() -> dict:
    return {
        "version": PROTOCOL_VERSION,
        "q_values": list(Q_VALUES), "beta_values": list(BETA_VALUES),
        "gamma_values": list(GAMMA_VALUES), "case_count": len(frozen_cases()),
        "case_ids": [case.case_id for case in frozen_cases()],
        "dimension": DIMENSION, "supports": [list(state) for state in SUPPORTS],
        "noise": 0, "dictionary": "identity", "population_weight": "q^|s| (1-q)^(2-|s|)",
        "bias_rule": "b = q * sigmoid(gamma - beta / 2)",
        "inactive_amplitude_rule": "epsilon = b / 16, common to baseline and candidate",
        "baseline_bias": "0", "native_threshold": "m > 0.5",
        "decimal_precision": DECIMAL_PRECISION,
        "roundoff_guard_rule": "8192 * float64_eps * max(1, absolute component scales of both models)",
        "roundoff_guard_formally_certified": False,
        "optimizer_steps": 0, "training_steps": 0, "sampled_inputs": 0,
        "random_seeds": [], "adaptive_point_selection": False,
        "device": "cpu", "gpu_computation": False,
        "required_runtime_flag": "VG_SAE_COLAB_CPU_ONLY=1",
        "reject_cuda_available": True,
        "total_process_cpu_seconds": TOTAL_CPU_SECONDS,
        "total_process_wall_seconds": TOTAL_WALL_SECONDS,
        "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
        "overwrite_existing_results": False,
        "all_cases_retained": True,
    }


def source_hashes() -> dict:
    paths = [
        "scripts/check_native_bias_witness.py", "tests/test_native_bias_witness.py",
        "src/sae_model.py", "src/model.py", "src/__init__.py",
        "refine-logs/runs/vg-sae-colab-continuation-20261003/NATIVE_FINITE_BETA_DERIVATION.md",
    ]
    return {path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
            for path in paths}


def _write_outputs(output: Path, result: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / RESULT_FILENAME).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    columns = ["case_id", "q", "beta", "gamma", "status", "candidate_bias",
               "ideal_finite_delta_lower", "saved_tensor_decimal_delta",
               "native_float64_delta", "float64_guard", "all_off_profile"]
    with (output / "native_bias_witness_summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in result["cases"]:
            summary = {key: row.get(key, "") for key in columns}
            if "analytic_design" in row:
                delta = row["population_objective_difference_baseline_minus_candidate"]
                summary.update(
                    candidate_bias=row["analytic_design"]["candidate_bias"],
                    ideal_finite_delta_lower=delta["ideal_finite_family_analytic_lower"],
                    saved_tensor_decimal_delta=delta["saved_tensor_decimal_reference"],
                    native_float64_delta=delta["native_float64"],
                    float64_guard=row["checks"]["float64_resolution_guard"],
                    all_off_profile=row["analytic_design"]["all_off_entire_restricted_profile"],
                )
            writer.writerow(summary)
    counts = result["status_counts"]
    lines = [
        "# Native finite-beta witness: frozen population calculation", "",
        "All 16 prescribed cases are retained. The protocol has four exact support states per case,",
        "no observations sampled, no seeds, and no optimizer updates. Every decoder is the",
        "identity teacher dictionary. This is not a dictionary-gap counterexample.", "",
        f"- Resolved float64 decreases: {counts.get('finite_native_decrease_resolved_float64', 0)}",
        f"- Unresolved float64 differences: {counts.get('unresolved_float64_difference', 0)}",
        f"- Budget-unresolved cases: {sum(value for key, value in counts.items() if key.startswith('budget_'))}",
        f"- Implementation failures: {sum(value for key, value in counts.items() if key.startswith('implementation_'))}",
        "",
        "The analytic lower bound applies to the ideal finite construction. Saved float64",
        "tensors are independently evaluated at 240-digit Decimal precision. The float64",
        "guard is an engineering resolution rule, not an outward-rounding proof. Tiny",
        "differences remain unresolved; no alternative point is substituted.", "",
        "Exact parameters, four-state forwards, population weights, parity discrepancies,",
        "closure defects and source hashes are in native_bias_witness_results.json.", "",
    ]
    with (output / "native_bias_witness_report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def main() -> int:
    output_name = os.environ.get("COLAB_TASK_OUTPUT")
    if not output_name:
        raise SystemExit("Run through the Colab worker with COLAB_TASK_OUTPUT set; no local output default.")
    output = Path(output_name)
    validate_runtime(output)
    torch.set_num_threads(1)
    evaluation_started_wall, evaluation_started_cpu = time.monotonic(), time.process_time()
    frozen_protocol = protocol()
    budget = ExecutionBudget()
    rows = []
    initial_sources = {}
    interrupted_case = False
    current_case_id = None
    budget_reason = None
    try:
        budget.install()
        budget.check()
        initial_sources = source_hashes()
        for case in frozen_cases():
            interrupted_case = False
            budget.check()
            interrupted_case = True
            current_case_id = case.case_id
            try:
                row = evaluate_case(case)
            except BudgetExceeded:
                raise
            except Exception as error:
                # Retain a failed predetermined case; never replace or retry it.
                row = {"case_id": case.case_id, **asdict(case),
                       "status": "implementation_exception",
                       "error_type": type(error).__name__, "error": str(error)}
            rows.append(row)
            interrupted_case = False
            current_case_id = None
    except BudgetExceeded as error:
        budget_reason = str(error)
        if rows and rows[-1]["case_id"] == current_case_id:
            interrupted_case = False
        retain_budget_remainder(rows, budget_reason, interrupted_case)
    finally:
        budget.begin_finalization()
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    final_sources = source_hashes()
    sources_unchanged = final_sources == initial_sources if initial_sources else None
    result = {
        "execution_status": ("completed_with_budget_unresolved" if budget_reason
                             else "completed"),
        "budget_reason": budget_reason,
        "protocol": frozen_protocol,
        "protocol_sha256": hashlib.sha256(json.dumps(
            frozen_protocol, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "source_sha256": initial_sources, "final_source_sha256": final_sources,
        "initial_source_seal_complete": bool(initial_sources),
        "sources_unchanged": sources_unchanged, "cases": rows,
        "status_counts": counts,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "platform": platform.system(), "device": "cpu", "torch_threads": 1,
                    "decimal_precision": DECIMAL_PRECISION},
        "resources": {
            "evaluation_wall_seconds": time.monotonic() - evaluation_started_wall,
            "evaluation_process_cpu_seconds": time.process_time() - evaluation_started_cpu,
            "process_cpu_seconds_before_serialization": time.process_time(),
            "process_wall_seconds_before_serialization": time.monotonic() - _PROCESS_WALL_STARTED,
            "cpu_scope": "whole_process_including_imports",
            "wall_scope": _PROCESS_WALL_SCOPE,
            "cpu_limit_seconds": budget.hard_cpu,
            "wall_limit_seconds": TOTAL_WALL_SECONDS,
            "science_cpu_limit_seconds": budget.science_cpu,
            "science_wall_limit_seconds": budget.science_wall,
            "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
            "final_receipt": "native_bias_witness_runtime_receipt.json",
        },
    }
    _write_outputs(output, result)
    # This receipt includes writing the three principal artifacts. Its own small
    # JSON write is still subject to the hard CPU and wall process deadlines.
    receipt = {
        "process_cpu_seconds": time.process_time(),
        "process_wall_seconds": time.monotonic() - _PROCESS_WALL_STARTED,
        "cpu_scope": "whole_process_including_imports",
        "wall_scope": _PROCESS_WALL_SCOPE,
        "cpu_limit_seconds": budget.hard_cpu,
        "wall_limit_seconds": TOTAL_WALL_SECONDS,
        "includes_principal_output_serialization": True,
        "excludes_receipt_own_serialization": True,
        "sources_unchanged": sources_unchanged,
        "execution_status": result["execution_status"],
    }
    with (output / "native_bias_witness_runtime_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"case_count": len(rows), "status_counts": counts}, sort_keys=True), flush=True)
    return int(sources_unchanged is False
               or any(row["status"].startswith("implementation_") for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
