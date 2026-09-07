"""
Incident loading. Two sources:

  - make_synthetic_incident(): a hand-built example with a known ground
    truth, used to sanity-check the pipeline for free.

  - load_rcaeval_case(...): TODO once you've downloaded RCAEval RE1.
    Fill this in once you've inspected the actual file format -- the
    incident dict shape below (topology / telemetry / alert_text /
    ground_truth) is the contract the rest of the code expects, so as
    long as this function returns that shape, nothing else needs to change.
"""
import os
import random


def make_synthetic_incident(seed: int = 0) -> dict:
    """
    Recreates the payment-service-connection-leak example from the proposal:
    a leak in payment-service causes cascading latency in order-service and
    api-gateway, with database CPU rising last, as an innocent bystander.
    """
    random.seed(seed)

    def series(base, spike_at, spike_size, length=20):
        return [(t, base + (spike_size if t >= spike_at else 0) + random.uniform(-1, 1))
                for t in range(length)]

    topology = {
        "api-gateway": ["order-service"],
        "order-service": ["payment-service", "database"],
        "payment-service": [],
        "database": [],
    }

    telemetry = {
        "metrics": {
            "payment-service": {"connection_pool_usage": series(20, 3, 60)},
            "order-service": {"latency_ms": series(50, 5, 80)},
            "api-gateway": {"latency_ms": series(30, 7, 70)},
            "database": {"cpu_percent": series(40, 9, 50)},
        },
        "logs": {
            "payment-service": [
                "t=3 connection pool exhausted, retrying",
                "t=4 failed to acquire connection after 3 retries",
            ],
            "order-service": ["t=6 upstream payment-service timeout"],
            "api-gateway": [],
            "database": [],
        },
    }

    return {
        "incident_id": "synthetic-001",
        "alert_text": "Checkout is extremely slow.",
        "topology": topology,
        "telemetry": telemetry,
        "ground_truth": "payment-service",
    }


def load_rcaeval_case(case_path: str, window_before: int = 120, window_after: int = 120) -> dict:
    """
    Loads one RCAEval RE1 case, based on the REAL observed format (confirmed
    by running inspect_case.py on downloaded data -- this differs from the
    official README, which describes an older/different layout):

      data/RE1/RE1-OB/{service}_{fault}/{instance}/data.csv
      data/RE1/RE1-OB/{service}_{fault}/{instance}/inject_time.txt

    data.csv is WIDE format: one row per unix timestamp, one column per
    (service, metric_type) pair, named like "cartservice_cpu",
    "frontend_latency", "PassthroughCluster_load". There's also a duplicate
    "time" column (pandas renames the 2nd one to "time.1" -- we just drop it).

    inject_time.txt contains a single unix timestamp: when the fault was
    injected. We use it to trim the data.csv time range down to a window
    around the injection (default: 2 minutes before/after) rather than
    loading all ~4200 rows -- this keeps things fast and keeps the
    "anomaly onset" logic in evidence.py meaningful (it doesn't need hours
    of pre-fault baseline).

    Ground truth root cause service and fault type come from the PARENT
    folder name (e.g. "productcatalogservice_cpu"), NOT the case_path
    itself -- the case_path's last component is just the repetition number
    (1-5), not part of the label.
    """
    import pandas as pd
    from .topology import get_topology

    case_path = case_path.rstrip("/")
    instance = os.path.basename(case_path)                      # e.g. "1"
    parent_folder = os.path.basename(os.path.dirname(case_path))  # e.g. "productcatalogservice_cpu"

    # RE1 fault types are single words (cpu, mem, disk, delay, loss), so the
    # service name is everything before the LAST underscore.
    if "_" not in parent_folder:
        raise ValueError(f"Unexpected folder name '{parent_folder}', expected '{{service}}_{{fault}}'.")
    ground_truth_service, fault_type = parent_folder.rsplit("_", 1)

    # Figure out which system this is from the path (RE1-OB / RE1-SS / RE1-TT)
    if "RE1-OB" in case_path:
        system_code = "ob"
    elif "RE1-SS" in case_path:
        system_code = "ss"
    elif "RE1-TT" in case_path:
        system_code = "tt"
    else:
        raise ValueError(f"Can't tell which system this case belongs to: {case_path}")

    topology = get_topology(system_code)

    with open(os.path.join(case_path, "inject_time.txt")) as f:
        inject_time = int(f.read().strip())

    df = pd.read_csv(os.path.join(case_path, "data.csv"))
    df = df.loc[:, ~df.columns.duplicated()]  # drop the duplicate "time" column

    # Trim to a window around the fault injection.
    df = df[(df["time"] >= inject_time - window_before) & (df["time"] <= inject_time + window_after)]

    known_metric_suffixes = {"cpu", "mem", "load", "latency", "error"}
    metrics = {}
    for col in df.columns:
        if col == "time":
            continue
        if "_" not in col:
            continue
        svc, metric_type = col.rsplit("_", 1)
        if metric_type not in known_metric_suffixes:
            continue  # skip anything we don't recognize rather than guessing
        series = list(zip(df["time"].tolist(), df[col].tolist()))
        metrics.setdefault(svc, {})[metric_type] = series

    return {
        "incident_id": f"{parent_folder}_{instance}",
        "alert_text": f"Fault detected (type={fault_type}) around service {ground_truth_service}",
        "topology": topology,
        "telemetry": {"metrics": metrics, "logs": {}},  # RE1 has no logs
        "ground_truth": ground_truth_service,
        "inject_time": inject_time,  # kept in case you want to sanity-check onset detection against it
    }

