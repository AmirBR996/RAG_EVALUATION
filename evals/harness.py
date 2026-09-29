import json


def load_goldens(path):
    with open(path) as file:
        return json.load(file)


def summarize_by_metric(result):
    test_results = getattr(result, "test_results", None)

    if test_results is None:
        test_results = result if isinstance(result, list) else []

    buckets = {}

    for test_result in test_results:
        metrics = (
            getattr(test_result, "metrics_data", None)
            or getattr(test_result, "metrics", None)
            or []
        )

        for metric in metrics:
            name = getattr(metric, "name", "unknown")

            if name not in buckets:
                buckets[name] = {
                    "scores": [],
                    "passed": 0,
                    "total": 0,
                }

            buckets[name]["total"] += 1

            score = getattr(metric, "score", None)

            if score is not None:
                buckets[name]["scores"].append(score)

            if getattr(metric, "success", False):
                buckets[name]["passed"] += 1

    summary = {}

    for name, bucket in buckets.items():
        scores = bucket["scores"]
        total = bucket["total"]

        summary[name] = {
            "n": total,
            "pass_rate": (
                100 * bucket["passed"] / total
                if total
                else 0.0
            ),
            "avg_score": (
                sum(scores) / len(scores)
                if scores
                else float("nan")
            ),
            "min_score": min(scores) if scores else float("nan"),
            "max_score": max(scores) if scores else float("nan"),
        }

    return summary


def print_summary(title, summary):
    print("\n" + "=" * 60)
    print(f"{title} (per-metric summary)")
    print("=" * 60)

    for name, summary_data in summary.items():
        avg = (
            f"{summary_data['avg_score']:.2f}"
            if summary_data["avg_score"] == summary_data["avg_score"]
            else "nan"
        )

        print(
            f"  {name:<26} "
            f"pass_rate={summary_data['pass_rate']:5.0f}%  "
            f"avg={avg}  "
            f"n={summary_data['n']}"
        )

    print("=" * 60)
