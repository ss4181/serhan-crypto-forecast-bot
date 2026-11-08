"""Inventory saved Trade3 experiments without treating repeated tests as validation.

This audits recorded gate claims/provenance, not historical fills or model
implementation. A positive claim always requires a separate numeric audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SKIP_ROWS = {"trades", "selected_rows", "out_of_sample_predictions", "source_manifest",
             "manifest", "open_trades", "by_symbol"}


def gate_maps(value: Any, path: str = "$") -> list[dict]:
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if "gate" in key.lower() and isinstance(child, dict):
                if child and all(type(v) is bool for v in child.values()):
                    result.append({"path": child_path, "checks": child,
                                   "claim": "ALL_RECORDED_CHECKS_PASS" if all(child.values()) else "RECORDED_CHECK_FAILED"})
                    continue
                if not child:
                    result.append({"path": child_path, "checks": {}, "claim": "UNKNOWN_EMPTY_GATES"})
            if key not in SKIP_ROWS:
                result.extend(gate_maps(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(gate_maps(child, f"{path}[{index}]"))
    return result


def audit(root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Preserve previous audit: output exists")
    paths = sorted(set(root.glob("*/study.json")) | set(root.glob("*/results/study.json")))
    records, duplicates, failure_counts = [], defaultdict(list), Counter()
    for path in paths:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        name = str(path.relative_to(root)).replace("\\", "/")
        duplicates[digest].append(name)
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeError):
            records.append({"file": name, "sha256": digest, "status": "UNKNOWN_UNREADABLE"})
            continue
        claims = gate_maps(data)
        for claim in claims:
            failure_counts.update(k for k, v in claim["checks"].items() if v is False)
        passes = [c["path"] for c in claims if c["claim"] == "ALL_RECORDED_CHECKS_PASS"]
        report = path.with_name("report.md")
        records.append({"file": name, "sha256": digest,
                        "status": "PASS_CLAIM_NEEDS_NUMERIC_AND_INDEPENDENT_AUDIT" if passes else
                                  "NO_RECORDED_PASS" if claims else "UNKNOWN_NO_GATE_MAP",
                        "gate_maps": claims, "pass_claim_paths": passes,
                        "explicit_independent_validation": data.get("independent_validation"),
                        "report_exists": report.exists(),
                        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest() if report.exists() else None,
                        "fresh_in_filename_is_not_validation": "fresh" in name})
    summary = {"study_files": len(records),
               "statuses": dict(Counter(r["status"] for r in records)),
               "recorded_gate_maps": sum(len(r.get("gate_maps", [])) for r in records),
               "all_recorded_checks_pass_claims": sum(len(r.get("pass_claim_paths", [])) for r in records),
               "identical_json_file_groups": [names for names in duplicates.values() if len(names) > 1],
               "failed_check_occurrences": dict(failure_counts.most_common()),
               "warning": "Gate maps include repeated selections, weekly folds and sensitivity cuts. Counts are NOT independent experiments or trades.",
               "completion_proven": False}
    result = {"summary": summary, "records": records,
              "audit_scope": "Saved claim/provenance inventory only. Not a re-execution of all historical studies.",
              "next_gate": "Any candidate must pass original numeric gates AND independent chronological validation before implementation/activation."}
    output.mkdir(parents=True)
    (output / "registry.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Trade3 araştırma kanıt envanteri", "",
             f"Kaydedilmiş çalışma dosyası: {len(records)}. Kontrol kümesi: {summary['recorded_gate_maps']}.",
             f"Tüm kayıtlı kontrolleri geçen sonuç iddiası: {summary['all_recorded_checks_pass_claims']}.",
             "", "Bu sayılar bağımsız deney sayısı değildir; aynı veriyi kullanan varyantlar, haftalar ve tekrar raporları içerir.",
             "Bu araç hesaplanan getirileri yeniden üretmez; kayıtlı başarı iddialarını ve dosya bütünlüğünü denetler.",
             "Boş/eksik kontrol kümesi UNKNOWN'dur. Dosya adındaki fresh/oos, bağımsız doğrulama kanıtı değildir.",
             "", "| Çalışma dosyası | Kayıtlı kontrol kümesi | Durum |", "|---|---:|---|"]
    for row in records:
        lines.append(f"| {row['file']} | {len(row.get('gate_maps', []))} | {row['status']} |")
    lines += ["", "## Sonraki adım", "",
              "Geçmişte olumlu görünen herhangi bir aday önce sayı/örneklem/maliyet/güven aralığı denetiminden, ardından bağımsız kronolojik sınamadan geçmeli. Envanterden doğrudan canlıya terfi yok."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/research"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.root, args.output)["summary"], indent=2))
