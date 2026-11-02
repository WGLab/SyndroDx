"""Verify the local handoff without returning dataset contents to tools."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import shutil
import traceback

ROOT = Path(__file__).resolve().parent
FOLDER = Path((ROOT / "local_checkpoints/ACTIVE_SAVE.txt").read_text().strip())


def load(name):
    return json.loads((FOLDER / name).read_text())


def save(name, value):
    path = FOLDER / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def collect_capture_issues(value, prefix=""):
    issues = []
    if isinstance(value, dict):
        for key, item in value.items():
            location = prefix + "/" + key
            if key in {"returncode", "return_code", "exit_code"} and item not in (0, None):
                issues.append({"location": location, "value": item})
            elif "match" in key.lower() and item is False:
                issues.append({"location": location, "value": item})
            elif "changed" in key.lower() and item is True:
                issues.append({"location": location, "value": item})
            elif key.lower() in {"status", "state"} and isinstance(item, str) and item in {"error", "failed", "unavailable", "missing"}:
                issues.append({"location": location, "value": item})
            if isinstance(item, (dict, list)):
                issues.extend(collect_capture_issues(item, location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(collect_capture_issues(item, prefix + "/" + str(index)))
    return issues


def main():
    stamp = datetime.now(ZoneInfo("America/New_York")).isoformat()
    databases = load("SQLITE_RECHECK.json")
    initial_files = load("FILE_HASH_RECHECK.json")
    manifest = load("SNAPSHOT_MANIFEST.json")
    originals = {entry["snapshot"]: entry["path"] for entry in manifest["files"]}
    # A historical database copy has an unresolved structure anomaly. Preserve
    # it and report the limitation; never label that database as restored/tested.
    # All active CPU/Qwen databases and other current databases must still pass.
    historical_failures = [entry for entry in databases if not entry["passed"]
                           and originals[entry["path"]].startswith("face_qc/qwen_stream_")
                           and not originals[entry["path"]].startswith("face_qc/qwen_stream_v5/")]
    historical_paths = {entry["path"] for entry in historical_failures}
    current_databases = [entry for entry in databases if entry["path"] not in historical_paths]
    required = [
        "CURRENT_STATE.md", "CURRENT_STATE.json", "DECISIONS_AND_CONSTRAINTS.md",
        "NEXT_STEPS_AND_ACCEPTANCE.md", "SNAPSHOT_MANIFEST.json", "VALIDATION.json",
        "EXTERNAL_INPUT_SNAPSHOTS.json", "runtime_snapshot/capture_status.json",
        "runtime_snapshot/environment_capture_status.json", "runtime_snapshot/squeue.txt",
        "runtime_snapshot/steps.txt", "runtime_snapshot/sacct.txt", "runtime_snapshot/crontab.txt",
        "runtime_snapshot/filesystem.txt", "runtime_snapshot/processes.txt",
        "runtime_snapshot/faceqc_packages.txt", "runtime_snapshot/qwen_packages.txt",
    ]
    gates = {
        "initial_snapshot_hashes_passed": bool(initial_files) and all(x["passed"] for x in initial_files),
        "current_sqlite_integrity_passed": bool(current_databases) and all(x["passed"] for x in current_databases),
        "required_state_artifacts_present": all((FOLDER / x).is_file() for x in required),
    }
    limitations = []
    limitations.extend({"source": "historical_sqlite_structure_check", "path": entry["path"],
                        "original_reference": originals[entry["path"]],
                        "original_source_available_at_verification": (ROOT / originals[entry["path"]]).is_file(),
                        "detail": "Historical copy preserved and hashed, but SQLite structure validation failed. No recovery is claimed."}
                       for entry in historical_failures)
    for name in ["EXTERNAL_INPUT_SNAPSHOTS.json", "runtime_snapshot/capture_status.json",
                 "runtime_snapshot/environment_capture_status.json"]:
        if (FOLDER / name).is_file():
            limitations.extend({"source": name, **issue} for issue in collect_capture_issues(load(name)))
    limitations.extend({"source": "source_changed_during_copy", "path": entry["path"]}
                       for entry in manifest["files"] if entry["source_changed_during_copy"])
    if not all(gates.values()):
        save("CHECKPOINT_ACCEPTANCE.json", {"state": "verification_failed", "verified_at": stamp,
                                            "gates": gates, "limitations": limitations})
        print("CHECKPOINT_VERIFICATION_FAILED_DETAILS_LOCAL", flush=True)
        return 1

    # Preserve the latest save/verification utilities and safe handoff index.
    index = (
        "# 本地完整状态记录\n\n"
        "详细内容仅供 HPC 本地查看，不能打印到 OpenAI 对话。\n\n"
        f"- [当前进度]({FOLDER}/CURRENT_STATE.md)\n"
        f"- [完整结构化状态]({FOLDER}/CURRENT_STATE.json)\n"
        f"- [决策与约束]({FOLDER}/DECISIONS_AND_CONSTRAINTS.md)\n"
        f"- [接续步骤与验收]({FOLDER}/NEXT_STEPS_AND_ACCEPTANCE.md)\n"
        f"- [原始状态文件清单]({FOLDER}/SNAPSHOT_MANIFEST.json)\n"
        f"- [最终备份文件与哈希]({FOLDER}/FINAL_CHECKPOINT_MANIFEST.json)\n"
        f"- [最终保存验收]({FOLDER}/CHECKPOINT_ACCEPTANCE.json)\n"
        f"- [外部输入版本备份]({FOLDER}/EXTERNAL_INPUT_SNAPSHOTS.json)\n"
        f"- [资源与环境捕获状态]({FOLDER}/runtime_snapshot/capture_status.json)\n\n"
        "这是实验状态备份；原始照片和大型模型资产保留在原位置。"
        "原图审核与 Qwen 标注已完成，最终训练数据未完成，新的批量裁剪／质控任务尚未提交。"
        "没有删除数据、日志或聊天记录。最终验收文件明确列出保存边界及限制。\n"
    )
    (ROOT / "SAVED_STATE.md").write_text(index)
    resume_path = ROOT / "LOCAL_RESUME.md"
    resume_note = ("\nThe local checkpoint has a documented structure-check limitation in a historical database copy. "
                   "Current production databases passed their checks. Preserve historical evidence and consult the local acceptance "
                   "record through a privacy-safe program; do not claim all historical databases can be restored.\n")
    if historical_failures and resume_note.strip() not in resume_path.read_text():
        with resume_path.open("a") as handle:
            handle.write(resume_note)
    for name in ["save_complete_local_state.py", "verify_local_checkpoint.py", "SAVED_STATE.md", "LOCAL_RESUME.md"]:
        source = ROOT / name
        target = FOLDER / "workspace_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        os.chmod(target, 0o600)
        entry = {"path": name, "snapshot": str(target.relative_to(FOLDER)), "bytes": target.stat().st_size,
                 "sha256": digest(target), "kind": "final_handoff_refresh", "source_mtime": source.stat().st_mtime,
                 "source_changed_during_copy": False}
        manifest["files"] = [x for x in manifest["files"] if x["path"] != name] + [entry]
    manifest["files_copied"] = len(manifest["files"])
    manifest["bytes_copied"] = sum(x["bytes"] for x in manifest["files"])
    save("SNAPSHOT_MANIFEST.json", manifest)

    state = load("CURRENT_STATE.json")
    state["checkpoint_verification"] = {"verified_at": stamp, "gates": gates, "limitations": limitations,
                                        "authoritative_record": "CHECKPOINT_ACCEPTANCE.json"}
    state["snapshot_files"] = manifest["files_copied"]
    state["snapshot_bytes"] = manifest["bytes_copied"]
    save("CURRENT_STATE.json", state)
    with (FOLDER / "CURRENT_STATE.md").open("a") as handle:
        handle.write("\n## 最终保存验收\n\n"
                     f"验收时间：{stamp}。状态文件哈希及当前生产数据库复检通过；历史数据库副本的结构校验异常作为明确限制保留，未声称修复。调度、软件环境和外部输入版本另行保存。"
                     "最终结果以 CHECKPOINT_ACCEPTANCE.json 为准；其中记录捕获限制。"
                     "本地保存完成不表示最终训练数据已完成。\n")

    skipped = {"FINAL_CHECKPOINT_MANIFEST.json", "CHECKPOINT_ACCEPTANCE.json", "SAVE_STATUS.json"}
    final_entries = []
    stable = True
    for path in sorted(FOLDER.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.relative_to(FOLDER).as_posix() in skipped:
            continue
        before = path.stat()
        value = digest(path)
        after = path.stat()
        unchanged = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        stable = stable and unchanged
        os.chmod(path, 0o600)
        final_entries.append({"path": str(path.relative_to(FOLDER)), "bytes": after.st_size,
                              "sha256": value, "unchanged_while_hashing": unchanged})
    by_path = {x["path"]: x for x in final_entries}
    gates["final_snapshot_matches_saved_manifest"] = all(
        entry["snapshot"] in by_path and by_path[entry["snapshot"]]["sha256"] == entry["sha256"]
        and by_path[entry["snapshot"]]["bytes"] == entry["bytes"] for entry in manifest["files"])
    gates["final_checkpoint_files_stable"] = stable
    save("FINAL_CHECKPOINT_MANIFEST.json", {"verified_at": stamp, "files": final_entries,
                                          "excluded_self_and_final_status": sorted(skipped)})
    passed = all(gates.values())
    status = "verified" if passed and not limitations else "verified_with_capture_limitations" if passed else "verification_failed"
    save("CHECKPOINT_ACCEPTANCE.json", {"state": status, "verified_at": stamp,
                                        "gates": gates, "limitations": limitations,
                                        "boundary": "Experiment-state checkpoint, not a full source-photo/model archive or complete chat export.",
                                        "no_source_files_deleted": True, "no_dataset_content_returned": True})
    save("SAVE_STATUS.json", {"state": status, "verified_at": stamp,
                              "manifest_sha256": digest(FOLDER / "FINAL_CHECKPOINT_MANIFEST.json"),
                              "acceptance_sha256": digest(FOLDER / "CHECKPOINT_ACCEPTANCE.json")})
    print("COMPLETE_LOCAL_CHECKPOINT_VERIFIED" if passed and not limitations else
          "LOCAL_CHECKPOINT_VERIFIED_CAPTURE_LIMITATIONS_RECORDED" if passed else
          "CHECKPOINT_VERIFICATION_FAILED_DETAILS_LOCAL", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        with (FOLDER / "VERIFICATION_ERROR.log").open("a") as handle:
            traceback.print_exc(file=handle)
        print("CHECKPOINT_VERIFICATION_ERROR_DETAILS_LOCAL", flush=True)
        raise SystemExit(1)
