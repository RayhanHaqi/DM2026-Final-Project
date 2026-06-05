"""Track submitted CSV hashes to avoid duplicate Kaggle slot burns."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

DEFAULT_LEDGER = "output/kaggle_submission_ledger.jsonl"


def file_md5(path, nbytes=65536):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(nbytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_entries(ledger_path):
    if not os.path.isfile(ledger_path):
        return []
    entries = []
    with open(ledger_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def check_duplicate(path, ledger_path=DEFAULT_LEDGER):
    digest = file_md5(path)
    basename = os.path.basename(path)
    for entry in load_entries(ledger_path):
        if entry.get("md5") == digest or entry.get("filename") == basename:
            return True, entry
    return False, {"md5": digest, "filename": basename}


def record_submission(path, description="", ledger_path=DEFAULT_LEDGER):
    digest = file_md5(path)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "filename": os.path.basename(path),
        "path": os.path.abspath(path),
        "md5": digest,
        "description": description,
    }
    os.makedirs(os.path.dirname(ledger_path) or ".", exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Kaggle submission duplicate guard.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Exit 1 if file already in ledger.")
    check.add_argument("path", help="Candidate CSV path")
    check.add_argument("--ledger", default=DEFAULT_LEDGER)

    record = sub.add_parser("record", help="Append file hash to ledger after submit.")
    record.add_argument("path")
    record.add_argument("--description", default="")
    record.add_argument("--ledger", default=DEFAULT_LEDGER)

    list_cmd = sub.add_parser("list", help="Show recent ledger entries.")
    list_cmd.add_argument("--ledger", default=DEFAULT_LEDGER)
    list_cmd.add_argument("-n", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "check":
        dup, info = check_duplicate(args.path, args.ledger)
        if dup:
            print("DUPLICATE", json.dumps(info))
            raise SystemExit(1)
        print("OK", json.dumps(info))
        return

    if args.command == "record":
        row = record_submission(args.path, args.description, args.ledger)
        print("RECORDED", json.dumps(row))
        return

    entries = load_entries(args.ledger)[-args.n :]
    for entry in entries:
        print(json.dumps(entry))


if __name__ == "__main__":
    main()
