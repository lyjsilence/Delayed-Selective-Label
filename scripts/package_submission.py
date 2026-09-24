"""Create a code archive without Git history, caches, or generated runs."""

from pathlib import Path
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "dist" / "delayed-selective-label-simulation.zip"
    output.parent.mkdir(exist_ok=True)
    directories = ["configs", "src", "scripts", "tests"]
    files = [ROOT / name for name in ["README.md", "requirements.txt", "pyproject.toml"]]
    for directory in directories:
        files.extend(
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and "fixtures" not in p.parts
            and p.name != "check_results.py"
            and p.suffix not in {".pyc", ".nbc", ".nbi"}
        )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            entry = zipfile.ZipInfo(
                str(path.relative_to(ROOT)).replace("\\", "/"), date_time=(2026, 1, 1, 0, 0, 0)
            )
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, path.read_bytes())
    print(output)
    print("SHA256:", hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
