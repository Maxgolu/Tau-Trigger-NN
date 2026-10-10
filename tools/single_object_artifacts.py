"""Verify/restore archived single-object artifacts. Standard library only; no ML."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'results/single_object'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verified_members(package=PACKAGE):
    manifest = json.loads((package / 'manifest.json').read_text(encoding='utf-8'))
    if digest((package / 'test_counts.json').read_text(encoding='utf-8').encode('utf-8')) != manifest['test_counts_sha256']:
        raise ValueError('test-count binding mismatch')
    for name, expected in manifest.get('metadata_bindings_lf', {}).items():
        if digest((package/name).read_text(encoding='utf-8').encode('utf-8')) != expected:
            raise ValueError(f'metadata binding mismatch: {name}')
    members = {}
    for name, expected in manifest['archives'].items():
        path = package / name
        if digest(path.read_bytes()) != expected:
            raise ValueError(f'archive binding mismatch: {name}')
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                rel = PurePosixPath(item.filename)
                if rel.is_absolute() or '..' in rel.parts or '\\' in item.filename or ':' in item.filename:
                    raise ValueError(f'unsafe archive path: {item.filename}')
                if item.filename in members:
                    raise ValueError(f'duplicate archive path: {item.filename}')
                data = archive.read(item)
                binding = manifest['artifacts'][item.filename]
                if len(data) != binding['bytes'] or digest(data) != binding['sha256']:
                    raise ValueError(f'artifact binding mismatch: {item.filename}')
                members[item.filename] = data
    if set(members) != set(manifest['artifacts']):
        raise ValueError('artifact manifest/archive membership mismatch')
    for dep in manifest['ft_dependencies']:
        if dep['weights_path'] not in members:
            raise ValueError(f'missing FT prerequisite: {dep}')
    return manifest, members


def restore(members, destination, prefix):
    destination = Path(destination).resolve()
    planned = []
    for name, data in members.items():
        if not name.startswith(prefix):
            continue
        target = (destination / name).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f'path escapes destination: {name}')
        if target.exists():
            if not target.is_file() or target.read_bytes() != data:
                raise FileExistsError(f'Refusing to replace existing, different artifact: {target}')
        else:
            planned.append((target, data))
    # Check every destination before any write. Existing identical files are left alone.
    for target, data in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(data)
    return len(planned)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'restore-pretrained', 'restore-metrics'])
    args = parser.parse_args()
    manifest, members = verified_members()
    print(f"Verified {len(manifest['ft_dependencies'])} FT references and {len(manifest['metric_runs'])} archived metric runs.")
    if args.action == 'restore-pretrained':
        print(f"Restored {restore(members, ROOT, 'experiments/')} files under experiments/.")
    elif args.action == 'restore-metrics':
        dest = ROOT / 'results/generated/single_object'
        print(f"Restored {restore(members, dest, 'metrics/')} files under {dest}.")


if __name__ == '__main__':
    main()
