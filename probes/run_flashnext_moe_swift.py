"""Compile the MoE verifier against the existing production Swift build."""
import argparse
from pathlib import Path
import subprocess
import tempfile

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('fixtures', type=Path)
ap.add_argument('--backend', type=Path, default=Path.home() / '.mlx128/Rindi-NativeMLX')
args = ap.parse_args()
root = args.backend.resolve()
build = root / '.build/release'
fixtures = sorted(args.fixtures.glob('*.safetensors'))
if not fixtures:
    raise SystemExit('No Swift fixtures found')
with tempfile.TemporaryDirectory(prefix='ane-moe-swift-') as tmp:
    output = Path(tmp) / 'verify'
    command = ['xcrun', 'swiftc', str(Path(__file__).with_name('flashnext_moe_swift.swift')),
               '-o', str(output)]
    for directory in (build, build / 'include',
                      root / '.build/checkouts/swift-numerics/Sources/_NumericsShims/include',
                      root / 'vendor/mlx-swift/Source/Cmlx/include'):
        command += ['-I', str(directory)]
    command += [str(p) for p in sorted(build.glob('*.o'))]
    command += [str(build / 'libtokenizers_rust-macos.a'), '-lc++']
    for framework in ('Foundation', 'Metal', 'Accelerate'):
        command += ['-framework', framework]
    subprocess.run(command, check=True)
    # Preserve the production precompiled Metal kernels used for parity.
    for name in ('mlx.metallib', 'mlx-swift_Cmlx.bundle'):
        if (build / name).exists():
            (Path(tmp) / name).symlink_to(build / name)
    subprocess.run([str(output), *(str(p.resolve()) for p in fixtures)], check=True)
