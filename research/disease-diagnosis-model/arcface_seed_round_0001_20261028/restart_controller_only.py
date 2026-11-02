"""Replace only this comparison's CPU coordinator; GPU worker steps stay alive."""
import os,signal
from pathlib import Path
target=str(Path(__file__).resolve().parent/'controller.py').encode()
found=[]
for folder in Path('/proc').iterdir():
    if not folder.name.isdigit():continue
    try:
        if folder.stat().st_uid!=os.getuid():continue
        args=(folder/'cmdline').read_bytes().split(b'\x00')
        if target in args and b'python' in args[0]:found.append(int(folder.name))
    except (OSError,IndexError):pass
if len(found)!=1:raise RuntimeError('unique_matching_controller_required')
os.kill(found[0],signal.SIGTERM)
print('Only the matching CPU controller was signalled; no GPU allocation or worker was targeted.')
