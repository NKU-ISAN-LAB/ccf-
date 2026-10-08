"""Build a checksum-pinned reference bank from explicitly reviewed frames."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,required=True,help='Original dataset directory')
DATA=parser.parse_args().data
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
spec=json.loads((ROOT/'reference_specs.json').read_text())
(ROOT/'reference').mkdir(exist_ok=True)
for ref in spec['references']:
    source=DATA/'color'/f"{ref['frame']}.jpg"
    ref['sha256']=sha(source)
    shutil.copyfile(source,ROOT/'reference'/source.name)
(ROOT/'reference/manifest.json').write_text(json.dumps(spec,indent=2))
