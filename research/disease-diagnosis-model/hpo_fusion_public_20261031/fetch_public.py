"""Download public resources only. No local project data leave this process."""
import urllib.request
from common import WORK, save, digest, now

RESOURCES = {
    'phenopackets_0.1.27.zip': 'https://github.com/monarch-initiative/phenopacket-store/releases/download/0.1.27/all_phenopackets.zip',
    'rarebench_data.zip': 'https://huggingface.co/datasets/chenxz/RareBench/resolve/main/data.zip',
    'rarebench_loader.py.txt': 'https://huggingface.co/datasets/chenxz/RareBench/resolve/main/RareBench.py',
    'orphadata_phenotypes.xml': 'https://www.orphadata.com/data/xml/en_product4.xml',
}


def main():
    receipt = {}
    for name, url in RESOURCES.items():
        target = WORK / 'public' / name
        if not target.exists():
            request = urllib.request.Request(url, headers={'User-Agent': 'craniofacial-HPO-research/1.0'})
            temp = target.with_suffix('.download')
            with urllib.request.urlopen(request, timeout=90) as response, temp.open('wb') as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            temp.replace(target)
        receipt[name] = dict(url=url, bytes=target.stat().st_size, sha256=digest(target))
        print(name, target.stat().st_size, flush=True)
    save(WORK / 'public_download_receipt.json', dict(state='complete', sources=receipt,
         outbound_patient_data=False, updated_at_utc=now()))


if __name__ == '__main__':
    main()
