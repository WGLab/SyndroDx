"""Version-pinned, offline gene knowledge lookup. No disease-name fuzzy matching."""
import csv
import itertools
import sqlite3
from contextlib import closing
import xml.etree.ElementTree as ET
from common import *

ASSETS = PROJECT / 'variant_knowledge_20261101'


def clingen(path):
    with path.open() as f:
        for line in f:
            if 'GENE SYMBOL' in line:
                yield from csv.DictReader(itertools.chain([line], f))
                return
    raise ValueError('missing_public_csv_header')


def lookup_gene(symbol, path=None):
    path = Path(path) if path else ROOT / 'public_gene_knowledge.sqlite'
    if path == ROOT / 'public_gene_knowledge.sqlite' and sha(path) != load(ROOT / 'knowledge_receipt.json')['index_sha256']:
        raise ValueError('public_knowledge_index_changed')
    with closing(sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)) as db:
        return [{'source': source, 'source_sha256': digest, 'assertion': json.loads(payload),
                 'evidence_type': 'public_gene_knowledge_not_patient_observation'}
                for source, digest, payload in db.execute(
                    'SELECT source,source_sha256,payload FROM knowledge WHERE gene=? ORDER BY source,payload',
                    (symbol.strip().upper(),))]


def main():
    downloads = load(ASSETS / 'download_receipt.json')
    sources = ['clingen_gene_validity.csv', 'clingen_gene_dosage.csv', 'orphadata_genes.xml']
    for name in sources:
        if sha(ASSETS / 'public' / name) != downloads['sources'][name]['sha256']:
            raise ValueError('public_source_hash_mismatch')
    dest = ROOT / 'public_gene_knowledge.sqlite'
    temp = dest.with_name(dest.name + '.tmp.' + str(os.getpid()))
    counts = {}
    with sqlite3.connect(temp) as db:
        db.execute('CREATE TABLE knowledge(gene TEXT, source TEXT, source_sha256 TEXT, payload TEXT)')
        for name in sources[:2]:
            count = 0
            for row in clingen(ASSETS / 'public' / name):
                gene = row['GENE SYMBOL'].strip().upper()
                if gene:
                    db.execute('INSERT INTO knowledge VALUES(?,?,?,?)',
                               (gene, name, downloads['sources'][name]['sha256'], json.dumps(row, sort_keys=True)))
                    count += 1
            counts[name] = count
        count = 0
        name = sources[2]
        for disorder in ET.parse(ASSETS / 'public' / name).iter('Disorder'):
            for association in disorder.findall('./DisorderGeneAssociationList/DisorderGeneAssociation'):
                gene = (association.findtext('./Gene/Symbol') or '').strip().upper()
                if gene:
                    row = {'disease_id': 'ORPHA:' + disorder.findtext('OrphaCode'),
                           'disease_name': disorder.findtext('Name'),
                           'association_type': association.findtext('./DisorderGeneAssociationType/Name'),
                           'association_status': association.findtext('./DisorderGeneAssociationStatus/Name')}
                    db.execute('INSERT INTO knowledge VALUES(?,?,?,?)',
                               (gene, name, downloads['sources'][name]['sha256'], json.dumps(row, sort_keys=True)))
                    count += 1
        counts[name] = count
        db.execute('CREATE INDEX gene_index ON knowledge(gene)')
        genes = db.execute('SELECT COUNT(DISTINCT gene) FROM knowledge').fetchone()[0]
    temp.chmod(0o600)
    temp.replace(dest)
    result = {'state': 'offline_public_gene_index_ready', 'rows_by_source': counts,
              'distinct_exact_gene_symbols': genes,
              'source_sha256': {n: downloads['sources'][n]['sha256'] for n in sources},
              'index_sha256': sha(dest), 'disease_namespaces_preserved': True,
              'gene_alias_expansion': False, 'patient_records_used': False,
              'gnomad_integrated': False, 'updated_at_utc': now()}
    save(ROOT / 'knowledge_receipt.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    guarded_main(main)
