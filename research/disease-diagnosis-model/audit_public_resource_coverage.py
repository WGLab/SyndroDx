"""Aggregate local public-resource snapshots against the craniofacial MONDO inventory.

Read-only upstream audit. No patient-level records are exported. Exact mappings
require active MONDO equivalentTo/exactMatch relations; ambiguous mappings are
excluded. Descendant coverage is reported separately, never as exact coverage.
"""
import csv
import hashlib
import json
from itertools import dropwhile
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
BASE = Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs')
OUT = HERE / 'public_resource_coverage'


def read_tsv(path):
    with path.open() as f:
        yield from csv.DictReader(dropwhile(lambda line: line.startswith('#'), f), delimiter='\t')


def write_tsv(name, rows):
    if not rows:
        return
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    inventory_path = HERE / 'diagnosis_strata/craniofacial_mondo_counts.tsv'
    ontology_path = HERE / 'ontology/mondo.obo'
    hpo_path = BASE / 'HPO/phenotype.hpoa'
    packet_path = BASE / 'phenopacket_store/phenopacket_store.summary.tsv'
    orpha_path = BASE / 'Orphanet/v4.8_20260422/data/raw/en_product4.xml'
    gmdb_root = BASE / 'GMDB_v1.1.0/gmdb_metadata_v1.1.0/gmdb_metadata'
    inventory = list(read_tsv(inventory_path))
    target = {r['mondo_id'] for r in inventory}
    terms, xrefs, lexical = {}, defaultdict(set), defaultdict(set)
    norm = lambda s: ' '.join(s.casefold().split())
    for block in ontology_path.read_text().split('[Term]')[1:]:
        mid = re.search(r'^id: (MONDO:\d{7})$', block, re.M)
        name = re.search(r'^name: (.+)$', block, re.M)
        if not mid or not name or 'is_obsolete: true' in block:
            continue
        mid = mid[1]
        terms[mid] = {'name': name[1], 'parents': re.findall(r'^is_a: (MONDO:\d{7})', block, re.M)}
        lexical[norm(name[1])].add(mid)
        for synonym in re.findall(r'^synonym: "(.+?)" EXACT', block, re.M):
            lexical[norm(synonym)].add(mid)
        for line in block.splitlines():
            m = re.match(r'xref: ((?:OMIM|Orphanet|ORDO):\d+)', line)
            if m and 'MONDO:equivalentTo' in line:
                xrefs[m[1].replace('ORDO:', 'Orphanet:')].add(mid)
            m = re.match(r'property_value: skos:exactMatch ((?:OMIM|Orphanet|ORDO):\d+)', line)
            if m:
                xrefs[m[1].replace('ORDO:', 'Orphanet:')].add(mid)
    assert len(target) == 165 and target <= terms.keys()
    mapping_status = Counter()

    def resolve(identifier):
        identifier = identifier.replace('ORPHA:', 'Orphanet:').replace('ORPHANET:', 'Orphanet:').replace('ORDO:', 'Orphanet:')
        matches = {identifier} if identifier in terms else xrefs.get(identifier, set())
        if len(matches) != 1:
            mapping_status['ambiguous' if matches else 'unmapped'] += 1
            return None
        return next(iter(matches))

    @lru_cache(None)
    def ancestors(mid):
        result = set()
        for parent in terms.get(mid, {}).get('parents', []):
            result.add(parent)
            result.update(ancestors(parent))
        return frozenset(result)

    positive, negative, hpo_sources = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in read_tsv(hpo_path):
        mid = resolve(row['database_id'])
        if not mid:
            continue
        (negative if row['qualifier'] == 'NOT' else positive)[mid].add(row['hpo_id'])
        hpo_sources[mid].add(row['database_id'])
    print('HPO parsed', flush=True)

    orpha_positive, orpha_negative = defaultdict(set), defaultdict(set)
    orpha_root = ET.parse(orpha_path).getroot()
    for disorder in orpha_root.findall('.//HPODisorderSetStatus/Disorder'):
        mid = resolve('Orphanet:' + disorder.findtext('OrphaCode', ''))
        if not mid:
            continue
        for assoc in disorder.findall('./HPODisorderAssociationList/HPODisorderAssociation'):
            hp = assoc.findtext('HPO/HPOId', '')
            frequency = assoc.findtext('HPOFrequency/Name', '')
            if hp:
                (orpha_negative if frequency == 'Excluded (0%)' else orpha_positive)[mid].add(hp)

    # Summary records are individual source cases, not proven unique people across papers.
    cases, publications = defaultdict(set), defaultdict(set)
    packet_rows = list(read_tsv(packet_path))
    packet_missing = 0
    missing_packet_files = []
    packet_digest = hashlib.sha256()
    packet_files = 0
    for row in packet_rows:
        relative = Path(row['filename'])
        actual = BASE / 'phenopacket_store' / row['cohort'] / (relative.name + '.json')
        if not actual.exists():
            actual = BASE / 'phenopacket_store' / (row['filename'] + '.json')
        if not actual.exists():
            packet_missing += 1
            missing_packet_files.append(row['filename'])
            continue
        packet_files += 1
        # Read actual diagnosis, rather than infer diagnosis from cohort/gene names.
        raw = actual.read_bytes()
        packet_digest.update(row['filename'].encode() + b'\0' + hashlib.sha256(raw).digest())
        packet = json.loads(raw)
        disease_ids = {d.get('term', {}).get('id', '') for d in packet.get('diseases', []) if not d.get('excluded', False)}
        mids = {resolve(d) for d in disease_ids if d}
        for mid in mids - {None}:
            cases[mid].add(row['filename'])
            if row['PMID']:
                publications[mid].add(row['PMID'])
    print('Phenopackets parsed', packet_files, flush=True)

    syndrome_map = {}
    for row in read_tsv(gmdb_root / 'gmdb_syndromes_v1.1.0.tsv'):
        omim = row['OMIM'].removesuffix('.0')
        syndrome_map[row['syndrome_id']] = resolve('OMIM:' + omim) if omim else None
    gmdb_patients, gmdb_images = defaultdict(set), defaultdict(set)
    gmdb_unparsed = Counter()
    gmdb_rows = list(read_tsv(gmdb_root / 'image_metadata_v1.1.0.tsv'))
    for row in gmdb_rows:
        sid = row['internal_syndrome_id']
        if sid not in syndrome_map:
            gmdb_unparsed[sid] += 1
            continue
        mid = syndrome_map[sid]
        if mid:
            gmdb_patients[mid].add(row['patient_id'])
            gmdb_images[mid].add(row['image_id'])

    # Count only exact chapter titles/synonyms and chapter-specific OMIM tables.
    # Incidental disease mentions and differential-diagnosis tables are excluded.
    chapters, chapter_evidence = defaultdict(set), []
    gene_files = sorted((BASE / 'GeneReviews/gene_NBK1116').glob('*.nxml'))
    gene_errors = []
    gene_digest = hashlib.sha256()
    for path in gene_files:
        try:
            raw = path.read_bytes()
            gene_digest.update(path.name.encode() + b'\0' + hashlib.sha256(raw).digest())
            root = ET.fromstring(raw)
        except ET.ParseError:
            gene_errors.append(path.name)
            continue
        title_node = root.find('.//book-part-meta/title-group/title')
        if title_node is None:
            continue
        title = ''.join(title_node.itertext()).strip()
        matches = defaultdict(set)
        title_ids = lexical.get(norm(title), set())
        if len(title_ids) == 1:
            matches[next(iter(title_ids))].add('exact_title_or_exact_synonym')
        for table in root.findall('.//table-wrap'):
            caption = table.find('caption')
            if caption is None or 'OMIM Entries' not in ''.join(caption.itertext()):
                continue
            for link in table.findall('.//tbody//ext-link'):
                if link.get('ext-link-type') != 'omim':
                    continue
                for omim in re.findall(r'\b\d{6}\b', link.get('{http://www.w3.org/1999/xlink}href', '')):
                    mid = resolve('OMIM:' + omim)
                    if mid:
                        matches[mid].add('chapter_OMIM_entries_exact_mapping')
        for mid, methods in matches.items():
            chapters[mid].add(path.name)
            if mid in target or ancestors(mid) & target:
                chapter_evidence.append({'mondo_id': mid, 'mondo_name': terms[mid]['name'], 'chapter_title': title, 'chapter_path': str(path), 'mapping_method': ';'.join(sorted(methods))})
    print('GeneReviews parsed', len(gene_files), flush=True)

    resources = {'hpo_terms': positive, 'orphadata_hpo_terms': orpha_positive, 'phenopacket_cases': cases, 'gmdb_patients': gmdb_patients, 'genereviews_chapters': chapters}
    descendant = {k: defaultdict(set) for k in resources}
    for key, data in resources.items():
        for mid, values in data.items():
            for ancestor in ancestors(mid) & target:
                descendant[key][ancestor].update(values)
    rows = []
    for item in inventory:
        mid = item['mondo_id']
        row = {'mondo_id': mid, 'diagnosis_name': item['diagnosis_name'], 'chop_candidate_patients': int(item['patients']), 'chop_candidate_images': int(item['images']), 'hpo_positive_terms_exact': len(positive[mid]), 'hpo_negative_terms_exact': len(negative[mid]), 'hpo_source_ids_exact': ';'.join(sorted(hpo_sources[mid])), 'phenopacket_case_records_exact': len(cases[mid]), 'phenopacket_publications_exact': len(publications[mid]), 'gmdb_patients_exact': len(gmdb_patients[mid]), 'gmdb_images_exact': len(gmdb_images[mid]), 'genereviews_chapters_exact': len(chapters[mid]), 'genereviews_files_exact': ';'.join(sorted(chapters[mid]))}
        for key in resources:
            row[key + '_strict_descendants'] = len(descendant[key][mid])
        row['orphadata_positive_hpo_terms_exact'] = len(orpha_positive[mid])
        row['orphadata_negative_hpo_terms_exact'] = len(orpha_negative[mid])
        row['pmc_case_report_coverage_status'] = 'not_audited'
        row['clinical_reference_status'] = 'linked_GeneReviews_chapter_available' if chapters[mid] else 'no_exact_GeneReviews_link_in_local_snapshot'
        rows.append(row)
    summary = {'scope': '165 craniofacial MONDO image-label concepts; local resource snapshots; candidate availability, not training admission', 'disease_count': len(rows), 'coverage': {}, 'intersection_counts': {}, 'phenopacket_summary_rows': len(packet_rows), 'phenopacket_files_read': packet_files, 'phenopacket_files_missing': packet_missing, 'genereviews_nxml_files': len(gene_files), 'genereviews_parse_errors': gene_errors, 'gmdb_metadata_rows': len(gmdb_rows), 'gmdb_unrecognized_syndrome_id_rows': dict(gmdb_unparsed), 'mapping_failure_occurrences': dict(mapping_status)}
    sets = {}
    for key, data in resources.items():
        covered = {mid for mid in target if data[mid]}
        sets[key] = covered
        summary['coverage'][key] = {'exact_diseases': len(covered), 'exact_or_descendant_diseases': len(covered | {mid for mid in target if descendant[key][mid]}), 'exact_unique_resource_items': len(set().union(*(data[mid] for mid in target)))}
    summary['intersection_counts'] = {'all_five_exact': len(set.intersection(*sets.values())), 'HPO_and_phenopackets_exact': len(sets['hpo_terms'] & sets['phenopacket_cases']), 'HPO_and_GeneReviews_exact': len(sets['hpo_terms'] & sets['genereviews_chapters'])}
    summary['orphadata_export_date'] = orpha_root.get('date')
    summary['phenopacket_missing_files'] = missing_packet_files
    inputs = [inventory_path, ontology_path, hpo_path, orpha_path, packet_path, gmdb_root / 'gmdb_syndromes_v1.1.0.tsv', gmdb_root / 'image_metadata_v1.1.0.tsv', Path(__file__)]
    versions = {str(p): {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs}
    versions['genereviews_directory'] = {'path': str(gene_files[0].parent), 'nxml_count': len(gene_files), 'files': {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in gene_files}}
    versions['phenopacket_file_contents_sha256_aggregate'] = packet_digest.hexdigest()
    versions['genereviews_file_contents_sha256_aggregate'] = gene_digest.hexdigest()
    write_tsv('disease_resource_matrix.tsv', rows)
    write_tsv('genereviews_mapping_evidence.tsv', chapter_evidence)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (OUT / 'input_versions.json').write_text(json.dumps(versions, indent=2) + '\n')
    assert len(rows) == len({r['mondo_id'] for r in rows}) == 165
    assert sum(r['chop_candidate_images'] for r in rows) >= 47974  # Multi-label images may repeat.
    print(json.dumps(summary, indent=2), flush=True)
    write_report(rows, summary)


def write_report(rows, summary):
    labels = {'hpo_terms': 'HPO disease annotations', 'orphadata_hpo_terms': 'Orphadata phenotype associations', 'phenopacket_cases': 'Phenopacket Store cases', 'gmdb_patients': 'GMDB v1.1.0 patients', 'genereviews_chapters': 'GeneReviews chapter links'}
    lines = ['# Public Resource Coverage for Craniofacial MONDO Concepts', '',
             'Audit date: October 11, 2026. Denominator: 165 distinct MONDO concepts in the conservative craniofacial image-label inventory.', '',
             'This is an audit of available local snapshots, not a claim of exhaustive or latest worldwide resource coverage. No source data, patient labels, reservations, or training eligibility were changed.', '',
             '## Coverage summary', '', '| Resource | Exact concept coverage | Exact or descendant coverage |', '|---|---:|---:|']
    for key, label in labels.items():
        value = summary['coverage'][key]
        lines.append(f"| {label} | {value['exact_diseases']} / 165 | {value['exact_or_descendant_diseases']} / 165 |")
    lines += ['', 'Exact resource totals: 470 distinct Phenopacket source-case records, 594 GMDB patient IDs, and 22 GeneReviews chapter files. These totals refer to resources linked to the exact target concepts. They are not additive across sources or proof of independent patient identities.', '',
              'HPO and Orphadata overlap and must not be added. Their term counts count distinct HPO identifiers, not patient observations. Four target concepts have exact coverage in all five resource categories.', '',
              '## Definitions and limits', '',
              '- Exact cross-resource mapping uses active MONDO IDs or explicit OMIM/Orphanet equivalentTo/exactMatch links in local MONDO. Ambiguous mappings are excluded. No fuzzy names or gene-to-disease inference is used.',
              '- Descendant coverage follows MONDO is_a edges. It indicates available resources for more specific concepts beneath a target, and does not turn those diagnoses into equivalent labels. A subtype can relate to multiple targets; counts across rows are not additive.',
              '- Exact plus descendant coverage is the union at the disease level. Columns ending in strict_descendants count resources under strict descendants separately; they are not an increment that can always be added to the exact count.',
              '- CHOP counts are candidate image-label associations, including structural diagnoses, not clinically confirmed cases. One patient/image can carry multiple concepts.',
              '- Phenopacket counts use distinct source-case records with available JSON files and disease terms. They do not imply that every record has images, complete phenotypes, or a unique person across publications. No synthetic cases were added.',
              '- GMDB counts come from original image metadata linked to its syndrome table; they are not new file-read/QC or affected-subject checks. GMDB is controlled-access research data. Local possession does not establish approval for every proposed reuse.',
              '- GeneReviews is conservatively linked through exact chapter titles/MONDO exact synonyms or chapter-specific OMIM Entries tables. Incidental mentions and differential-diagnosis tables do not count. Some overview chapters lack these mappings: for example, the FGFR craniosynostosis overview discusses Crouzon and Pfeiffer syndromes, but they have no exact chapter link under this rule. Zero therefore means no strict automatic match, not absence of clinical reference material.',
              '- GeneReviews linkage establishes chapter availability, not verification of every diagnostic recommendation, evidence quality, or permission for bulk model training. PMC case-report coverage is explicitly not audited.',
              '- Public cases and their originating publications must be reconciled across training, retrieval and evaluation before claiming independent testing. Public knowledge associations cannot be substituted for observed patient phenotypes.', '',
              '## Source snapshots', '',
              '- MONDO: local release 2026-09-01.',
              '- HPO phenotype.hpoa: annotation and HPO version 2025-01-16; positive and NOT annotations counted separately.',
              '- Orphadata: XML export 2025-12-09; explicit excluded associations counted separately. The directory name v4.8_20260422 is not treated as the XML export date.',
              '- Phenopacket Store: 8,207 summary records and all 8,207 corresponding JSON files read. Snapshot release identifier is not established; input hashes identify the local snapshot.',
              '- GMDB: v1.1.0, 10,980 metadata rows.',
              '- GeneReviews: 987 NXML files parsed without XML errors. Chapter dates vary; no single release date is assumed.', '',
              '## Complete disease matrix', '',
              'HPO and Orpha columns are distinct positive HPO terms under exact mappings. Cases are Phenopacket source-case records; GMDB is distinct patient IDs; GR is chapter files. All resource columns in this table use exact matching.', '',
              '| MONDO ID | Diagnosis | CHOP patients | CHOP images | HPO | Orpha | Cases | GMDB patients | GR |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(rows, key=lambda r: (-r['chop_candidate_patients'], r['mondo_id'])):
        lines.append('| ' + ' | '.join(str(r[k]).replace('|', '/') for k in ['mondo_id', 'diagnosis_name', 'chop_candidate_patients', 'chop_candidate_images', 'hpo_positive_terms_exact', 'orphadata_positive_hpo_terms_exact', 'phenopacket_case_records_exact', 'gmdb_patients_exact', 'genereviews_chapters_exact']) + ' |')
    lines += ['', '## Files and reproduction', '',
              '- [Full machine-readable matrix](disease_resource_matrix.tsv): includes strict-descendant counts, source identifiers and explicit not-audited status.',
              '- [Summary](summary.json).',
              '- [GeneReviews linkage evidence](genereviews_mapping_evidence.tsv): chapter names, source paths and match methods.',
              '- [Input versions](input_versions.json): input hashes and aggregate file-content digests.',
              '- Run `python audit_public_resource_coverage.py` from the repository root to reproduce.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
