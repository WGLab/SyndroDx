#!/usr/bin/env python
"""Normalise every label source in this project onto MONDO (+OMIM) / HPO.

Three sources are normalised:
  1. our 59 folder-derived taxonomy groups
  2. the cohort's EHR ICD-10 codes
  3. the cohort's SNOMED CT concepts

Routing rule: MONDO is a disease ontology. Roughly half of a craniofacial
surgery cohort is not a disease -- micrognathia, hypertelorism, plagiocephaly,
facial asymmetry are phenotypic abnormalities (HPO), and nasal fractures and dog
bites are injuries that belong in neither. Each group therefore declares what
kind of thing it is, and anything that fails to resolve is reported rather than
silently forced into a wrong term.

Candidate names are proposals; the ontology decides. Unresolved rows are the
manual-curation worklist.
"""
import csv, os, sys, collections, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ontology import Ontology, normstr

csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__))

# kind: 'disease' -> expect MONDO; 'phenotype' -> expect HPO;
#       'injury'/'procedure'/'heterogeneous' -> deliberately unmapped
GROUPS = {
 'cleft_lip_palate_unilateral': ('disease', ['unilateral cleft lip and palate', 'cleft lip and palate', 'orofacial cleft']),
 'cleft_lip_palate_bilateral':  ('disease', ['bilateral cleft lip and palate', 'cleft lip and palate']),
 'cleft_lip_only':              ('disease', ['cleft lip', 'nonsyndromic cleft lip']),
 'cleft_palate_only':           ('disease', ['cleft palate', 'isolated cleft palate']),
 'cleft_lip_and_palate_unspec': ('disease', ['cleft lip and palate', 'orofacial cleft']),
 'pierre_robin':                ('disease', ['Pierre Robin syndrome', 'Pierre Robin sequence']),
 'cleft_nasal_deformity':       ('phenotype', ['abnormality of the nose', 'cleft of the nose']),
 'rare_facial_cleft':           ('disease', ['facial cleft', 'oblique facial cleft', 'Tessier cleft']),
 'sagittal_synostosis':         ('disease', ['sagittal craniosynostosis', 'scaphocephaly', 'craniosynostosis']),
 'metopic_synostosis':          ('disease', ['metopic craniosynostosis', 'trigonocephaly', 'craniosynostosis']),
 'coronal_synostosis':          ('disease', ['coronal craniosynostosis', 'craniosynostosis']),
 'lambdoid_synostosis':         ('disease', ['lambdoid synostosis', 'craniosynostosis']),
 'multisuture_synostosis':      ('disease', ['craniosynostosis']),
 'craniosynostosis_unspec':     ('disease', ['craniosynostosis']),
 'positional_plagiocephaly':    ('phenotype', ['plagiocephaly']),
 'crouzon':                     ('disease', ['Crouzon syndrome']),
 'apert':                       ('disease', ['Apert syndrome']),
 'pfeiffer':                    ('disease', ['Pfeiffer syndrome']),
 'saethre_chotzen':             ('disease', ['Saethre-Chotzen syndrome']),
 'muenke':                      ('disease', ['Muenke syndrome']),
 'treacher_collins':            ('disease', ['Treacher Collins syndrome', 'mandibulofacial dysostosis']),
 'nager':                       ('disease', ['Nager syndrome', 'acrofacial dysostosis Nager type']),
 'goldenhar_ocav':              ('disease', ['Goldenhar syndrome', 'oculoauriculovertebral spectrum']),
 'hemifacial_microsomia':       ('disease', ['hemifacial microsomia', 'craniofacial microsomia']),
 'microtia_atresia':            ('disease', ['microtia', 'congenital aural atresia']),
 'del22q11':                    ('disease', ['22q11.2 deletion syndrome', 'DiGeorge syndrome']),
 'beckwith_wiedemann':          ('disease', ['Beckwith-Wiedemann syndrome']),
 'parry_romberg':               ('disease', ['Parry-Romberg syndrome', 'progressive hemifacial atrophy']),
 'moebius':                     ('disease', ['Moebius syndrome']),
 'facial_palsy':                ('phenotype', ['facial palsy', 'facial paralysis']),
 'neurofibromatosis':           ('disease', ['neurofibromatosis type 1', 'neurofibromatosis']),
 'fibrous_dysplasia':           ('disease', ['fibrous dysplasia of bone', 'fibrous dysplasia']),
 'craniofrontonasal_dysplasia': ('disease', ['craniofrontonasal dysplasia', 'frontonasal dysplasia']),
 'hypertelorism_orbital':       ('phenotype', ['hypertelorism', 'orbital hypertelorism']),
 'mandibular_hypoplasia':       ('phenotype', ['micrognathia', 'mandibular hypoplasia']),
 'mandibular_hyperplasia_asym': ('phenotype', ['facial asymmetry', 'mandibular prognathia']),
 'maxillary_hypoplasia':        ('phenotype', ['maxillary hypoplasia', 'midface retrusion']),
 'malocclusion':                ('phenotype', ['malocclusion', 'dental malocclusion']),
 'hemangioma':                  ('disease', ['infantile hemangioma', 'hemangioma']),
 'vascular_malformation':       ('disease', ['vascular malformation']),
 'congenital_nevus':            ('disease', ['congenital melanocytic nevus', 'melanocytic nevus']),
 'dermoid':                     ('disease', ['dermoid cyst']),
 'encephalocele_skull_defect':  ('disease', ['encephalocele']),
 'ectodermal_dysplasia':        ('disease', ['ectodermal dysplasia']),
 'charge':                      ('disease', ['CHARGE syndrome']),
 'poland':                      ('disease', ['Poland syndrome']),
 'antley_bixler':               ('disease', ['Antley-Bixler syndrome']),
 'conjoined_twins':             ('disease', ['conjoined twins']),
 'macroglossia':                ('phenotype', ['macroglossia']),
 'torticollis':                 ('phenotype', ['torticollis', 'congenital torticollis']),
 'deviated_septum_obstruction': ('phenotype', ['deviated nasal septum', 'nasal obstruction']),
 'microcephaly_macrocephaly':   ('phenotype', ['microcephaly']),      # AMBIGUOUS: split it
 'syndactyly_polydactyly':      ('phenotype', ['syndactyly']),        # AMBIGUOUS: split it
 'malignancy':                  ('heterogeneous', ['neoplasm']),
 'benign_skin_soft_tissue':     ('heterogeneous', []),
 'nasal_deformity_other':       ('heterogeneous', []),
 'nasal_fracture':              ('injury', []),
 'facial_fracture_other':       ('injury', []),
 'soft_tissue_trauma':          ('injury', []),
}


def main():
    o = Ontology()
    # children index so grouping terms can surface OMIM ids held by subtypes
    kids = collections.defaultdict(list)
    for t in o.mondo.values():
        for p in t.parents:
            kids[p].append(t.id)

    def omim_deep(mid, depth=2):
        out, frontier = list(o.omim_for(mid)), [mid]
        for _ in range(depth):
            nxt = []
            for x in frontier:
                for c in kids.get(x, []):
                    out += o.omim_for(c); nxt.append(c)
            frontier = nxt
            if out or not frontier:
                break
        seen, ded = set(), []
        for x in out:
            if x not in seen: seen.add(x); ded.append(x)
        return ded

    rows, unresolved = [], []
    for g, (kind, cands) in GROUPS.items():
        mid = hid = None; matched = None
        for c in cands:
            if kind == 'phenotype':
                h = o.hpo_by_label(c)
                if h: hid, matched = h[0], c; break
            m = o.mondo_by_label(c)
            if m: mid, matched = m[0], c; break
            h = o.hpo_by_label(c)
            if h and not hid: hid, matched = h[0], c
        omim = omim_deep(mid) if mid else []
        rows.append({'group': g, 'kind': kind, 'matched_on': matched or '',
                     'mondo': mid or '', 'mondo_name': o.name_of(mid) if mid else '',
                     'hpo': hid or '', 'hpo_name': o.name_of(hid) if hid else '',
                     'omim': ';'.join(omim[:5]),
                     'orphanet': ';'.join(o.xrefs_for(mid, 'Orphanet')[:3]) if mid else '',
                     'umls': ';'.join(o.xrefs_for(mid, 'UMLS')[:2]) if mid else ''})
        if kind in ('disease', 'phenotype') and not (mid or hid):
            unresolved.append(g)

    out = os.path.join(HERE, 'label_ontology_map.tsv')
    cols = ['group','kind','matched_on','mondo','mondo_name','hpo','hpo_name','omim','orphanet','umls']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter='\t')
        w.writeheader(); w.writerows(rows)

    nd = sum(1 for r in rows if r['mondo'])
    np_ = sum(1 for r in rows if r['hpo'] and not r['mondo'])
    nom = sum(1 for r in rows if r['omim'])
    print(f"=== taxonomy groups: {len(rows)} ===")
    print(f"  resolved to MONDO : {nd}")
    print(f"  resolved to HPO   : {np_}")
    print(f"  with OMIM id(s)   : {nom}")
    print(f"  deliberately unmapped (injury/procedure/heterogeneous): "
          f"{sum(1 for r in rows if r['kind'] in ('injury','procedure','heterogeneous'))}")
    print(f"  UNRESOLVED (needs manual curation): {len(unresolved)} -> {unresolved}")
    print(f"\n{'group':<30} {'MONDO':<15} {'OMIM':<18} name")
    for r in sorted(rows, key=lambda x: (x['kind'], x['group'])):
        if r['mondo'] or r['hpo']:
            print(f"  {r['group']:<28} {r['mondo'] or r['hpo']:<15} "
                  f"{r['omim'][:17]:<18} {(r['mondo_name'] or r['hpo_name'] or '')[:42]}")
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
