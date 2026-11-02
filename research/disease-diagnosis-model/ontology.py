#!/usr/bin/env python
"""Minimal OBO parser + cross-reference indices for MONDO / HPO.

MONDO is the right normalisation target because a single term carries xrefs to
OMIM, Orphanet, SNOMED, ICD-10, UMLS and DOID -- so one file normalises every
label source in this project at once (folder strings, EHR ICD-10, SNOMED FSNs,
and the expert MONDO file).

HPO is loaded alongside because roughly half of this cohort's taxonomy is not a
disease: micrognathia, hypertelorism, plagiocephaly and macroglossia are
phenotypic abnormalities (HPO), not Mendelian disease entities (MONDO). Forcing
those into MONDO produces wrong or absent mappings.
"""
import re, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ONTO = os.path.join(HERE, 'ontology')

# Prefixes as they actually appear in the MONDO release (verified by counting
# the xref lines): there is no plain 'ICD10' -- it is ICD10CM / ICD10WHO.
XREF_PREFIXES = ('OMIM', 'OMIMPS', 'Orphanet', 'SCTID', 'ICD10CM', 'ICD10WHO',
                 'UMLS', 'DOID', 'MESH', 'NCIT', 'MEDGEN', 'GARD')


def normstr(s):
    s = s.lower().strip()
    s = re.sub(r"[‘’']", '', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


class Term:
    __slots__ = ('id', 'name', 'defn', 'synonyms', 'xrefs', 'parents',
                 'obsolete', 'replaced_by', 'subsets')

    def __init__(self, tid):
        self.id = tid; self.name = None; self.defn = None
        self.synonyms = []          # (text, scope)
        self.xrefs = []             # (prefix, value)
        self.parents = []
        self.obsolete = False; self.replaced_by = None; self.subsets = []

    def __repr__(self):
        return f"<{self.id} {self.name}>"


def parse_obo(path):
    terms, cur, in_term = {}, None, False
    syn_re = re.compile(r'^synonym:\s*"((?:[^"\\]|\\.)*)"\s*(\w+)?')
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if line == '[Term]':
                in_term, cur = True, None
                continue
            if line.startswith('[') and line.endswith(']'):
                in_term = False          # [Typedef] etc.
                continue
            if not in_term or not line:
                continue
            if line.startswith('id: '):
                cur = Term(line[4:].strip()); terms[cur.id] = cur
            elif cur is None:
                continue
            elif line.startswith('name: '):
                cur.name = line[6:].strip()
            elif line.startswith('def: '):
                m = re.match(r'def:\s*"((?:[^"\\]|\\.)*)"', line)
                if m: cur.defn = m.group(1)
            elif line.startswith('synonym: '):
                m = syn_re.match(line)
                if m: cur.synonyms.append((m.group(1), m.group(2) or 'RELATED'))
            elif line.startswith('xref: '):
                v = line[6:].split('{')[0].strip()
                if ':' in v:
                    p, _, rest = v.partition(':')
                    if p in XREF_PREFIXES:
                        cur.xrefs.append((p, rest.strip()))
            elif line.startswith('is_a: '):
                cur.parents.append(line[6:].split('!')[0].strip())
            elif line.startswith('is_obsolete: true'):
                cur.obsolete = True
            elif line.startswith('replaced_by: '):
                cur.replaced_by = line[13:].strip()
            elif line.startswith('subset: '):
                cur.subsets.append(line[8:].strip())
    return terms


class Ontology:
    def __init__(self, mondo_path=None, hp_path=None):
        self.mondo = parse_obo(mondo_path or os.path.join(ONTO, 'mondo.obo'))
        self.hp = parse_obo(hp_path or os.path.join(ONTO, 'hp.obo'))
        self._index()

    def _index(self):
        self.by_name = {}            # normalised label -> [ids]  (MONDO)
        self.hp_by_name = {}
        self.xref = collections.defaultdict(list)   # (prefix, value) -> [MONDO ids]
        for t in self.mondo.values():
            if t.obsolete or 'not_a_disease' in t.subsets:
                continue
            if not t.id.startswith('MONDO:'):
                continue          # mondo.obo also ships imported HP/UBERON/GO terms
            if t.name:
                self.by_name.setdefault(normstr(t.name), []).append(t.id)
            for s, scope in t.synonyms:
                if scope in ('EXACT', 'NARROW'):
                    self.by_name.setdefault(normstr(s), []).append(t.id)
            for p, v in t.xrefs:
                self.xref[(p, v)].append(t.id)
        for t in self.hp.values():
            if t.obsolete:
                continue
            if t.name:
                self.hp_by_name.setdefault(normstr(t.name), []).append(t.id)
            for s, scope in t.synonyms:
                if scope in ('EXACT', 'NARROW'):
                    self.hp_by_name.setdefault(normstr(s), []).append(t.id)

    # -- lookups ---------------------------------------------------------
    def mondo_by_label(self, s):
        return dedup(self.by_name.get(normstr(s), []))

    def hpo_by_label(self, s):
        return dedup(self.hp_by_name.get(normstr(s), []))

    def mondo_by_snomed(self, code):
        code = str(code).replace('SNOMED#', '').strip()
        return dedup(self.xref[('SCTID', code)])

    def mondo_by_icd10(self, code):
        code = str(code).strip().upper()
        hits = self.xref[('ICD10CM', code)] + self.xref[('ICD10WHO', code)]
        if not hits and '.' in code:          # Q75.0 -> Q75
            base = code.split('.')[0]
            hits = self.xref[('ICD10CM', base)] + self.xref[('ICD10WHO', base)]
        return dedup(hits)

    def omim_for(self, mondo_id):
        t = self.mondo.get(mondo_id)
        return dedup([v for p, v in t.xrefs if p == 'OMIM']) if t else []

    def xrefs_for(self, mondo_id, prefix):
        t = self.mondo.get(mondo_id)
        return dedup([v for p, v in t.xrefs if p == prefix]) if t else []

    def name_of(self, oid):
        t = self.mondo.get(oid) or self.hp.get(oid)
        return t.name if t else None

    def ancestors(self, oid, maxdepth=50):
        """transitive is_a closure, for rolling rare terms up to a trainable level"""
        src = self.mondo if oid.startswith('MONDO') else self.hp
        out, frontier = set(), [oid]
        for _ in range(maxdepth):
            nxt = []
            for x in frontier:
                t = src.get(x)
                if not t: continue
                for p in t.parents:
                    if p not in out:
                        out.add(p); nxt.append(p)
            if not nxt: break
            frontier = nxt
        return out


if __name__ == '__main__':
    o = Ontology()
    print(f"MONDO terms {len(o.mondo):,}   HPO terms {len(o.hp):,}")
    print(f"MONDO name/synonym keys {len(o.by_name):,}   HPO keys {len(o.hp_by_name):,}")
    print(f"xref keys {len(o.xref):,}")
    for q in ['Crouzon syndrome', 'Apert syndrome', 'Treacher Collins syndrome',
              '22q11.2 deletion syndrome', 'Pfeiffer syndrome', 'cleft lip',
              'Beckwith-Wiedemann syndrome', 'Goldenhar syndrome']:
        ids = o.mondo_by_label(q)
        omim = [x for i in ids for x in o.omim_for(i)]
        print(f"  {q:<30} -> {ids[:2]}  OMIM={omim[:3]}")
    print("\nphenotypes that are NOT MONDO diseases:")
    for q in ['micrognathia', 'hypertelorism', 'plagiocephaly', 'macroglossia', 'microcephaly']:
        print(f"  {q:<16} MONDO={o.mondo_by_label(q)[:1]}  HPO={o.hpo_by_label(q)[:1]}")
    print("\ncrosswalks:")
    print("  SNOMED 1003586008 ->", o.mondo_by_snomed('1003586008'),
          [o.name_of(i) for i in o.mondo_by_snomed('1003586008')])
    print("  ICD10  Q75.0      ->", o.mondo_by_icd10('Q75.0'),
          [o.name_of(i) for i in o.mondo_by_icd10('Q75.0')])
