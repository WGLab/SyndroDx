import csv, collections, re, json
csv.field_size_limit(10**9)
P="/controlled-data/craniofacial_data/craniofacial_metadata_frontal_updated_123025_deduplicated.csv"

def norm(s):
    s=s.lower().strip()
    s=re.sub(r'^\d+(\.\d+)*\s+','',s)          # strip billing prefixes "26.3 ", "14.3.3 "
    s=re.sub(r'[^a-z0-9]+',' ',s).strip()
    s=re.sub(r'\s+',' ',s)
    return s

# ---- non-diagnostic: procedures, workflow/billing artifacts, generic descriptors
PROC = [r'\bnot done\b', r'\brhino\b', r'rhinoplast', r'otoplast', r'\bbleph\b', r'facelift',
 r'browlift', r'genio', r'forehead flap', r'\bflaps?\b', r'distraction osteogenesis',
 r'\bcpt\b', r'\bcombo facial\b', r'scar rev', r'nerve graft', r'cranioplasty',
 r'cranial vault reshap', r'fat grafting', r'malar aug', r'reconstruction', r'\brecon\b',
 r'skin graft', r'cart graft', r'free muscle transfer', r'ear molding', r'bimaxillary sx',
 r'\bmisc\b.*proc', r'\bteam\b', r'^oj$', r'^oj ', r'gender dysphoria', r'gynecomastia',
 r'macromastia', r'\bdwl\b', r'\bsx\b']
GENERIC = [r'^fgfr\d*$', r'^fgfr', r'^lesion', r'^cyst', r'^tumor', r'^mass', r'^trauma$', r'^scar$', r'^burns?$',
 r'^laceration', r'^hematoma', r'^skin tags?$', r'^birthmark', r'^unclassified', r'^undetermined',
 r'^acquired def', r'^acq ', r'^head shape$', r'^abnormal head shape$', r'complex defects',
 r'craniofac misc', r'^neoplasms?$', r'^cfd$']

# ---- canonical diagnosis groups (regex -> canonical name)
GROUPS = [
 ('cleft_lip_palate_unilateral', [r'^uclp$',r'^uclcp$',r'^lclp$',r'^lclcp$',r'^unilateral cleft lip']),
 ('cleft_lip_palate_bilateral',  [r'^bclp$',r'^bcl$',r'^bilat cleft lip']),
 ('cleft_lip_only',              [r'^cl$',r'^cleft lip$',r'^cl alv',r'^cl and alveolar']),
 ('cleft_palate_only',           [r'^cp$',r'^cleft palate$',r'^submucous cleft palate',r'^high arched palate',r'^bifid uvula',r'^cpp$']),
 ('cleft_lip_and_palate_unspec', [r'^clp$',r'^cleft lip palate$',r'^cleft$',r'^alveolar cleft']),
 ('pierre_robin',                [r'pierre robin',r'^cp ?[-. ]?prs?$',r'^cp pr$',r'^prs$']),
 ('cleft_nasal_deformity',       [r'cleft nasal deformity',r'^nasal cleft']),
 ('rare_facial_cleft',           [r'rare cleft',r'tessier',r'^facial cleft',r'cleft def of jaw',r'^ear cleft',r'macrostomia']),
 ('sagittal_synostosis',         [r'^sag',r'sagittal syn',r'scaphocephaly']),
 ('metopic_synostosis',          [r'^met',r'metopic',r'trigon']),
 ('coronal_synostosis',          [r'coronal syn',r'unicoronal',r'bicoronal']),
 ('lambdoid_synostosis',         [r'lambdoid']),
 ('multisuture_synostosis',      [r'multi.?suture',r'pansynostosis',r'pancranio',r'mercedes',r'squamosal syn']),
 ('craniosynostosis_unspec',     [r'^craniosynostosis$',r'^craniosyn']),
 ('positional_plagiocephaly',    [r'plagiocephaly',r'brachycephaly',r'metopic ridge']),
 ('apert_or_pfeiffer_unspec',    [r'apert\w*\s+(pfeiffer|phieffer|clove)']),
 ('crouzon',                     [r'^crouzon']),
 ('apert',                       [r'^apert']),
 ('pfeiffer',                    [r'^pfeiffer',r'phieffer']),
 ('saethre_chotzen',             [r'saethre']),
 ('muenke',                      [r'muenke']),
 ('treacher_collins',            [r'treacher']),
 ('nager',                       [r'^nager']),
 ('goldenhar_ocav',              [r'goldenhar']),
 ('hemifacial_microsomia',       [r'^hfm',r'hemifacial microsomia',r'bifac microsomia',r'oro mand limb',r'oro costo',r'cerebro costo']),
 ('microtia_atresia',            [r'microtia',r'cryptotia',r'^ear deformity',r'protruding ears',r'^pro ears']),
 ('del22q11',                    [r'^22q']),
 ('beckwith_wiedemann',          [r'beckwith',r'hemihypertroph',r'hemihyperplasia']),
 ('parry_romberg',               [r'romberg',r'parry',r'perry',r'scleroderma']),
 ('moebius',                     [r'moebius',r'mobius']),
 ('facial_palsy',                [r'facial palsy',r'facial paralysis',r'bell',r'facial motion disorder',r'facial par ']),
 ('neurofibromatosis',           [r'^nf$',r'neurofibrom',r'sphenoid wing dysplasia']),
 ('fibrous_dysplasia',           [r'fibrous dysplasia']),
 ('craniofrontonasal_dysplasia', [r'^cfnd$',r'cfn dysplasia',r'frontonasal dysplasia',r'midline anomal',r'holoprosencephaly']),
 ('hypertelorism_orbital',       [r'hypertelorism',r'orbital dystopia',r'orbital dysplasia']),
 ('mandibular_hypoplasia',       [r'micrognathia',r'retrognathia',r'mand hypoplasia',r'mandibular hypoplasia',r'mand def',r'mand deformity']),
 ('mandibular_hyperplasia_asym', [r'mandibular hyperplasia',r'jaw asymmetry',r'facial asym',r'fac asymmetry',r'facial asymetry',r'jaw deformity']),
 ('maxillary_hypoplasia',        [r'maxillary hypoplasia',r'maxillary retrusion',r'midface retrusion',r'midface hypoplasia',r'binder']),
 ('malocclusion',                [r'malocclusion']),
 ('hemangioma',                  [r'hemangioma',r'pyogenic granuloma']),
 ('vascular_malformation',       [r'vascular malformation',r'vasc malformation',r'lymphatic malformation',r'port wine']),
 ('congenital_nevus',            [r'^nevus',r'pigmented nevi',r'nevus']),
 ('dermoid',                     [r'dermoid']),
 ('malignancy',                  [r'melanoma',r'basal cell carcinoma',r'rhabdomyo',r'sarcoma',r'teratoma',r'histiocytosis']),
 ('benign_skin_soft_tissue',     [r'pilomatrixoma',r'keloid',r'fibroma',r'branchial',r'cephalohematoma',r'granuloma']),
 ('nasal_fracture',              [r'nasal fx',r'nasal injury',r'nasal trauma']),
 ('facial_fracture_other',       [r'mandibular fx',r'jaw fx',r'orbital fx',r'orb fx',r'zygomatic fx',r'facial fx',r'pan facial fx',r'facial bone fracture',r'facial jaw fracture',r'orbital fracture',r'pediatric facial fx',r'mandibular fxs']),
 ('soft_tissue_trauma',          [r'dog bite',r'ferret bite',r'facial trauma',r'facial injury',r'post traumatic',r'eye trauma',r'soft tissue trauma']),
 ('deviated_septum_obstruction', [r'deviated septum',r'nasal obstruction',r'hypernasal']),
 ('nasal_deformity_other',       [r'nasal deformity',r'nas def',r'nasal def',r'nasal mass']),
 ('encephalocele_skull_defect',  [r'encephalocele',r'cutis aplasia',r'skull defect',r'cranial defect',r'skull deformity',r'slit ventricle']),
 ('microcephaly_macrocephaly',   [r'microcephaly',r'macrocephaly']),
 ('ectodermal_dysplasia',        [r'ectodermal dysplasia',r'cleidocranial',r'reiger',r'rieger']),
 ('charge',                      [r'charge']),
 ('poland',                      [r'^poland']),
 ('macroglossia',                [r'macroglossia']),
 ('torticollis',                 [r'torticollis']),
 ('syndactyly_polydactyly',      [r'syndact',r'polydact']),
 ('antley_bixler',               [r'antley']),
 ('conjoined_twins',             [r'conjoined']),
]
GC=[(n,[re.compile(p) for p in ps]) for n,ps in GROUPS]
PROC_R=[re.compile(p) for p in PROC]; GEN_R=[re.compile(p) for p in GENERIC]

def classify(raw):
    s=norm(raw)
    if not s: return ('UNLABELED',None)
    for r in PROC_R:
        if r.search(s): return ('PROCEDURE',s)
    for r in GEN_R:
        if r.search(s): return ('GENERIC',s)
    for name,rs in GC:
        for r in rs:
            if r.search(s): return ('DX',name)
    return ('OTHER',s)


# ---------------------------------------------------------------- hierarchy /
# A parent label co-occurring with one of its children is a granularity
# difference, not a disagreement -> collapse to the child.
HIER = {
 'cleft_lip_and_palate_unspec': {'cleft_lip_palate_unilateral','cleft_lip_palate_bilateral',
    'cleft_lip_only','cleft_palate_only','cleft_nasal_deformity','pierre_robin','rare_facial_cleft'},
 'craniosynostosis_unspec': {'sagittal_synostosis','metopic_synostosis','coronal_synostosis',
    'lambdoid_synostosis','multisuture_synostosis'},
 'multisuture_synostosis': {'sagittal_synostosis','metopic_synostosis','coronal_synostosis',
    'lambdoid_synostosis'},
 # A patient with cleft lip AND palate also has a cleft lip, and a cleft palate.
 # "cl" and "cp" are the general terms the archive files them under alongside
 # the specific one -- a hierarchy, not a disagreement. ("cp" is additionally
 # overloaded as the Cleft Palate Team clinic; the team folders themselves --
 # cpt, oj cpt, dwl cpt, oj cp team -- are already routed to PROCEDURE.)
 'cleft_palate_only': {'cleft_lip_palate_unilateral','cleft_lip_palate_bilateral','pierre_robin'},
 'cleft_lip_only': {'cleft_lip_palate_unilateral','cleft_lip_palate_bilateral'},
 'mandibular_hypoplasia': {'pierre_robin'},
 # one combined folder covering both syndromes
 'apert_or_pfeiffer_unspec': {'apert','pfeiffer'},
}

# Pairs that cannot both be true of the same patient. Co-occurrence across two
# folders is a hard label error, not a comorbidity.
# Only pairs that genuinely cannot both be true of one patient. Removed, with
# the reason each was wrong:
#   * suture pairs -- multiple sutures DO fuse together; that is multisuture
#     synostosis, a real entity, not a contradiction (45 patients affected)
#   * plagiocephaly vs synostosis -- these co-occur (20 patients)
#   * cl/cp vs uclp/bclp -- general vs specific, now handled by HIER (258)
TRUE_CONFLICTS = [
 # a cleft is unilateral or bilateral; it cannot be both
 {'cleft_lip_palate_unilateral','cleft_lip_palate_bilateral'},
 # distinct single-gene craniosynostosis syndromes
 {'crouzon','apert'}, {'crouzon','pfeiffer'}, {'apert','pfeiffer'},
 {'apert','saethre_chotzen'}, {'crouzon','muenke'},
]

def resolve(groups):
    """Collapse parent labels when a more specific child is also present."""
    out = set(groups)
    for parent, kids in HIER.items():
        if parent in out and (out & kids):
            out.discard(parent)
    return out

def has_conflict(groups):
    return any(len(groups & pair) > 1 for pair in TRUE_CONFLICTS)

def assign_tier(folders, groups):
    """folders: set of distinct parent_path seen for this patient.
       groups:  set of DX groups seen for this patient."""
    resolved = resolve(groups)
    if has_conflict(resolved):
        return 'C_conflict'
    if len(folders) >= 2:
        return 'A_corroborated' if len(resolved) == 1 else 'B_multi_compatible'
    return 'D_single_folder'
