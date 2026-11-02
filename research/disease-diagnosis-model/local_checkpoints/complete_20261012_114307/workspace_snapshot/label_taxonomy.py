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
GENERIC = [r'^lesion', r'^cyst', r'^tumor', r'^mass', r'^trauma$', r'^scar$', r'^burns?$',
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
 ('crouzon',                     [r'^crouzon']),
 ('apert',                       [r'^apert']),
 ('pfeiffer',                    [r'^pfeiffer',r'phieffer']),
 ('saethre_chotzen',             [r'saethre']),
 ('muenke',                      [r'muenke',r'^fgfr3$']),
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

rows=[]
with open(P,newline='',encoding='utf-8',errors='replace') as f:
    for row in csv.DictReader(f,delimiter='\t'):
        rows.append(((row.get('disease_or_phenotypes') or '').strip(),
                     (row.get('norm_mrn') or '').strip()))
cat=collections.Counter(); dximg=collections.Counter()
dxpat=collections.defaultdict(set); other=collections.Counter(); otherpat=collections.defaultdict(set)
for d,m in rows:
    k,v=classify(d); cat[k]+=1
    if k=='DX':
        dximg[v]+=1
        if m: dxpat[v].add(m)
    elif k=='OTHER':
        other[v]+=1
        if m: otherpat[v].add(m)
print("BUCKETS:", cat.most_common())
tot=sum(cat.values())
print("\n=== DIAGNOSIS GROUPS (%d groups) ==="%len(dximg))
print("%-32s %7s %7s"%("group","images","patients"))
for k,v in dximg.most_common():
    print("%-32s %7d %7d"%(k,v,len(dxpat[k])))
print("\nDX total images %d, unique patients %d"%(sum(dximg.values()),len(set().union(*dxpat.values()))))
for t in [20,50,100,200]:
    sel=[k for k in dximg if len(dxpat[k])>=t]
    print("  groups w/ >=%3d patients: %2d  -> %6d imgs, %5d patients"%(
        t,len(sel),sum(dximg[k] for k in sel),len(set().union(*[dxpat[k] for k in sel]))))
print("\n=== TOP UNMAPPED 'OTHER' (%d imgs, %d distinct strings) ==="%(sum(other.values()),len(other)))
for k,v in other.most_common(30): print("  %5d %4d  %s"%(v,len(otherpat[k]),k))

# ---- multi-label / patient structure after grouping
pg=collections.defaultdict(set)
for d,m in rows:
    k,v=classify(d)
    if k=='DX' and m: pg[m].add(v)
c=collections.Counter(len(s) for s in pg.values())
print("\nDX groups per patient:",sorted(c.items()))
print("patients w/ >=2 DX groups: %d (%.1f%%)"%(sum(v for k,v in c.items() if k>=2),
      100*sum(v for k,v in c.items() if k>=2)/len(pg)))
# images per patient within DX set
ip=collections.Counter()
for d,m in rows:
    k,v=classify(d)
    if k=='DX' and m: ip[m]+=1
vv=sorted(ip.values())
print("DX imgs/patient: mean %.1f median %d p90 %d max %d"%(sum(vv)/len(vv),vv[len(vv)//2],vv[int(.9*len(vv))],vv[-1]))
# 70/15/15 patient-level split feasibility for >=20-patient groups
sel=[k for k in dximg if len(dxpat[k])>=20]
print("\n70/15/15 patient split on %d groups: ~%d train / %d val / %d test patients"%(
    len(sel), int(.7*12553), int(.15*12553), int(.15*12553)))
small=[(k,len(dxpat[k])) for k in sel if len(dxpat[k])<60]
print("groups where 15%% test <10 patients:",[(k,n) for k,n in small if .15*n<10])
