"""Render original scientific diagrams from approved aggregate counts only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Ellipse, PathPatch
from matplotlib.path import Path as MPath

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent / 'experiment_results_20261019' / 'aggregate_results.json'
aggregate = json.loads(SOURCE.read_text())
train = aggregate['data']['released_partitions']['train']
assert sum(train['views'].values()) == train['images']

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'axes.unicode_minus': False, 'savefig.facecolor': 'white',
})
INK='#202B36'; MUTED='#5E6A76'; LINE='#D1D9DF'
BLUE='#2C6DA8'; TEAL='#087F8C'; VIOLET='#7554A2'
BLUE_LIGHT='#EEF4FA'; TEAL_LIGHT='#EDF7F7'; VIOLET_LIGHT='#F3EFF7'

def canvas(height):
    fig=plt.figure(figsize=(183/25.4,height/25.4), facecolor='white')
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,183); ax.set_ylim(height,0)
    ax.axis('off')
    return fig,ax

def label(ax,x,y,s,size=7,color=INK,weight='normal',ha='left',va='top',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,
                   linespacing=1.4,**kw)

def rule(ax,y): ax.plot([4,179],[y,y],c=LINE,lw=.6)

def panel(ax,letter,title,y):
    label(ax,4,y,letter,size=9,weight='bold')
    label(ax,10,y,title,size=8,weight='bold')

def box(ax,x,y,w,h,title,body='',color=BLUE,fill=BLUE_LIGHT,ts=7):
    ax.add_patch(Rectangle((x,y),w,h,fc=fill,ec=color,lw=.65))
    label(ax,x+w/2,y+3,title,size=ts,weight='bold',ha='center',color=color)
    if body: label(ax,x+w/2,y+9,body,size=6.3,ha='center')

def arrow(ax,p,q,color=MUTED,lw=.8,connectionstyle='arc3'):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=8,
                                linewidth=lw,color=color,connectionstyle=connectionstyle,
                                shrinkA=0,shrinkB=0))

def elbow(ax,points,color=MUTED):
    for p,q in zip(points[:-2],points[1:-1]):
        ax.plot([p[0],q[0]],[p[1],q[1]],c=color,lw=.75)
    arrow(ax,points[-2],points[-1],color=color)

def head(ax,x,y,view,scale=.8):
    """Abstract view glyph, not a patient image or anatomical measurement."""
    if view=='profile':
        verts=[(-3,5),(-4,1),(-4,-4),(0,-6),(3,-4),(3,-1),(5,1),(3,1.8),
               (3,3),(1,4),(1,6)]
        codes=[MPath.MOVETO]+[MPath.LINETO]*(len(verts)-1)
        patch=PathPatch(MPath([(x+a*scale,y+b*scale) for a,b in verts],codes),
                        fill=False,ec=BLUE,lw=.85)
        ax.add_patch(patch)
        ax.plot([x+1.5*scale],[y-.3*scale],'.',c=BLUE,ms=1.8)
    else:
        ax.add_patch(Ellipse((x,y),7*scale,10*scale,fc='none',ec=BLUE,lw=.85))
        eyes=[-1.5,1.5] if view=='frontal' else [-.5,2]
        for dx in eyes: ax.plot([x+dx*scale],[y-scale],'.',c=BLUE,ms=1.8)
        nx=0 if view=='frontal' else 1
        ax.plot([x+nx*scale,x+(nx+.5)*scale,x+nx*scale],
                [y,y+1.3*scale,y+1.5*scale],c=BLUE,lw=.6)
        ax.plot([x-scale,x+scale],[y+2.8*scale,y+2.8*scale],c=BLUE,lw=.6)

def save(fig,name):
    for suffix in ['svg','pdf','png']:
        fig.savefig(BASE/f'{name}.{suffix}',dpi=220)
    plt.close(fig)

fig,ax=canvas(193)
label(ax,4,2,'CLINICAL CRANIOFACIAL PRETRAINING → MULTIMODAL DIAGNOSIS',6.6,color=MUTED)
label(ax,179,2,'DESIGN DRAFT',6.2,color=MUTED,ha='right')
panel(ax,'a','Clinical visual resource',10)
label(ax,5,19,f"{train['images']:,}",18,BLUE,weight='bold')
label(ax,5,28,'clinical photographs',6.5)
label(ax,74,19,f"{train['groups']:,}",18,BLUE,weight='bold')
label(ax,74,28,'patient / pseudo-ID groups',6.5)
label(ax,179,20,'Released training partition\nHistorical counts: 19 Oct 2026',6.1,MUTED,ha='right')

start=5; width=173
colors={'frontal':'#2C6DA8','profile':'#648DB4','oblique':'#A2BBD1'}
for view in ['frontal','profile','oblique']:
    w=width*train['views'][view]/train['images']
    ax.add_patch(Rectangle((start,35),w,5.5,fc=colors[view],ec='white',lw=.6))
    start+=w
label(ax,5,42,'Frontal 35,171     ·     Profile 52,801     ·     Oblique 9,821',6.7)
label(ax,5,47,'Reference source: 86,606 general-face images; clinical health status unverified.',6,MUTED)
rule(ax,53)

panel(ax,'b','Staged learning from unpaired and paired clinical data',57)
label(ax,5,64,'1  VISUAL PRETRAINING',6.2,BLUE,weight='bold')
box(ax,5,69,42,24,'Clinical photographs',color=BLUE)
for i,view in enumerate(['frontal','profile','oblique']):
    head(ax,15+i*11,85,view,.75)
arrow(ax,(48.5,81),(55,81),BLUE)
box(ax,56,69,49,24,'Self-supervised learning','Disease / HPO labels\nnot required for this stage')
arrow(ax,(106.5,81),(113,81),BLUE)
box(ax,114,69,64,24,'Craniofacial encoder','Reusable visual representation\nBenefit assessed against public initialization')

label(ax,5,99,'2  MULTIMODAL ADAPTATION — PLANNED',6.2,VIOLET,weight='bold')
box(ax,5,106,39,17,'Paired facial images','One or more available views',ts=6.6)
arrow(ax,(45.5,114.5),(51,114.5),BLUE)
box(ax,52,106,48,17,'Visual representation','Patient-level view aggregation',ts=6.6)
elbow(ax,[(146,94),(146,97),(76,97),(76,104.5)],BLUE)
box(ax,5,133,39,17,'Observed HPO terms','Unknown ≠ explicitly absent',TEAL,TEAL_LIGHT,ts=6.6)
arrow(ax,(45.5,141.5),(51,141.5),TEAL)
box(ax,52,133,48,17,'Phenotype representation','Paired development patients',TEAL,TEAL_LIGHT,ts=6.6)
box(ax,111,115,30,26,'Fusion','Diagnostic\nadaptation',VIOLET,VIOLET_LIGHT,ts=6.6)
elbow(ax,[(101,114.5),(105,114.5),(105,123),(109.5,123)],BLUE)
elbow(ax,[(101,141.5),(105,141.5),(105,134),(109.5,134)],TEAL)
arrow(ax,(142.5,128),(149,128),VIOLET)
box(ax,150,115,28,26,'Diagnosis','Ranked\ndifferential',VIOLET,VIOLET_LIGHT,ts=6.6)
label(ax,111,146,'Diagnostic labels supervise this stage.',5.7,MUTED)
rule(ax,155)

panel(ax,'c','Independent clinical evaluation — planned',159)
columns=[(5,55,'CHOP expert cohort','Expert-referenced diagnoses\nPhenotype validity where annotated'),
         (65,55,'BCH external cohort','Locked patient-level evaluation\nTop-5 distinct diagnoses + uncertainty'),
         (125,53,'Clinician assistance','Optional reader study\nDiagnostic decisions with / without AI')]
for x,w,title,body in columns:
    ax.plot([x,x+w],[168,168],c=INK,lw=1)
    label(ax,x,170,title,6.9,weight='bold')
    label(ax,x,176,body,6)
label(ax,5,187,'Evaluation patients must be excluded from all development stages. Downstream benefits remain unproven.',5.8,MUTED)
save(fig,'figure1_study_framework_draft')

fig,ax=canvas(169)
label(ax,5,5,'A single clinical question, five figures',12,weight='bold')
label(ax,5,14,'Does clinical craniofacial pretraining improve diagnosis when combined with HPO?',7.2,MUTED)
label(ax,5,21,'MANUSCRIPT STORYBOARD  ·  PLANNED EVIDENCE, NOT OBSERVED RESULTS',6,MUTED)

rows=[
    ('01','Define the resource and the task','Cohorts → visual SSL → paired facial–HPO adaptation',
     'Show clinical counts, data boundaries and the diagnostic output.',BLUE),
    ('02','Establish what pretraining adds','Public vs clinical encoder × image-only vs image + HPO',
     'Matched downstream training; label efficiency; expert morphology.',BLUE),
    ('03','Test independent diagnostic performance','CHOP expert reference + BCH held-out evaluation',
     'Primary endpoint, strong comparators, HPO-only control and paired effects.',VIOLET),
    ('04','Measure the limits of robustness','Incomplete HPO + missing views + cohort differences',
     'Natural missingness, uncertainty, error patterns and denominators.',TEAL),
    ('05','Connect predictions to clinical decisions','Reader study, if conducted; otherwise retrospective validity',
     'Assisted decisions, time and errors. Do not infer unmeasured clinical benefit.',INK),
]
for i,(n,title,detail,meaning,color) in enumerate(rows):
    y=32+i*25
    ax.plot([5,178],[y,y],c=LINE,lw=.65)
    label(ax,5,y+4,n,13,color,weight='bold')
    label(ax,22,y+3,title,8.3,weight='bold')
    label(ax,22,y+10,detail,7,color)
    label(ax,22,y+16,meaning,6.6,MUTED)
label(ax,5,161,'Visual conventions: white ground · editable vectors · fixed modality colors · explicit uncertainty',6.5,MUTED)
save(fig,'five_figure_storyboard')

manifest={
    'status':'writing_and_design_draft',
    'aggregate_source':str(SOURCE),
    'aggregate_snapshot_utc':aggregate['snapshot_at_utc'],
    'patient_images_or_rows_read':False,
    'private_mapping_read':False,
    'patient_imagery_in_figures':False,
    'simulated_performance_in_figures':False,
    'downstream_stages':'proposed_not_completed',
    'outputs':[p.name for p in BASE.iterdir() if p.suffix in ['.svg','.pdf','.png','.md']],
}
(BASE/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'rendered':manifest['outputs']}))
