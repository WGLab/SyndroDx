"""Original multipanel scientific diagram; only aggregate release statistics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, PathPatch
from matplotlib.path import Path as P

ROOT=Path(__file__).resolve().parent
summary=json.loads((ROOT.parent/'experiment_results_20261019/aggregate_results.json').read_text())
parts=summary['data']['released_partitions']; training=parts['train']
assert sum(training['views'].values())==training['images']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':6.5,'svg.fonttype':'none',
                     'pdf.fonttype':42,'savefig.facecolor':'white','axes.linewidth':.6})
INK='#252B33'; GRAY='#65717C'; LIGHT='#D8DFE5'
BLUE='#789BC6'; BLUE_DARK='#375D89'; BLUE_FILL='#EDF3FA'
GREEN='#85B7A1'; GREEN_DARK='#347E69'; GREEN_FILL='#ECF5F0'
SAND='#EFCF95'; PURPLE='#A493BF'; PURPLE_DARK='#6B558B'; PURPLE_FILL='#F3EFF8'
H=202
fig=plt.figure(figsize=(183/25.4,H/25.4),facecolor='white')
ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,183),ylim=(H,0)); ax.axis('off')

def text(x,y,t,size=6.5,color=INK,bold=False,ha='left',va='top',**kw):
    return ax.text(x,y,t,fontsize=size,color=color,weight='bold' if bold else 'normal',
                   ha=ha,va=va,linespacing=1.25,**kw)

def panel(letter,title,x,y):
    text(x,y,letter,9,bold=True); text(x+6,y+.3,title,7,bold=True)

def line(x1,y1,x2,y2,color=GRAY,lw=.65,style='-'):
    ax.plot([x1,x2],[y1,y2],color=color,lw=lw,ls=style)

def arrow(x1,y1,x2,y2,color=GRAY,style='-'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=7,
                 lw=.7,color=color,linestyle=style,shrinkA=0,shrinkB=0))

def region(x,y,w,h,fc,ec='none',radius=1.5,lw=.6):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
                              fc=fc,ec=ec,lw=lw))

def rect(x,y,w,h,fc='white',ec=LIGHT,lw=.6):
    ax.add_patch(Rectangle((x,y),w,h,fc=fc,ec=ec,lw=lw))

def path(points,codes,color=BLUE_DARK,fill='none',lw=.7):
    ax.add_patch(PathPatch(P(points,codes),ec=color,fc=fill,lw=lw))

def head(x,y,s=1,view='frontal',color=BLUE_DARK):
    # Generic anatomical outline. No patient likeness, measurements or diagnoses.
    if view=='profile':
        v=[(-3,6),(-3,4),(-5,2),(-4,-3),(-3,-7),(3,-7),(3,-2),
           (3,-1),(4,0),(5,1),(5.5,1.5),(3.5,2),(3.6,2.7),
           (4,3.3),(3.7,4.7),(1,5),(1,6.5)]
        c=[P.MOVETO,P.CURVE4,P.CURVE4,P.CURVE4,P.CURVE4,P.CURVE4,P.CURVE4,
           P.CURVE4,P.CURVE4,P.CURVE4,P.LINETO,P.LINETO,P.LINETO,
           P.CURVE4,P.CURVE4,P.CURVE4,P.LINETO]
        path([(x+a*s,y+b*s) for a,b in v],c,color=color)
        ax.add_patch(Ellipse((x-.8*s,y+.4*s),1.4*s,2.3*s,fill=False,ec=color,lw=.5))
        line(x+1.4*s,y-.2*s,x+2.5*s,y-.2*s,color,.5)
    else:
        ax.add_patch(Ellipse((x,y),8*s,11*s,fill=False,ec=color,lw=.7))
        # Smooth neutral feature contours, not an expression.
        shift=0 if view=='frontal' else 1.1
        for dx in [-2,1]:
            line(x+(dx+shift)*s,y-1.1*s,x+(dx+1.1+shift)*s,y-1.1*s,color,.5)
        line(x+shift*s,y-.4*s,x+(shift+.4)*s,y+1.6*s,color,.5)
        line(x+(shift+.4)*s,y+1.6*s,x+(shift-.3)*s,y+1.9*s,color,.5)
        line(x-1.1*s,y+3*s,x+1.4*s,y+3*s,color,.5)
        line(x-1.8*s,y+5.1*s,x-2*s,y+7*s,color,.6)
        line(x+1.8*s,y+5.1*s,x+2*s,y+7*s,color,.6)

def photo_stack(x,y):
    for dx,dy in [(3,-2),(1.5,-1),(0,0)]: rect(x+dx,y+dy,11,16,'white',BLUE)
    head(x+5.5,y+7,.7)

def hpo_icon(x,y):
    points=[(0,0),(-5,4),(5,4),(-7,8),(-2,8),(5,8)]
    for a,b in [(0,1),(0,2),(1,3),(1,4),(2,5)]:
        line(x+points[a][0],y+points[a][1],x+points[b][0],y+points[b][1],GREEN_DARK,.7)
    for i,(dx,dy) in enumerate(points):
        ax.add_patch(Circle((x+dx,y+dy),.9,fc=GREEN if i else GREEN_DARK,ec='white',lw=.4))

def hospital(x,y,color=GRAY):
    rect(x,y,9,12,'white',color)
    line(x+3.3,y+2.5,x+5.7,y+2.5,color,1)
    line(x+4.5,y+1.3,x+4.5,y+3.7,color,1)
    for xx in [x+2,x+5.5]:
        for yy in [y+5.2,y+7.7]: rect(xx,yy,1.3,1.3,color,color,.4)
    rect(x+3.4,y+10,2.2,2,'white',color,.5)

def patch_grid(x,y,nx,ny,color,masked=False):
    for row in range(ny):
        for col in range(nx):
            fc=color if not masked or (row+col)%3 else '#FFFFFF'
            rect(x+col*2.2,y+row*2.2,1.8,1.8,fc,BLUE_DARK,.35)

def feature_tokens(x,y,color,n=4):
    for j in range(n): rect(x+j*2.4,y,1.8,3.2,color,'white',.2)

# a: actual clinical view composition; widths encode the reported counts.
panel('a','Clinical photographs used for visual pretraining',3,3)
text(179,4,'Historical release · 19 Oct 2026',5.6,GRAY,ha='right')
colors={'frontal':BLUE,'profile':GREEN,'oblique':SAND}
x=9
for name in ['frontal','profile','oblique']:
    count=training['views'][name]; w=169*count/training['images']
    rect(x,12,w,9,colors[name],'white',.5)
    pct=100*count/training['images']
    # Narrow oblique segment gets a small external label.
    if name!='oblique':
        text(x+w/2,13.5,f'{name.capitalize()}\n{count:,} ({pct:.1f}%)',6.2,INK,ha='center')
    else:
        text(x+w/2,13.5,f'Oblique\n{pct:.1f}%',5.8,INK,ha='center')
    x+=w
text(9,23,'97,793 clinical training images · 13,857 recorded patient groups',6.3)
text(178,23,'Oblique: 9,821 images',5.9,GRAY,ha='right')

# b: release partitions; c: sampling policy, not dataset-size proportions.
panel('b','Clinical release partitions',3,32)
bx=26; maxw=57; top=45
for i,(name,title) in enumerate([('train','Training'),('validation','Validation'),('test','Test')]):
    y=top+i*9.5; n=parts[name]['images']; w=maxw*n/100000
    text(bx-2,y+2,title,6.2,ha='right',va='center')
    rect(bx,y,w,4.2,['#8BA7C9','#B8C8DD','#D8E1ED'][i],'none')
    text(bx+w+1.5,y+2,f'{n:,}',6.1,va='center')
line(bx,71,bx+maxw,71,INK,.5)
for k in [0,50000,100000]:
    xx=bx+maxw*k/100000
    line(xx,71,xx,72,INK,.5)
    text(xx,73,'0' if k==0 else str(k//1000)+'k',5.7,ha='center')
text(bx+maxw/2,77,'Photographs',6,ha='center')

panel('c','Current SSL sampling policy',100,32)
pie=fig.add_axes([102/183,1-74/H,31/183,31/H])
pie.pie([.8,.2],startangle=90,counterclock=False,colors=[BLUE,'#D8DDE2'],
        wedgeprops={'width':.31,'edgecolor':'white','linewidth':1})
pie.text(0,.12,'80%',ha='center',va='center',fontsize=10,color=BLUE_DARK,weight='bold')
pie.text(0,-.26,'clinical',ha='center',va='center',fontsize=5.7,color=GRAY)
pie.axis('equal'); pie.axis('off')
rect(137,46,2.5,2.5,BLUE,'none'); text(141,45.7,'Clinical images',6.2)
rect(137,55,2.5,2.5,'#D8DDE2','none'); text(141,54.7,'General-face reference',6.0)
text(137,64,'80 : 20 target exposure\nReference: 86,606 images',5.9,GRAY)
text(106,77,'Sampling weights ≠ image-count proportions',5.7,GRAY)

# d: planned clinical use, not an illustration of achieved outcomes.
line(3,85,180,85,LIGHT,.6)
panel('d','Patient-level multimodal diagnosis and clinical evaluation',3,89)
text(179,90,'Proposed workflow',5.8,PURPLE_DARK,ha='right')
region(9,99,31,45,BLUE_FILL)
photo_stack(15,103)
head(32,110,.7,view='profile')
text(24.5,122,'Facial views',6.3,BLUE_DARK,ha='center')
hpo_icon(25,130)
text(24.5,140,'Observed HPO',6.3,GREEN_DARK,ha='center')

arrow(42,111,49,111,BLUE_DARK)
arrow(42,135,49,135,GREEN_DARK)
region(51,99,59,45,'#FAFAFC',ec=LIGHT)
region(54,103,27,13,BLUE_FILL,ec=BLUE)
text(67.5,106,'Craniofacial\nencoder',6.4,BLUE_DARK,bold=True,ha='center')
region(54,128,27,13,GREEN_FILL,ec=GREEN)
text(67.5,131,'HPO\nencoder',6.4,GREEN_DARK,bold=True,ha='center')
feature_tokens(84,107,BLUE,3); feature_tokens(84,132,GREEN,3)
line(92,108.5,95,108.5,BLUE_DARK)
line(95,108.5,95,119,BLUE_DARK)
line(92,133.5,95,133.5,GREEN_DARK)
line(95,133.5,95,125,GREEN_DARK)
region(97,116,10,12,PURPLE_FILL,ec=PURPLE)
text(102,122,'Joint\nfeatures',5.2,PURPLE_DARK,ha='center',va='center')
arrow(95,119,97,119,BLUE_DARK); arrow(95,125,97,125,GREEN_DARK)
text(80.5,146,'Paired facial–HPO diagnostic adaptation',6.0,PURPLE_DARK,ha='center')

arrow(111.5,122,117,122,PURPLE_DARK)
region(119,103,24,36,PURPLE_FILL)
rect(124,108,14,18,'white',PURPLE)
for j in range(3):
    text(126,111+j*4,str(j+1),5.1,PURPLE_DARK)
    line(129,112+j*4,135,112+j*4,PURPLE_DARK,.6)
text(131,129,'Ranked\ndiagnoses',6.1,PURPLE_DARK,ha='center')
arrow(144.5,122,150,122,GRAY)
hospital(152,104,BLUE_DARK); text(171,110,'CHOP\nExpert cohort',5.6,ha='center',va='center')
hospital(152,127,GREEN_DARK); text(171,133,'BCH\nExternal cohort',5.6,ha='center',va='center')
text(9,150,'One clinical task: diagnostic prioritization from complementary visual and structured phenotype evidence.',6.1,GRAY)

# e: small, accurate SSL schematic. f: a design matrix, not fabricated result plots.
line(3,156,180,156,LIGHT,.6)
panel('e','Visual self-supervision',3,160)
patch_grid(10,170,3,3,BLUE,masked=True)
patch_grid(10,184,3,3,BLUE)
text(9,193,'Views',5.5,GRAY)
arrow(18,173,25,173,BLUE_DARK)
arrow(18,187,25,187,BLUE_DARK)
region(27,168,24,10,BLUE_FILL,ec=BLUE)
region(27,182,24,10,'#F0F1F3',ec='#ADB5BF')
text(39,173,'Student',6.4,BLUE_DARK,ha='center',va='center')
text(39,187,'EMA teacher',6.0,GRAY,ha='center',va='center')
arrow(39,179,39,181,GRAY,style='--')
arrow(52.5,173,59,173,BLUE_DARK); arrow(52.5,187,59,187,GRAY)
feature_tokens(61,171.5,BLUE,4); feature_tokens(61,185.5,'#ADB5BF',4)
line(72,173,76,173,GRAY); line(76,173,76,178,GRAY)
line(72,187,76,187,GRAY); line(76,187,76,182,GRAY)
text(77.5,180,'DINO\n+ iBOT',5.8,BLUE_DARK,va='center')

panel('f','Tests of the central claim',99,160)
text(139,167,'Images',5.8,ha='center')
text(168,167,'Images + HPO',5.8,ha='center')
text(124,176,'Public encoder',5.8,ha='right',va='center')
text(124,186,'Clinical encoder',5.8,ha='right',va='center')
for xx,yy,fc,ec in [(128,172,'#EFF1F4','#AAB5C0'),(157,172,'#EFF1F4','#AAB5C0'),
                    (128,182,BLUE_FILL,BLUE),(157,182,PURPLE_FILL,PURPLE)]:
    region(xx,yy,22,8,fc,ec=ec,radius=.7)
for x,y,label_ in [(139,176,'Baseline'),(168,176,'Baseline'),(139,186,'Adapted'),(168,186,'Full model')]:
    text(x,y,label_,5.6,ha='center',va='center')
text(101,193,'Add HPO-only control; report CHOP / BCH separately.',5.6,GRAY)
text(3,199,'Design draft · Aggregate counts only · Multimodal gains and independent clinical validation remain to be measured.',5.3,GRAY)

stem='figure1_panderm_inspired_v2'
for ext in ['svg','pdf','png']:
    fig.savefig(ROOT/f'{stem}.{ext}',dpi=260)
plt.close(fig)
manifest={
    'artifact':stem,'version':2,'purpose':'original scientific framework design',
    'reference':'https://www.nature.com/articles/s41591-025-03747-y/figures/1',
    'reference_access':'public author PDF; figure 1 visually inspected',
    'source_snapshot':summary['snapshot_at_utc'],
    'patient_images_read':False,'patient_rows_read':False,
    'public_paper_panels_reproduced':False,'synthetic_performance_values':False,
    'panels_d_f':'proposed workflow and comparisons, not completed clinical results',
}
(ROOT/f'{stem}_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'outputs':[str(ROOT/f'{stem}.{ext}') for ext in ['svg','pdf','png']]}))
