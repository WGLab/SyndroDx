"""Publication-oriented architecture drawing; no clinical image/row I/O.

Panel b is grounded in the local DINOv3 implementation. Panel c is a
concrete proposed downstream design, not an implemented or validated model.
"""
from pathlib import Path
import json
import hashlib
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, PathPatch
from matplotlib.path import Path as P

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'experiment_results_20261019/aggregate_results.json'
aggregate = json.loads(SOURCE.read_text())
training = aggregate['data']['released_partitions']['train']
assert sum(training['views'].values()) == training['images']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':6.6,
                     'svg.fonttype':'none','pdf.fonttype':42,
                     'savefig.facecolor':'white','mathtext.default':'regular'})
INK='#27303A'; GRAY='#65717C'; LINE='#D7DFE6'
B='#83A3CC'; BD='#355F90'; BF='#EFF4FB'
G='#8ABAA6'; GD='#387D69'; GF='#EDF6F1'
PINK='#B09BCB'; PD='#72558D'; PF='#F5F0F8'
SAND='#EBD29E'; SAND_DARK='#9C7B39'; GR='#AAB4C0'; GRF='#F3F5F7'

class Drawing:
    def __init__(self,height):
        self.height=height
        self.fig=plt.figure(figsize=(183/25.4,height/25.4),facecolor='white')
        self.ax=self.fig.add_axes([0,0,1,1]); self.ax.set(xlim=(0,183),ylim=(height,0))
        self.ax.axis('off'); self.texts=[]

    def text(self,x,y,s,size=6.6,color=INK,bold=False,ha='left',va='top',**kw):
        t=self.ax.text(x,y,s,fontsize=size,color=color,weight='bold' if bold else 'normal',
                       ha=ha,va=va,linespacing=1.25,**kw)
        self.texts.append(t)
        return t

    def line(self,points,color=GRAY,lw=.65,style='-'):
        self.ax.plot([p[0] for p in points],[p[1] for p in points],c=color,lw=lw,ls=style,zorder=2)

    def arrow(self,p,q,color=GRAY,lw=.7,style='-'):
        self.ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=7,
                          color=color,lw=lw,linestyle=style,shrinkA=0,shrinkB=0,zorder=3))

    def elbow(self,points,color=GRAY,lw=.65,style='-'):
        if len(points)>2: self.line(points[:-1],color,lw,style)
        self.arrow(points[-2],points[-1],color,lw,style)

    def rect(self,x,y,w,h,fc='white',ec=LINE,lw=.6):
        self.ax.add_patch(Rectangle((x,y),w,h,fc=fc,ec=ec,lw=lw,zorder=1))

    def region(self,x,y,w,h,fc,ec='none',r=1,lw=.65):
        self.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                          fc=fc,ec=ec,lw=lw,zorder=1))

    def block(self,x,y,w,h,title,body='',color=BD,fill=BF,ts=6.6,bs=5.9):
        self.region(x,y,w,h,fill,color)
        if body:
            self.text(x+w/2,y+2,title,ts,color,True,'center')
            self.text(x+w/2,y+7,body,bs,INK,ha='center')
        else:
            self.text(x+w/2,y+h/2,title,ts,color,True,'center','center')

    def panel(self,letter,title,y,note=None):
        self.text(3,y,letter,9,bold=True)
        self.text(9,y+.35,title,7.4,bold=True)
        if note: self.text(179,y+.8,note,5.7,GRAY,ha='right')

    def head(self,x,y,s=.65,view='front'):
        if view=='profile':
            v=[(-3,6),(-3,4),(-5,2),(-4,-3),(-3,-7),(3,-7),(3,-2),
               (3,-1),(4,0),(5,1),(5.5,1.5),(3.5,2),(3.6,2.7),
               (4,3.3),(3.7,4.7),(1,5),(1,6.5)]
            c=[P.MOVETO]+[P.CURVE4]*9+[P.LINETO]*3+[P.CURVE4]*3+[P.LINETO]
            self.ax.add_patch(PathPatch(P([(x+a*s,y+b*s) for a,b in v],c),
                                       ec=BD,fc='none',lw=.65))
            self.ax.add_patch(Ellipse((x-.8*s,y+.3*s),1.4*s,2.2*s,ec=BD,fill=False,lw=.45))
        else:
            self.ax.add_patch(Ellipse((x,y),8*s,11*s,fill=False,ec=BD,lw=.65))
            z=0 if view=='front' else 1
            for dx in [-2,1]: self.line([(x+(dx+z)*s,y-s),(x+(dx+z+1)*s,y-s)],BD,.45)
            self.line([(x+z*s,y),(x+(z+.4)*s,y+1.5*s),(x+(z-.3)*s,y+1.8*s)],BD,.45)
            self.line([(x-s,y+3*s),(x+s,y+3*s)],BD,.45)

    def token(self,x,y,w=3,h=4,color=B,label=None,outlined=False):
        self.rect(x,y,w,h,'white' if outlined else color,color if outlined else 'white',.45)
        if label: self.text(x+w/2,y+h/2,label,5.4,INK,ha='center',va='center')

    def grid(self,x,y,n=3,size=1.6,masked=False,color=B):
        for i in range(n):
            for j in range(n):
                self.rect(x+j*size,y+i*size,size-.25,size-.25,
                          '#FFFFFF' if masked and (i+j)%3==0 else color,BD,.28)

    def hospital(self,x,y,color=BD):
        self.rect(x,y,7,10,'white',color)
        self.line([(x+2.3,y+2),(x+4.7,y+2)],color,.9)
        self.line([(x+3.5,y+.8),(x+3.5,y+3.2)],color,.9)
        for dx in [1.5,4.5]:
            for dy in [4.5,6.5]: self.rect(x+dx,y+dy,1,1,color,color,.3)
        self.rect(x+2.7,y+8.3,1.6,1.7,'white',color,.4)

    def save(self,stem):
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer(); page=self.fig.bbox
        clipped=[]
        for t in self.texts:
            b=t.get_window_extent(renderer)
            if b.x0 < page.x0-1 or b.x1 > page.x1+1 or b.y0 < page.y0-1 or b.y1 > page.y1+1:
                clipped.append(t.get_text())
        if clipped: raise RuntimeError('Text outside page: '+repr(clipped))
        for suffix in ['svg','pdf','png']:
            self.fig.savefig(ROOT/f'{stem}.{suffix}',dpi=300)
        root=ET.parse(ROOT/f'{stem}.svg').getroot(); ns={'s':'http://www.w3.org/2000/svg'}
        assert root.findall('.//s:text',ns)
        assert not root.findall('.//s:image',ns), 'Raster/patient image unexpectedly embedded'
        plt.close(self.fig)


def data_panel(d):
    d.panel('a','Clinical data for visual representation learning',3,'Aggregate snapshot · 19 Oct 2026')
    x=8
    for name,col in [('frontal',B),('profile',G),('oblique',SAND)]:
        n=training['views'][name]; w=171*n/training['images']
        d.rect(x,12,w,9,col,'white',.5)
        if name=='oblique':
            s=f'Oblique\n{100*n/training["images"]:.1f}%'; fs=5.7
        else:
            s=f'{name.capitalize()}  {n:,}\n{100*n/training["images"]:.1f}%'; fs=6.3
        d.text(x+w/2,13.4,s,fs,ha='center'); x+=w
    d.text(8,24,'97,793 clinical photographs · 13,857 recorded patient groups',6.5)
    d.text(179,24,'Oblique: 9,821 images',5.8,GRAY,ha='right')
    d.text(8,29,'Current development mixture: 80% clinical / 20% general-face sampling; reference collection = 86,606 images.',5.8,GRAY)


def ssl_panel(d,y,letter='b'):
    d.panel(letter,'Clinical visual pretraining',y,'Implementation-grounded')
    d.text(7,y+7,'One clinical image',5.8,BD)
    d.rect(7,y+27,10,15,'white',B); d.head(12,y+33,.7)
    d.text(12,y+44,'Augment',5.5,GRAY,ha='center')
    d.arrow((18,y+34),(21,y+34),BD)
    d.line([(21,y+15),(21,y+50)],BD,.55)
    for yy in [15,31,50]: d.arrow((21,y+yy),(25,y+yy),BD,.55)

    d.grid(27,y+11,3,1.7,True); d.grid(34,y+11,3,1.7,True)
    d.text(26,y+18,'2 global · 256 × 256',5.6,BD)
    d.text(26,y+21.6,'Token masking',5.3,GRAY)
    for j in range(8): d.grid(27+(j%4)*3.1,y+27+(j//4)*3.1,2,1.1,False)
    d.text(26,y+35,'8 local · 112 × 112',5.6,BD)
    d.grid(27,y+46,3,1.7,False,color=GR); d.grid(34,y+46,3,1.7,False,color=GR)
    d.text(26,y+53,'2 global · unmasked',5.6,GRAY)

    d.elbow([(42,y+14),(49,y+14),(49,y+19),(54,y+19)],BD)
    d.elbow([(42,y+30),(49,y+30),(49,y+25),(54,y+25)],BD)
    d.arrow((42,y+49),(54,y+49),GRAY)

    d.block(55,y+12,33,19,'Student ViT-7B','16 × 16 patch embedding\nTransformer backbone',BD,BF,6.8,5.7)
    d.block(55,y+40,33,19,'EMA teacher','Same architecture\nStop-gradient',GRAY,GRF,6.7,5.7)
    d.arrow((71.5,y+32.5),(71.5,y+38.5),GRAY,style='--')
    d.text(74,y+34.2,'EMA update',5.2,GRAY)

    # Backbone outputs enter distinct DINO and iBOT projection heads.
    for base,teacher in [(12,False),(40,True)]:
        color=GRAY if teacher else BD
        d.line([(89,y+base+9.5),(94,y+base+9.5)],color)
        d.line([(94,y+base+4.5),(94,y+base+15)],color)
        d.arrow((94,y+base+4.5),(100,y+base+4.5),PD)
        d.arrow((94,y+base+15),(100,y+base+15),color)
        d.token(96,y+base+2.3,3,4,PINK)
        for j in range(2): d.token(95.5+j*1.8,y+base+13.4,1.4,3.1,GR if teacher else B)
        d.block(101,y+base+.7,25,7.6,'DINO head',color=PD,fill=PF,ts=6.2)
        d.block(101,y+base+11.2,25,7.6,'iBOT head',color=color,fill=GRF if teacher else BF,ts=6.2)
        who='t' if teacher else 's'
        d.arrow((127,y+base+4.5),(143,y+base+4.5),PD,.55)
        d.arrow((127,y+base+15),(143,y+base+15),color,.55)
        d.text(134.7,y+base+.2,rf'$p_{{{who}}}^{{\mathrm{{CLS}}}}$',5.6,PD,ha='center')
        d.text(134.7,y+base+10.7,rf'$p_{{{who}}}^{{\mathrm{{patch}}}}$',5.6,color,ha='center')
    d.line([(143,y+16.5),(143,y+55)],GRAY,.6)
    d.arrow((143,y+35.5),(148,y+35.5),GRAY)
    d.region(150,y+13,29,44,'#FAFBFC',LINE)
    d.text(164.5,y+16,'Self-distillation',6.2,bold=True,ha='center')
    d.text(164.5,y+22,'Teacher: Sinkhorn targets',5.2,GRAY,ha='center')
    d.line([(153,y+27),(176,y+27)],LINE,.55)
    d.text(164.5,y+30,'DINO',6.5,PD,True,'center')
    d.text(164.5,y+34.3,'CLS across views',5.5,ha='center')
    d.text(164.5,y+42,'iBOT',6.5,BD,True,'center')
    d.text(164.5,y+46.5,'Masked global patches',5.5,ha='center')
    d.text(7,y+63,'Objectives: DINO + iBOT + 0.1 × KoLeo',6.1)
    d.text(179,y+63,'Transfer EMA backbone; discard SSL heads.',5.8,GRAY,ha='right')


def fusion_panel(d,y,letter='c'):
    d.panel(letter,'Patient-level facial–HPO fusion',y,'Proposed downstream architecture')
    d.text(7,y+7,'Available facial views',6.0,BD,bold=True)
    d.text(43,y+7,'Frozen visual backbone',6.0,BD,bold=True)
    d.text(79,y+7,'View-level tokens',6.0,BD,bold=True)
    d.text(144,y+7,'Compact fusion',6.0,PD,bold=True)

    d.region(6,y+13,29,24,BF)
    for x,view in [(12,'front'),(21,'profile'),(30,'oblique')]:
        d.rect(x-3.3,y+16,6.8,12,'white',B,.5); d.head(x,y+21.2,.55,view)
    d.text(20.5,y+30,'Frontal · profile · oblique',5.4,BD,ha='center')
    d.text(20.5,y+34,'One patient',5.5,GRAY,ha='center')
    d.arrow((36.5,y+24),(41,y+24),BD)
    d.block(42,y+15,30,19,'Shared ViT-7B','Per-image CLS feature\nFrozen initially',BD,BF,6.3,5.6)
    d.elbow([(73.5,y+24),(76,y+24),(76,y+18),(78,y+18)],BD)
    d.block(79,y+14,29,8,'Within-view mean',color=BD,fill=BF,ts=6.0)
    d.arrow((93.5,y+23),(93.5,y+25),BD)
    d.block(79,y+26,29,8,'Linear → 256',color=BD,fill=BF,ts=6.1)
    d.text(93.5,y+36,'+ view / modality embeddings',5.4,BD,ha='center')
    d.arrow((109.5,y+30),(113,y+30),BD)
    for j,t in enumerate(['F','P','O']): d.token(114+j*4.2,y+27.5,3.6,5.2,B,t)
    d.text(119.5,y+34.2,'≤3 tokens',5.3,BD,ha='center')

    d.text(7,y+43,'Observed HPO terms',6.0,GD,bold=True)
    d.region(6,y+49,29,20,GF)
    for j,s in enumerate(['Term names / definitions','Present or explicit negative','Unrecorded = unknown']):
        d.text(20.5,y+52+j*4.8,s,5.3,GD if j!=2 else GRAY,ha='center')
    d.arrow((36.5,y+59),(41,y+59),GD)
    d.block(42,y+49,30,20,'Frozen term encoder','One embedding per term\nBiomedical text backbone',GD,GF,6.1,5.4)
    d.elbow([(73.5,y+59),(76,y+59),(76,y+54.5),(78,y+54.5)],GD)
    d.block(79,y+50,29,9,'Linear → 256',color=GD,fill=GF,ts=6.1)
    d.text(93.5,y+61,'+ finding-state /\nmodality embeddings',5.5,GD,ha='center')
    d.arrow((109.5,y+55),(113,y+55),GD)
    for j,t in enumerate(['h₁','h₂','…']): d.token(114+j*4.2,y+52.4,3.6,5.2,G,t)
    d.text(119.5,y+59,'N term tokens',5.3,GD,ha='center')

    # Both token sets form the same attention memory. An availability mask
    # controls padding / removed tokens; absence is not a negative finding.
    d.elbow([(126.8,y+30),(131,y+30),(131,y+33),(137,y+33)],BD)
    d.elbow([(126.8,y+55),(131,y+55),(131,y+41),(137,y+41)],GD)
    d.text(132,y+27,'K, V',5.6,GRAY)
    d.text(132,y+45,'K, V',5.6,GRAY)
    d.token(155,y+14,5,5,PINK,'q')
    d.text(162,y+14.5,'Learned\npatient query',5.5,PD)
    d.arrow((157.5,y+20),(157.5,y+25.5),PD)
    d.text(160,y+21.5,'Q',5.6,PD)
    d.region(139,y+28,40,18,'white',PINK,.8,.5)
    d.region(137,y+26,40,18,PF,PINK,.8,.7)
    d.text(157,y+29,'Cross-attention',6.6,PD,True,'center')
    d.line([(140,y+35),(174,y+35)],'#DCCFE8',.55)
    d.text(157,y+37,'Residual + feed-forward',5.7,PD,ha='center')
    d.text(157,y+47,'2 blocks · 4 heads · d = 256',5.5,PD,ha='center')
    d.arrow((157,y+51),(157,y+53),PD)
    d.ax.add_patch(Circle((157,y+55),1.5,fc=PINK,ec=PD,lw=.5))
    d.text(162,y+53.4,'Patient embedding',5.4,PD)
    d.arrow((157,y+57),(157,y+59.5),PD)
    d.block(138,y+61,40,8,'Linear disease scores',color=PD,fill=PF,ts=6.0)
    d.arrow((157,y+70),(157,y+72),PD)
    d.text(157,y+74,'Top-k distinct diagnoses',5.9,PD,bold=True,ha='center')
    d.text(157,y+79,'Fixed candidate disease set',5.3,GRAY,ha='center')

    # A compact mask legend and training target make the interface explicit.
    d.rect(7,y+74,3.3,3.3,'white',GRAY,.6)
    d.line([(7.3,y+76.9),(10,y+74.3)],GRAY,.6)
    d.text(12,y+73.5,'Availability masks + modality dropout',5.8,GRAY)
    d.text(7,y+79,'Missing views / unrecorded HPO contribute no observed tokens.',5.6,GRAY)
    d.text(7,y+84,'Paired training: reference diagnosis y → cross-entropy on disease scores. At least one modality is retained.',5.6,GRAY)


def evaluation_panel(d,y):
    d.panel('d','Independent clinical assessment',y,'Planned')
    d.hospital(8,y+10,BD)
    d.text(18,y+10,'CHOP expert cohort',6.2,bold=True)
    d.text(18,y+15,'Reference diagnosis / phenotype',5.6,GRAY)
    d.hospital(78,y+10,GD)
    d.text(88,y+10,'BCH external cohort',6.2,bold=True)
    d.text(88,y+15,'Locked diagnostic evaluation',5.6,GRAY)
    d.text(146,y+10,'Required controls',6.0,PD,bold=True)
    d.text(146,y+15,'Image-only · HPO-only\nPublic encoder + HPO',5.5,GRAY)
    d.text(8,y+24,'Patient / family separation · diagnosis-level Top-k · missing-information sensitivity · calibration',5.7,GRAY)


main=Drawing(234)
data_panel(main)
main.line([(3,35),(180,35)],LINE,.65)
ssl_panel(main,39)
main.line([(3,108),(180,108)],LINE,.65)
fusion_panel(main,112)
main.line([(3,201),(180,201)],LINE,.65)
evaluation_panel(main,204)
main.save('figure1_architecture_v3')

detail=Drawing(166)
ssl_panel(detail,3,'a')
detail.line([(3,72),(180,72)],LINE,.65)
fusion_panel(detail,76,'b')
detail.save('architecture_detail_v3')

sources={
    'training_config': '/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official/training/config.yaml',
    'official_ssl_implementation': str(ROOT.parent/'training_diagnostics_20261017/vendor/dinov3_fino/dinov3/train/ssl_meta_arch.py'),
    'warmup_config': str(ROOT.parent/'training_diagnostics_20261017/warmup_config.py'),
    'controlled_continuation': str(ROOT.parent/'controlled_cpt_20261019/configuration.py'),
}
manifest={
    'version':3,'type':'original vector architecture design',
    'patient_images_read':False,'patient_rows_read':False,'private_mapping_read':False,
    'actual_training_config_changed':False,'fusion_module_implemented':False,
    'clinical_results_claimed':False,
    'aggregate_snapshot':aggregate['snapshot_at_utc'],
    'reference_figure':'https://www.nature.com/articles/s41591-025-03747-y/figures/1',
    'implementation_sources':sources,
    'implementation_sha256':{k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in sources.items()},
    'proposed_fusion':{'dimension':256,'cross_attention_blocks':2,'heads':4,'ffn_width':1024,
                      'patient_queries':1,'term_encoder':'frozen biomedical text model; exact checkpoint not selected',
                      'visual_tokens':'one projected CLS mean per available view',
                      'term_tokens':'one per observed term, with present/explicit-negative state',
                      'loss':'cross entropy for one adjudicated primary diagnosis',
                      'candidate_universe':'fixed; no unseen-disease capability implied'},
}
(ROOT/'architecture_v3_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'rendered':['figure1_architecture_v3','architecture_detail_v3'],
                  'editable_svg':True,'raster_images_embedded':False}))
