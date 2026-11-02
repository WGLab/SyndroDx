"""Build a patient-balanced, training-only constant face prior for matched control."""
import numpy as np
from runtime import *
def main():
    patients=load(WORK/'private/sft_patients.json')['patients']
    patches=np.load(WORK/'private/patches.npy',mmap_mode='r');features=np.load(WORK/'private/global_features.npy',mmap_mode='r')
    p=np.zeros((144,768),dtype=np.float64);g=np.zeros(512,dtype=np.float64);n=0
    for row in patients:
        if row['split']!='train':continue
        ix=row['feature_indices'];p+=np.array(patches[ix],dtype=np.float32).mean(0);g+=np.array(features[ix],dtype=np.float32).mean(0);n+=1
    path=WORK/'private/hpo_constant_prior.npz'
    np.savez(path,patches=np.tile((p/n).astype(np.float32),(2,1)),global_features=(g/n).astype(np.float32))
    save(WORK/'hpo_control_protocol.json',dict(state='complete',version=2,
        input='same train-patient-averaged constant features for every train and validation patient; no individual face information',
        connector='same trainable architecture and aligned initialization as fusion',
        train_patients_used=n,validation_patients_used=0,prior_sha256=digest(path),
        supersedes='zero-prefix pilot retained separately; not the matched comparison',updated_at_utc=now()))
    print('Training-only constant-prefix control sealed.')
if __name__=='__main__':main()
