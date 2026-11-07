"""Record task/GLM/image/model correspondence without changing any input."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model-check', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(str(output))
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg['model_rsms']).parents[2]
    old = root.parent/'localize'
    check = json.loads(Path(args.model_check).read_text())
    if not check['passed'] or check['features_sha256'] != digest(Path(cfg['model_rsms'])):
        raise ValueError('Independent model check has not passed for these features')
    task = old/'expcode/localize.py'
    source = task.read_text()
    required = ["alpha = string.ascii_uppercase", "items = np.arange(1,17)",
                "nindex = alpha.index(trials[0])", "curr_image = preloadims[nindex]",
                "./../images/final/im{}of16.png", "for letter in alpha[:16]"]
    if not all(re.sub(r'\s+', '', text) in re.sub(r'\s+', '', source) for text in required):
        raise ValueError('Task source mapping changed; inspect it before proceeding')
    records = []
    for subject in cfg['subjects']:
        designs = sorted((root/'subjects'/subject/'preprocess').glob('func*.feat/design.fsf'))
        if not designs:
            raise ValueError('No run designs: '+subject)
        for design in designs:
            text = design.read_text()
            titles = dict((int(i), value) for i,value in
                          re.findall(r'set fmri\(evtitle(\d+)\) "([^"]+)"', text))
            custom = dict((int(i), value) for i,value in
                          re.findall(r'set fmri\(custom(\d+)\) "([^"]+)"', text))
            contrast = dict(((int(i),int(j)), float(value)) for i,j,value in
                            re.findall(r'set fmri\(con_orig(\d+)\.(\d+)\) ([^\s]+)', text))
            for i in range(1,17):
                letter = chr(64+i)
                if titles.get(i) != letter or not custom.get(i,'').endswith('_'+letter+'.txt'):
                    raise ValueError('GLM label mismatch: '+str(design))
                for j in range(1,17):
                    if contrast.get((i,j)) != float(i==j):
                        raise ValueError('GLM contrast mismatch: '+str(design))
            records.append({'path': str(design), 'sha256': digest(design)})
    images=[]
    for index in range(16):
        name='im%02dof16.png'%(index+1)
        path=root/'images/final'/name
        if path.read_bytes() != (old/'images/final'/name).read_bytes():
            raise ValueError('Old/current stimulus differs: '+name)
        if digest(path) != check['images'][index]['sha256']:
            raise ValueError('Model check used different image: '+name)
        images.append({'condition':chr(65+index),'cope':index+1,'model_index':index,
                       'filename':name,'sha256':digest(path)})
    report={'passed':True,'task_source':str(task),'task_sha256':digest(task),
            'mapping':images,'run_design_count':len(records),'run_designs':records,
            'model_check':str(Path(args.model_check)), 'model_check_sha256':digest(Path(args.model_check)),
            'features_sha256':digest(Path(cfg['model_rsms'])),
            'scope':'Task source, current run designs, identical historical/current PNGs, independently recomputed model RSMs',
            'limitation':'Does not independently replay each historical presentation log or re-fit higher-level GLMs'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'passed':True,'run_design_count':len(records),'output':str(output)}))


if __name__=='__main__':
    main()
