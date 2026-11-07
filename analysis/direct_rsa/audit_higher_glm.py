"""Check actual higher-level FEAT inputs, rather than only copied run designs."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text());output=Path(args.output)
    if output.exists():raise FileExistsError(str(output))
    records=[];issues=[];source_count=0
    for subject in cfg['subjects']:
        mask=Path(cfg['mask'].format(subject=subject));design=mask.parent/'design.fsf'
        text=design.read_text()
        selected={int(i):int(v) for i,v in re.findall(r'set fmri\(copeinput\.(\d+)\) (\d+)',text)}
        if selected!={i:1 for i in range(1,17)}:issues.append(subject+': selected contrasts differ')
        if not re.search(r'set fmri\(ncopeinputs\) 16\s',text):issues.append(subject+': ncopeinputs not 16')
        sources=[]
        for run,path in re.findall(r'set feat_files\((\d+)\) ([^\n]+)',text):
            source=Path(path.strip().strip('"'))/'design.fsf'
            if not source.exists():issues.append('missing '+str(source));continue
            run_text=source.read_text();source_count+=1
            titles={int(i):v for i,v in re.findall(r'set fmri\(evtitle(\d+)\) "([^"]+)"',run_text)}
            contrasts={(int(i),int(j)):float(v) for i,j,v in re.findall(r'set fmri\(con_orig(\d+)\.(\d+)\) ([^\s]+)',run_text)}
            if any(titles.get(i)!=chr(64+i) for i in range(1,17)):issues.append('EV labels '+str(source))
            if any(contrasts.get((i,j))!=float(i==j) for i in range(1,17) for j in range(1,17)):issues.append('contrast identity '+str(source))
            sources.append({'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
        if not sources:issues.append('no verified run inputs '+subject)
        records.append({'subject':subject,'higher_design':str(design),'sha256':hashlib.sha256(design.read_bytes()).hexdigest(),'sources':sources})
    report={'passed':not issues,'higher_designs':len(records),'referenced_run_designs':source_count,'issues':issues,'records':records,
            'scope':'all 16 contrasts selected in each higher-level design; actual referenced run EV A-P and 16x16 identity contrasts checked'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))
    if issues:raise ValueError('Higher-level GLM identity check failed')


if __name__=='__main__':main()
