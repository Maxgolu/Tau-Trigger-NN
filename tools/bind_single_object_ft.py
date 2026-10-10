"""Write a new FT config bound to a compatible newly trained parent; never trains."""
import argparse
import json
from pathlib import Path


def bind(config_path, parent_run, output):
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    parent_run=Path(parent_run).resolve()
    parent=json.loads((parent_run/'config.json').read_text(encoding='utf-8'))
    weights=parent_run/'model_weights.pt'
    if config.get('loss',{}).get('name')!='constrained_trigger':
        raise ValueError('The target must be a constrained_trigger FT config')
    if not weights.is_file(): raise FileNotFoundError(weights)
    for field in ['seed','features_to_use','model']:
        if config.get(field)!=parent.get(field): raise ValueError(f'Parent mismatch: {field}')
    if not config.get('model') and config.get('hidden_layers',[32,16])!=parent.get('hidden_layers',[32,16]):
        raise ValueError('Parent mismatch: hidden_layers')
    if parent.get('max_events_per_class') is not None:
        raise ValueError('A smoke-test parent cannot initialize a full-data reproduction')
    config['initialization']={'mode':'pretrained','weights_path':weights.as_posix()}
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(config,f,indent=2); f.write('\n')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--parent-run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(bind(args.config,args.parent_run,args.output))
    print('Use the identical dataset and split as the parent. No training was run.')
