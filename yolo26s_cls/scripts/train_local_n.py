"""Run the full local YOLO26n E0 baseline and then evaluate the validation set."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import traceback

import torch
import wafer_colab as wc

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'yolo26s_cls/runs/yolo26n_e0_seed42'
STATUS = ROOT / 'yolo26s_cls/reports/local_n_status.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    start = datetime.now().astimezone().isoformat()
    state = dict(state='starting', pid=os.getpid(), started=start, model='yolo26n-cls.pt',
                 run=str(RUN), epochs=100, batch=32, test_evaluated=False)
    wc.dump(STATUS, state)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        os.chdir(ROOT / 'yolo26s_cls/weights')
        data = ROOT / 'data/processed/wafer38_v1'
        if not args.resume:
            wc.audit(data)
        state.update(state='training', environment=wc.environment())
        wc.dump(STATUS, state)
        wc.run_experiment(data, RUN, 'E0', model='yolo26n-cls.pt', resume=args.resume,
                          epochs=100, batch=32, workers=0, device=0)
        state['state'] = 'evaluating_validation'
        wc.dump(STATUS, state)
        metrics = wc.evaluate(RUN / 'weights/best_macro.pt', data, 'val', output=RUN / 'evaluation')
        wc.dump(RUN / 'evaluation/speed.json', wc.benchmark(RUN / 'weights/best_macro.pt', data))
        state.update(state='completed', finished=datetime.now().astimezone().isoformat(),
                     validation_accuracy=metrics['accuracy'], validation_macro_f1=metrics['macro_f1'],
                     selection=json.loads((RUN / 'selection.json').read_text(encoding='utf-8')))
        wc.dump(STATUS, state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    except BaseException:
        state.update(state='failed', error=traceback.format_exc(), finished=datetime.now().astimezone().isoformat())
        wc.dump(STATUS, state)
        raise


if __name__ == '__main__':
    main()
