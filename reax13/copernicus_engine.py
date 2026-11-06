# SPDX-License-Identifier: MIT
"""Two-GPU BF16 Copernicus variant of the REAX 1.3 engine contract."""
import atexit
import json
import multiprocessing as mp
import threading
import time
from pathlib import Path

from copernicus_workers import worker
from probabilities import mix

try:
    from decision_index.engines import Engine, Unsupported
except ImportError:
    class Engine:
        def __init__(self, **options): pass
    class Unsupported(ValueError): pass


class CopernicusEngine(Engine):
    name = 'copernicus-57b'
    def __init__(self, config=None, model=None, **options):
        super().__init__(**options)
        self.config = json.loads(Path(config or Path(__file__).with_name('copernicus57.json')).read_text())
        self.config.update(options)
        root = Path(model or self.config['model_root']).resolve()
        paths = [root/'part-g', root/'part-q']
        if any(not p.is_dir() for p in paths):
            raise FileNotFoundError('Model root must contain part-g and part-q')
        self.lock = threading.Lock()
        self.conns, self.processes = [], []
        ctx = mp.get_context('spawn')
        precision = self.config.get('precision', 'fp8')
        devices = self.config.get('devices', ['cuda:0','cuda:0'])
        if precision not in ('fp8','bf16') or len(devices) != 2:
            raise ValueError('Invalid precision or devices')
        worker_info = []
        try:
            for i, (kind, path) in enumerate(zip(('g', 'q'), paths)):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=worker, args=(child, kind, str(path), devices[i], precision), daemon=True)
                process.start();child.close()
                self.conns.append(parent);self.processes.append(process)
                if not parent.poll(float(self.config['startup_timeout'])):
                    raise TimeoutError('Model worker startup timed out')
                reply = parent.recv()
                if reply.get('status') != 'ready': raise RuntimeError(str(reply))
                worker_info.append(reply)
        except BaseException:
            self.close()
            raise
        self.ready = {'mode':'copernicus-57b', 'precision':precision, 'devices':devices, 'workers':worker_info, 'mix':'0.5 Gemma + 0.5 Qwen probabilities'}
        atexit.register(self.close)

    def alive(self):
        return len(self.processes) == 2 and all(p.is_alive() for p in self.processes)

    def __call__(self, state, questions):
        if not questions or len(questions)>64:
            raise Unsupported('Supports 1..64 questions')
        for q in questions.values():
            typ=q.get('type'); criteria=q.get('criteria')
            if typ == 'choice' and (not isinstance(criteria,dict) or not 2<=len(criteria)<=255):raise Unsupported('choice requires 2..255 options')
            if typ == 'score' and (not isinstance(criteria,list) or not 2<=len(criteria)<=10):raise Unsupported('score requires 2..10 options')
            if typ not in ('choice','score','noul'):raise Unsupported('Unsupported question type')
        size = len(json.dumps({'model':'jev-1.13.0','state':state,'questions':questions}, ensure_ascii=False,separators=(',',':'))) // 4
        if size > int(self.config['max_input_proxy_tokens']): raise Unsupported('Input exceeds proxy token limit')
        with self.lock:
            if not self.alive():raise RuntimeError('A model worker has stopped')
            request={'state':state,'questions':questions};start=time.monotonic()
            for conn in self.conns:conn.send(request)
            responses=[];deadline=start+float(self.config['request_timeout'])
            for conn in self.conns:
                if not conn.poll(max(0,deadline-time.monotonic())):
                    self.close();raise TimeoutError('Model worker timed out; restart required')
                reply=conn.recv()
                if reply.get('status') != 'ok':
                    self.close();raise RuntimeError(str(reply))
                responses.append(reply['response'])
            result=mix(request,*responses);result['model']='copernicus-57b'
            return result, {'elapsed_ms':1000*(time.monotonic()-start), 'precision':self.config.get('precision','fp8'), 'devices':self.config.get('devices',['cuda:0','cuda:0'])}

    def warmup(self):
        self('', {'warmup':{'type':'choice','instructions':'Choose one.', 'criteria':{'a':'A','b':'B'}}})

    def runtime(self): return self.ready

    def close(self):
        for conn in getattr(self,'conns',[]):
            try:conn.send(None)
            except (OSError,EOFError,BrokenPipeError):pass
        for process in getattr(self,'processes',[]):
            process.join(timeout=1)
            if process.is_alive():process.terminate();process.join(timeout=5)
        for conn in getattr(self,'conns',[]):conn.close()
        self.processes=[];self.conns=[]
