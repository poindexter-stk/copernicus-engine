# SPDX-License-Identifier: MIT
"""Single-process Copernicus variant of the REAX 1.3 engine contract."""
import gc
import json
import os
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from copernicus_workers import InputLimitError, load_gemma, load_qwen
from probabilities import mix

try:
    from decision_index.engines import Engine, Unsupported
except ImportError:
    class Engine:
        def __init__(self, **options): pass
    class Unsupported(ValueError): pass


class CopernicusEngine(Engine):
    name = 'copernicus-57b'
    latency = 'One request at a time; Gemma and Qwen run concurrently on CUDA streams; Qwen batches equal-length questions; includes in-process work and excludes startup and warmup.'

    def __init__(self, config=None, model=None, **options):
        super().__init__(**options)
        self.config = json.loads(Path(config or Path(__file__).with_name('copernicus57.json')).read_text())
        self.config.update(options)
        root = Path(model or self.config['model_root']).resolve()
        paths = [root/'part-g', root/'part-q']
        if any(not p.is_dir() for p in paths):raise FileNotFoundError('Model root must contain part-g and part-q')
        precision = self.config.get('precision', 'fp8')
        devices = self.config.get('devices', ['cuda:0','cuda:0'])
        if precision not in ('fp8','bf16') or len(devices) != 2:raise ValueError('Invalid precision or devices')
        import torch
        import transformers
        torch.set_num_threads(4)
        if precision == 'fp8':
            import fast_fp8_gemm
            from autotune_guard import install as install_autotune_guard
            install_autotune_guard()
            fast_fp8_gemm.install()
        self.lock = threading.Lock()
        self._alive = False
        self.gemma = self.qwen = None
        try:
            self.gemma = load_gemma(str(paths[0]), devices[0], precision)
            self.qwen = load_qwen(str(paths[1]), devices[1], precision,
                                  batch_size=int(self.config.get('qwen_batch_size', 8)),
                                  batch_tokens=int(self.config.get('qwen_batch_tokens', 4096)))
            self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='copernicus')
            self.streams = [torch.cuda.Stream(device=d) for d in devices]
            for stream in self.streams:
                stream.wait_stream(torch.cuda.current_stream(device=stream.device))
            self._alive = True
        except BaseException:
            self.close()
            raise
        self.provenance = {'model':str(root), 'model_repo':'Poindexter-Labs/Copernicus-57B', 'model_revision':'28a57bbc72a752a60acc8c4af7c9071b76d71e35', 'engine_repo':'https://github.com/poindexter-stk/copernicus-engine', 'engine_commit':os.environ.get('COPERNICUS_ENGINE_COMMIT','unspecified'), 'qwen_reader_revision':'0c1efcc7abad11c56eefc866334aab4b01ec38ec', 'precision':precision, 'devices':devices, 'mix':'0.5 Gemma + 0.5 Qwen probabilities', 'execution':'one request at a time; Gemma and Qwen concurrent CUDA streams; Qwen equal-length question batching', 'qwen_question_batch_size':int(self.config.get('qwen_batch_size', 8)), 'qwen_batch_tokens':int(self.config.get('qwen_batch_tokens', 4096)), 'gemma_question_workers':1, 'concurrent_proxy_limit':int(self.config.get('concurrent_proxy_limit', 16000)), 'max_input_proxy_tokens':int(self.config['max_input_proxy_tokens']), 'max_rendered_tokens_per_model':32768}
        self.ready = {'mode':'copernicus-57b', 'precision':precision, 'devices':devices, 'gpu':torch.cuda.get_device_name(devices[0]), 'torch':torch.__version__, 'transformers':transformers.__version__, 'allocated_bytes':torch.cuda.memory_allocated(devices[0]), 'reserved_bytes':torch.cuda.memory_reserved(devices[0])}

    def alive(self):return self._alive

    def __call__(self, state, questions):
        if not questions or len(questions)>64:raise Unsupported('Supports 1..64 questions')
        for q in questions.values():
            typ=q.get('type');criteria=q.get('criteria')
            if typ == 'choice' and (not isinstance(criteria,dict) or not 2<=len(criteria)<=255):raise Unsupported('choice requires 2..255 options')
            if typ == 'score' and (not isinstance(criteria,list) or not 2<=len(criteria)<=10):raise Unsupported('score requires 2..10 options')
            if typ not in ('choice','score','noul'):raise Unsupported('Unsupported question type')
        size=len(json.dumps({'model':'jev-1.13.0','state':state,'questions':questions},ensure_ascii=False,separators=(',',':'))) // 4
        if size > int(self.config['max_input_proxy_tokens']):raise Unsupported('Input exceeds proxy token limit')
        with self.lock:
            if not self.alive():raise RuntimeError('Engine is closed')
            request={'state':state,'questions':questions};start=time.monotonic()
            try:
                import torch
                def invoke(fn, stream):
                    with torch.cuda.stream(stream):
                        return fn(state, questions)
                if size > int(self.config.get('concurrent_proxy_limit', 16000)):
                    # Long requests run the same model paths serially to bound activation memory.
                    results=[invoke(self.gemma, self.streams[0]), invoke(self.qwen, self.streams[1])]
                    errors=[]
                else:
                    futures=[self.pool.submit(invoke, self.gemma, self.streams[0]),
                             self.pool.submit(invoke, self.qwen, self.streams[1])]
                    results=[]; errors=[]
                    for future in futures:
                        try: results.append(future.result()); errors.append(None)
                        except BaseException as exc: results.append(None); errors.append(exc)
                for device in set(self.config.get('devices', ['cuda:0', 'cuda:0'])):
                    torch.cuda.synchronize(device)
                if any(errors): raise next(e for e in errors if e is not None)
                gemma,qwen=results
            except InputLimitError as exc:
                raise Unsupported(str(exc)) from exc
            result=mix(request,gemma,qwen);result['model']='copernicus-57b'
            return result, {'elapsed_ms':1000*(time.monotonic()-start),'precision':self.config.get('precision','fp8'),'devices':self.config.get('devices',['cuda:0','cuda:0'])}

    def warmup(self):
        self('', {'warmup':{'type':'choice','instructions':'Choose one.', 'criteria':{'a':'A','b':'B'}}})

    def runtime(self):return self.ready

    def close(self):
        self._alive=False
        pool=getattr(self,'pool',None)
        if pool is not None:
            pool.shutdown(wait=True)
            self.pool=None
        self.gemma=self.qwen=None
        gc.collect()
