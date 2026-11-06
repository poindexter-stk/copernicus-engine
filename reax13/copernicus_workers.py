# SPDX-License-Identifier: MIT
"""BF16 native workers for Copernicus-G and Copernicus-Q."""
import json
import math
import sys
import traceback
from pathlib import Path


FP8_EXCLUDE = ['lm_head','visual','model.visual','vision_tower','model.vision_tower','multi_modal_projector','model.multi_modal_projector','audio_tower','model.audio_tower','embed_vision','embed_audio','in_proj_a','in_proj_b']


def fp8_config():
    from transformers import FineGrainedFP8Config
    return FineGrainedFP8Config(weight_block_size=(128,128), modules_to_not_convert=FP8_EXCLUDE)


def load_gemma(model_dir, device, precision):
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root/'third_party'/'knowline'))
    import knowline_server as ks
    from knowline_layout import Layout
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    opts = {'quantization_config': fp8_config()} if precision == 'fp8' else {}
    model = Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={'':device}, local_files_only=True, **opts).eval()
    if precision == 'fp8' and not any('float8' in str(p.dtype) for p in model.parameters()):
        raise RuntimeError('Gemma weights were not quantized to FP8')

    class Backend:
        def warm(self, *args):
            pass

        def score(self, text, media, ids):
            if media:
                raise ValueError('Only text inputs are supported')
            inputs = torch.tensor([tokenizer.encode(text, add_special_tokens=False)], device=device)
            with torch.inference_mode():
                logits = model(input_ids=inputs, logits_to_keep=1, use_cache=False).logits[0, -1].float().log_softmax(-1)
            return [float(logits[i]) for i in ids], int(inputs.shape[1])

    temperatures = Path(model_dir)/'temperatures.json'
    if not temperatures.exists():
        temperatures = root/'third_party'/'knowline'/'temperatures.json'
    runner = ks.KnowLine(tokenizer, Backend(), 1.0, 1, json.loads(temperatures.read_text()), Layout(read_twice=True, twice_max_tokens=3000))
    def decide(state, questions):
        answers, _ = runner.run(state, questions)
        return {'answers': answers}
    return decide


def load_qwen(model_dir, device, precision):
    from reax_subnet.spec import prompt as P
    from reax_subnet.spec.joint import JointRunner
    codes = json.loads((Path(model_dir)/'decision_config.json').read_text())['codes']

    def render(state, question):
        kind = question['type']
        criteria = question.get('criteria')
        if kind == 'choice':
            keys, values = list(criteria), list(criteria.values())
        elif kind == 'noul':
            criteria = criteria or {}
            keys, values = ['false', 'true'], [criteria.get('false', 'No'), criteria.get('true', 'Yes')]
        elif kind == 'score':
            keys, values = [str(i) for i in range(len(criteria))], list(criteria)
        else:
            raise ValueError('Unsupported question type')
        if not 2 <= len(keys) <= len(codes):
            raise ValueError('Unsupported option count')
        labels = tuple(codes[:len(keys)])
        text = 'SHARED STATE (JSON string):\n'+json.dumps(state, ensure_ascii=False)+'\n\nQUESTION:\n'
        text += P._text(question.get('instructions', ''))+'\nOPTIONS:\n'
        text += '\n'.join(f'{label}: {P._text(value)}' for label, value in zip(labels, values))
        text += '\nAnswer with one option letter only.'
        return P.Rendered(P.SYSTEM, text, labels, tuple(map(str, keys)))

    P.render = render
    if precision == 'fp8':
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
        original = Qwen3_5Model.from_pretrained
        def quantized(cls, *args, **kwargs):
            kwargs['quantization_config'] = fp8_config()
            return original(*args, **kwargs)
        Qwen3_5Model.from_pretrained = classmethod(quantized)
    runner = JointRunner.load(str(model_dir), device=device)
    if precision == 'fp8' and not any('float8' in str(p.dtype) for p in runner.model.parameters()):
        raise RuntimeError('Qwen weights were not quantized to FP8')
    runner.max_length = 32768
    def decide(state, questions):
        pairs = list(questions.items())
        probs = runner.decide([{'state':state, 'question':q} for _, q in pairs])
        answers = {}
        for (qid, question), dist in zip(pairs, probs):
            if dist is None:
                raise ValueError(f'Qwen could not score question {qid!r} (including context or GPU errors)')
            kind = question['type']
            if kind == 'noul':
                answers[qid] = {'type':kind, 'noul':dist['yes']}
            else:
                answers[qid] = {'type':kind, 'probabilities':dist}
        return {'answers':answers}
    return decide


def worker(conn, kind, model_dir, device, precision='bf16'):
    try:
        import torch
        torch.set_num_threads(4)
        if precision == 'fp8':
            import fast_fp8_gemm
            fast_fp8_gemm.install()
        decide = (load_gemma if kind == 'g' else load_qwen)(model_dir, device, precision)
        conn.send({'status':'ready', 'allocated_bytes':torch.cuda.memory_allocated(), 'reserved_bytes':torch.cuda.memory_reserved()})
        while True:
            request = conn.recv()
            if request is None:
                return
            try:
                conn.send({'status':'ok', 'response':decide(request['state'], request['questions'])})
            except Exception as exc:
                conn.send({'status':'error', 'error':f'{type(exc).__name__}: {exc}'})
    except BaseException as exc:
        try:
            conn.send({'status':'fatal', 'error':f'{type(exc).__name__}: {exc}', 'traceback':traceback.format_exc()})
        except Exception:
            pass
    finally:
        conn.close()
