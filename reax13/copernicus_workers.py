# SPDX-License-Identifier: MIT
"""BF16 native workers for Copernicus-G and Copernicus-Q."""
import json
import math
import sys
import traceback
from pathlib import Path


class InputLimitError(ValueError):
    """The rendered prompt exceeds the explicitly declared per-model token limit."""


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

    execution_stream = None
    class Backend:
        def warm(self, *args):
            pass

        def score(self, text, media, ids):
            if media:
                raise ValueError('Only text inputs are supported')
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) > 32768:
                raise InputLimitError('Gemma rendered prompt exceeds 32768 tokens')
            # KnowLine executes multi-question scoring in its own thread pool;
            # CUDA's current stream is thread-local, so propagate the request stream.
            with torch.cuda.stream(execution_stream), torch.inference_mode():
                inputs = torch.tensor([token_ids], device=device)
                logits = model(input_ids=inputs, logits_to_keep=1, use_cache=False).logits[0, -1].float().log_softmax(-1)
                return [float(logits[i]) for i in ids], int(inputs.shape[1])

    temperatures = Path(model_dir)/'temperatures.json'
    if not temperatures.exists():
        temperatures = root/'third_party'/'knowline'/'temperatures.json'
    runner = ks.KnowLine(tokenizer, Backend(), 1.0, 1, json.loads(temperatures.read_text()), Layout(read_twice=True, twice_max_tokens=3000))
    def decide(state, questions):
        nonlocal execution_stream
        execution_stream=torch.cuda.current_stream(device=device)
        try:
            answers, _ = runner.run(state, questions)
            return {'answers': answers}
        finally:
            execution_stream=None
    return decide


def load_qwen(model_dir, device, precision, batch_size=8, batch_tokens=4096):
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
    if batch_size < 1 or batch_tokens < 1:
        raise ValueError('Qwen batch limits must be positive')
    def decide(state, questions):
        import torch
        answers = {}; plans=[]
        for qid, question in questions.items():
            row = {'state':state, 'question':question}
            rendered = P.render(state, question)
            text = runner.tokenizer.apply_chat_template(
                [{'role':'system','content':rendered.system}, {'role':'user','content':rendered.user}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = runner.tokenizer.encode(text, add_special_tokens=False)
            if not ids or len(ids) > runner.max_length:
                raise InputLimitError('Qwen rendered prompt exceeds 32768 tokens or is empty')
            plans.append((len(ids), qid, row, rendered, ids))
        plans.sort(key=lambda plan: plan[0])
        while plans:
            length=plans[0][0]
            limit=min(batch_size,max(1,batch_tokens//length))
            group=[]
            while plans and plans[0][0]==length and len(group)<limit:
                group.append(plans.pop(0))
            input_ids = torch.tensor([plan[4] for plan in group], device=runner.weight.device)
            with torch.inference_mode():
                hidden = runner.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).last_hidden_state
                logits = torch.nn.functional.linear(hidden[:, -1], runner.weight).float().cpu().tolist()
            for (_, qid, row, rendered, _), values in zip(group, logits):
                values = values[:len(rendered.letters)]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError(f'Qwen returned non-finite logits for {qid!r}')
                dist = P.answer_probs(row, rendered, values)
                kind = row['question']['type']
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
        conn.send({'status':'ready', 'gpu':torch.cuda.get_device_name(device), 'torch':torch.__version__, 'allocated_bytes':torch.cuda.memory_allocated(device), 'reserved_bytes':torch.cuda.memory_reserved(device)})
        while True:
            request = conn.recv()
            if request is None:
                return
            try:
                conn.send({'status':'ok', 'response':decide(request['state'], request['questions'])})
            except Exception as exc:
                conn.send({'status':'error', 'exception':type(exc).__name__, 'error':f'{type(exc).__name__}: {exc}'})
    except BaseException as exc:
        try:
            conn.send({'status':'fatal', 'error':f'{type(exc).__name__}: {exc}', 'traceback':traceback.format_exc()})
        except Exception:
            pass
    finally:
        conn.close()
