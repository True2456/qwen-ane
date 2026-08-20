#!/usr/bin/env python3
"""Persistent OpenAI-compatible server for the framework-free pure ANE runtime.

The process bakes the model once, retains every ANE program, and resets only
per-sequence GDN/KV state between requests. Model arithmetic never falls back
to MLX, Metal, Core ML, PyTorch, or the CPU; optional token sampling is a small
CPU post-processing step over the ANE-produced logits.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

_TOOLS=Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0,str(_TOOLS))
from pure_ane import (  # noqa: E402
    Checkpoint, PureAneRuntime, PureAneSnapshot, StandaloneTokenizer,
    assert_standalone,
)


def _content_text(content:Any)->str:
    if content is None:return ""
    if isinstance(content,str):return content
    if isinstance(content,list):
        parts=[]
        for item in content:
            if isinstance(item,dict) and item.get("type","text")=="text":
                parts.append(str(item.get("text","")))
            else:
                raise ValueError("pure ANE server currently accepts text content only")
        return "".join(parts)
    raise ValueError("message content must be a string or text-content list")


def _tool_functions(tools:Any,choice:Any="auto")->tuple[list[dict[str,Any]],bool]:
    if choice=="none":return [],False
    if tools is None:
        if choice in (None,"auto"):return [],False
        raise ValueError("tool_choice requires at least one tool")
    if not isinstance(tools,list):raise ValueError("tools must be a list")
    functions=[]
    for tool in tools:
        if not isinstance(tool,dict) or tool.get("type","function")!="function":
            raise ValueError("only function tools are supported")
        fn=tool.get("function")
        if not isinstance(fn,dict) or not isinstance(fn.get("name"),str):
            raise ValueError("every function tool requires a name")
        functions.append(fn)
    required=choice=="required"
    if isinstance(choice,dict):
        wanted=(choice.get("function") or {}).get("name")
        selected=[fn for fn in functions if fn["name"]==wanted]
        if not selected:raise ValueError(f"tool_choice names unknown function {wanted!r}")
        functions=selected;required=True
    elif choice not in (None,"auto","required","none"):
        raise ValueError("tool_choice must be auto, required, none, or a named function")
    if required and not functions:raise ValueError("tool_choice requires at least one tool")
    return functions,required


def _tool_xml(call:dict[str,Any])->str:
    fn=call.get("function") or call
    name=fn.get("name")
    if not isinstance(name,str):raise ValueError("assistant tool call requires a function name")
    args=fn.get("arguments",{})
    if isinstance(args,str):
        try:args=json.loads(args or "{}")
        except json.JSONDecodeError:args={}
    if not isinstance(args,dict):raise ValueError("tool-call arguments must be an object")
    body=[f"<tool_call>\n<function={name}>\n"]
    for key,value in args.items():
        rendered=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,separators=(",",":"))
        body.append(f"<parameter={key}>\n{rendered}\n</parameter>\n")
    body.append("</function>\n</tool_call>")
    return "".join(body)


_REASONING_INSTRUCTIONS={
    "xhigh":("Reasoning effort is set to xhigh. Please think carefully through "
             "the task, validate key assumptions, consider plausible alternatives, "
             "and prioritize correctness, consistency, and clarity in the final answer."),
    "medium":"",
    "low":("Reasoning effort is set to low. Keep your thinking brief and focused, "
           "moving directly to the conclusion without unnecessary elaboration."),
}


def chat_prompt(messages:list[dict[str,Any]],enable_thinking:bool=True,
                reasoning_effort:str="xhigh",
                tools:list[dict[str,Any]]|None=None,
                tool_required:bool=False)->str:
    """Apply the checkpoint's text-only Qwen chat framing without Transformers."""
    if not messages:raise ValueError("messages must not be empty")
    if enable_thinking and reasoning_effort not in _REASONING_INSTRUCTIONS:
        raise ValueError(
            f"unexpected reasoning_effort {reasoning_effort!r}; supported values "
            "are xhigh (default), medium, and low"
        )
    reasoning_instruction=(_REASONING_INSTRUCTIONS[reasoning_effort]
                           if enable_thinking else "")
    out=[];start=0
    if tools:
        system_content=""
        if messages[0].get("role")=="system":
            system_content=_content_text(messages[0].get("content"));start=1
        block=[]
        if reasoning_instruction:block.append(reasoning_instruction+"\n\n")
        block.append("# Tools\n\nYou have access to the following functions:\n\n<tools>")
        for tool in tools:
            block.append("\n"+json.dumps(tool,ensure_ascii=False,separators=(",",":")))
        block.append("\n</tools>\n\nIf you choose to call a function ONLY reply in the "
                     "following format with NO suffix:\n\n<tool_call>\n"
                     "<function=example_function_name>\n"
                     "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
                     "</function>\n</tool_call>\n\nReminder: required parameters "
                     "must be present, and the function block must be nested inside "
                     "the tool_call tags.")
        if tool_required:block.append(" You must call one of the supplied functions.")
        if system_content:block.append("\n\n"+system_content)
        out.append("<|im_start|>system\n"+"".join(block)+"<|im_end|>\n")
    elif messages[0].get("role")=="system":
        system_content=_content_text(messages[0].get("content")).strip();start=1
        combined=((reasoning_instruction+"\n\n") if reasoning_instruction else "")+system_content
        if combined:out.append("<|im_start|>system\n"+combined+"<|im_end|>\n")
    elif reasoning_instruction:
        out.append("<|im_start|>system\n"+reasoning_instruction+"<|im_end|>\n")
    for index,message in enumerate(messages[start:],start=start):
        role=str(message.get("role",""))
        content=_content_text(message.get("content"))
        if role=="system":
            if index!=0:raise ValueError("system message must be first")
            out.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        elif role=="user":
            out.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role=="assistant":
            calls=message.get("tool_calls") or []
            rendered="\n".join(_tool_xml(call) for call in calls)
            joiner="\n\n" if content and rendered else ""
            reasoning_content=message.get("reasoning_content","")
            reasoning_content=(reasoning_content.strip()
                               if isinstance(reasoning_content,str) else "")
            # The checkpoint preserves historical thinking by default even if
            # the new turn disables thinking.
            thinking_prefix=f"<think>\n{reasoning_content}\n</think>\n\n"
            out.append(f"<|im_start|>assistant\n{thinking_prefix}{content}{joiner}{rendered}<|im_end|>\n")
        elif role=="tool":
            out.append("<|im_start|>user\n<tool_response>\n"+content+
                       "\n</tool_response><|im_end|>\n")
        else:
            raise ValueError(f"unsupported message role {role!r}")
    out.append("<|im_start|>assistant\n")
    out.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
    return "".join(out)


_TOOL_CALL_RE=re.compile(
    r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>",re.S
)
_TOOL_PARAM_RE=re.compile(r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>",re.S)


def split_tool_calls(text:str)->tuple[str,list[dict[str,Any]]]:
    calls=[]
    for match in _TOOL_CALL_RE.finditer(text):
        args={}
        for name,raw in _TOOL_PARAM_RE.findall(match.group(2)):
            raw=raw.strip()
            try:args[name]=json.loads(raw)
            except (json.JSONDecodeError,TypeError):args[name]=raw
        calls.append({"id":"call_"+uuid.uuid4().hex[:24],"type":"function",
                      "function":{"name":match.group(1),
                                  "arguments":json.dumps(args,ensure_ascii=False,separators=(",",":"))}})
    return _TOOL_CALL_RE.sub("",text).strip(),calls


def split_reasoning(text:str,enable_thinking:bool)->tuple[str,str]:
    """Split Qwen output after the opening ``<think>`` supplied in the prompt."""
    if not enable_thinking:return "",text.strip()
    marker="</think>";end=text.find(marker)
    if end<0:
        # A length-limited response that never closed the thinking block has
        # produced reasoning only; do not mislabel it as a final answer.
        return text.strip(),""
    return text[:end].strip(),text[end+len(marker):].strip()


class ReasoningStream:
    """Route live Qwen text to reasoning/content deltas across a split marker."""
    marker="</think>"
    def __init__(self,enabled:bool,reasoning:Callable[[str],None],
                 content:Callable[[str],None]):
        self.enabled=enabled;self.send_reasoning=reasoning
        self.send_content=content;self.pending="";self.in_reasoning=enabled
        self.trim_content_prefix=enabled

    def _content(self,delta:str)->None:
        if self.trim_content_prefix:
            delta=delta.lstrip("\r\n")
            if not delta:return
            self.trim_content_prefix=False
        self.send_content(delta)

    def feed(self,delta:str)->None:
        if not self.enabled:self.send_content(delta);return
        self.pending+=delta
        if not self.in_reasoning:
            self._content(self.pending);self.pending="";return
        end=self.pending.find(self.marker)
        if end>=0:
            if end:self.send_reasoning(self.pending[:end])
            self.pending=self.pending[end+len(self.marker):]
            self.in_reasoning=False
            if self.pending:self._content(self.pending)
            self.pending="";return
        safe=max(0,len(self.pending)-len(self.marker)+1)
        if safe:
            self.send_reasoning(self.pending[:safe]);self.pending=self.pending[safe:]

    def finish(self)->None:
        if not self.pending:return
        (self.send_reasoning if self.in_reasoning else self._content)(self.pending)
        self.pending=""


def _sampling_selector(prompt_ids:list[int],temperature:float,top_p:float,
                       top_k:int,repetition_penalty:float,seed:int
                       )->Callable[[np.ndarray,list[int]],int] | None:
    if temperature<=0:return None
    if not 0<top_p<=1:raise ValueError("top_p must be in (0, 1]")
    if top_k<0:raise ValueError("top_k must be non-negative")
    if repetition_penalty<=0:raise ValueError("repetition_penalty must be positive")
    rng=np.random.default_rng(seed);prompt_set=set(prompt_ids)
    def select(raw:np.ndarray,generated:list[int])->int:
        scores=np.asarray(raw,dtype=np.float32).copy()
        if repetition_penalty!=1:
            for token in prompt_set.union(generated):
                scores[token]=(scores[token]*repetition_penalty
                               if scores[token]<0 else scores[token]/repetition_penalty)
        scores/=temperature
        if top_k and top_k<len(scores):
            keep=np.argpartition(scores,-top_k)[-top_k:];chosen=scores[keep]
        else:
            keep=np.arange(len(scores));chosen=scores
        if top_p<1:
            order=np.argsort(chosen)[::-1];ordered=chosen[order]
            probs=np.exp(ordered-float(ordered.max()));probs/=float(probs.sum())
            count=int(np.searchsorted(np.cumsum(probs),top_p,side="left"))+1
            keep=keep[order[:count]];chosen=ordered[:count]
        probs=np.exp(chosen-float(chosen.max()));probs/=float(probs.sum())
        return int(rng.choice(keep,p=probs))
    return select


class TextEmitter:
    """Incrementally decode token IDs while withholding possible stop suffixes."""
    def __init__(self,tokenizer:StandaloneTokenizer,stops:list[str],
                 emit:Callable[[str],None] | None):
        self.tokenizer=tokenizer;self.stops=[x for x in stops if x]
        self.emit=emit;self.ids=[];self.sent=0;self.decoded="";self.stopped=False
        self.hold=max((len(x) for x in self.stops),default=0)

    def token(self,token_id:int)->bool:
        self.ids.append(token_id);self.decoded=self.tokenizer.decode(self.ids)
        ends=[self.decoded.find(stop) for stop in self.stops]
        ends=[x for x in ends if x>=0]
        if ends:
            visible=self.decoded[:min(ends)];self.stopped=True
            self._send(visible);return False
        safe=len(self.decoded) if not self.hold else max(self.sent,len(self.decoded)-self.hold+1)
        self._send(self.decoded[:safe]);return True

    def finish(self)->str:
        if not self.stopped:self._send(self.decoded)
        if self.stopped:
            ends=[self.decoded.find(stop) for stop in self.stops]
            ends=[x for x in ends if x>=0]
            return self.decoded[:min(ends)]
        return self.decoded

    def _send(self,visible:str)->None:
        if len(visible)<=self.sent:return
        delta=visible[self.sent:]
        if self.emit:self.emit(delta)
        self.sent=len(visible)


@dataclass
class PrefixCacheEntry:
    tokens: tuple[int,...]
    snapshot: PureAneSnapshot
    logits: np.ndarray
    mode: str
    created_at: float


class PureAneService:
    def __init__(self,args:argparse.Namespace):
        self.args=args;self.started=time.time();self.lock=threading.Lock()
        self.stats_lock=threading.Lock();self.requests=0;self.failures=0
        self.total_prompt_tokens=0;self.total_completion_tokens=0
        self.total_inference_seconds=0.0;self.last_result:dict[str,Any]|None=None
        self.prefix_cache:PrefixCacheEntry|None=None
        self.prefix_hits=0;self.prefix_misses=0;self.prefix_tokens_reused=0
        self.checkpoint=Checkpoint(args.model);self.tokenizer=StandaloneTokenizer(args.model)
        eos_id=self.tokenizer.impl.token_to_id("<|im_end|>")
        eos={eos_id} if eos_id is not None else set()
        configured=self.checkpoint.config.get("text_config",self.checkpoint.config).get("eos_token_id")
        if isinstance(configured,int):eos.add(configured)
        elif isinstance(configured,list):eos.update(int(x) for x in configured)
        self.eos_ids=eos
        bake_cache=args.bake_cache
        if bake_cache=="auto":
            bake_cache=str(Path.home()/"Library"/"Caches"/"q38-pure-ane")
        self.runtime=PureAneRuntime(self.checkpoint,args.engine_path,bits=args.bits,
                                    context=args.context,mtp_draft=args.mtp_draft,
                                    bake_cache=bake_cache or None,
                                    profile_decode=args.profile_decode,
                                    down_proj_parts=args.down_proj_parts)
        assert_standalone("persistent server ready")

    def model_info(self)->dict[str,Any]:
        return {"id":self.args.name,"object":"model","owned_by":"pure-ane",
                "backend":"AppleNeuralEngine.framework","pure_ane":True,
                "bits":self.args.bits,"context_length":self.args.context,
                "mtp_draft":self.args.mtp_draft,
                "down_proj_parts":self.args.down_proj_parts}

    def metrics(self)->dict[str,Any]:
        with self.stats_lock:
            return {"ready":True,"busy":self.lock.locked(),
                    "uptime_seconds":time.time()-self.started,
                    "requests":self.requests,"failures":self.failures,
                    "prompt_tokens":self.total_prompt_tokens,
                    "completion_tokens":self.total_completion_tokens,
                    "inference_seconds":self.total_inference_seconds,
                    "loaded_programs":self.runtime.program_count,
                    "compiled_blob_gb":self.runtime.blob_bytes/1e9,
                    "kv_capacity_gb":self.runtime.kv_cache_bytes/1e9,
                    "startup":self.runtime.startup_metrics,
                    "decode_profile":self.runtime.profile_snapshot(),
                    "prefix_cache":{"hits":self.prefix_hits,
                                    "misses":self.prefix_misses,
                                    "tokens_reused":self.prefix_tokens_reused,
                                    "cached_tokens":(len(self.prefix_cache.tokens)
                                                     if self.prefix_cache else 0),
                                    "mode":(self.prefix_cache.mode
                                            if self.prefix_cache else None)},
                    "last_result":self.last_result,"model":self.model_info()}

    def validate(self,prompt:str,max_tokens:int,temperature:float=0.0,
                 top_p:float=1.0,top_k:int=0,repetition_penalty:float=1.0,
                 **_:Any)->list[int]:
        prompt_ids=self.tokenizer.encode(prompt)
        if not prompt_ids:raise ValueError("prompt tokenized to nothing")
        if max_tokens<1:raise ValueError("max_tokens must be at least 1")
        if len(prompt_ids)+max_tokens>self.args.context:
            raise ValueError(f"prompt plus generation ({len(prompt_ids)+max_tokens}) exceeds configured context {self.args.context}")
        if temperature<0:raise ValueError("temperature must be non-negative")
        if not 0<top_p<=1:raise ValueError("top_p must be in (0, 1]")
        if top_k<0:raise ValueError("top_k must be non-negative")
        if repetition_penalty<=0:raise ValueError("repetition_penalty must be positive")
        return prompt_ids

    def generate(self,prompt:str,max_tokens:int,temperature:float=0.0,
                 top_p:float=1.0,top_k:int=0,repetition_penalty:float=1.0,
                 seed:int=0,stops:list[str]|None=None,
                 emit:Callable[[str],None] | None=None,
                 prefix_cache:bool=True)->dict[str,Any]:
        prompt_ids=self.validate(prompt,max_tokens,temperature,top_p,top_k,
                                 repetition_penalty)
        selector=_sampling_selector(prompt_ids,temperature,top_p,top_k,
                                    repetition_penalty,seed)
        emitter=TextEmitter(self.tokenizer,stops or [],emit)
        queued=time.perf_counter();first=[None]
        def token(token_id:int)->bool:
            if first[0] is None:first[0]=time.perf_counter()
            return emitter.token(token_id)
        with self.lock:
            entered=time.perf_counter();start=time.perf_counter()
            self.runtime.begin_profile()
            mode=("mtp" if self.runtime.mtp is not None and selector is None
                  else "target")
            cache=self.prefix_cache;prefilled=0;cached_logits=None;cache_hit=False
            if (prefix_cache and cache is not None and cache.mode==mode and
                    len(cache.tokens)<=len(prompt_ids) and
                    tuple(prompt_ids[:len(cache.tokens)])==cache.tokens):
                self.runtime.restore(cache.snapshot);prefilled=len(cache.tokens)
                cached_logits=cache.logits;cache_hit=True;self.prefix_hits+=1
                self.prefix_tokens_reused+=prefilled
            else:
                self.runtime.reset();self.prefix_cache=None
                if prefix_cache:self.prefix_misses+=1
            def cache_prefill(logits:np.ndarray)->None:
                if prefix_cache:
                    self.prefix_cache=PrefixCacheEntry(
                        tuple(prompt_ids),self.runtime.snapshot(),
                        np.asarray(logits,dtype=np.float16).copy(),mode,time.time()
                    )
            try:
                ids,_runtime_seconds=self.runtime.generate(self.tokenizer,prompt,max_tokens,
                    on_token=token,stop_token_ids=self.eos_ids,token_selector=selector,
                    prefilled_tokens=prefilled,prefill_logits=cached_logits,
                    on_prefill=cache_prefill if prefix_cache else None)
                seconds=time.perf_counter()-start
            except Exception:
                with self.stats_lock:self.failures+=1
                raise
        text=emitter.finish();ttft=((first[0]-start) if first[0] else seconds)
        result={"text":text,"prompt_tokens":len(prompt_ids),
                "completion_tokens":len(ids),"total_tokens":len(prompt_ids)+len(ids),
                "seconds":seconds,"time_to_first_token":ttft,
                "queue_seconds":entered-queued,
                "tokens_per_second":len(ids)/max(seconds,1e-9),
                "decode_tokens_per_second":max(0,len(ids)-1)/max(seconds-ttft,1e-9),
                "finish_reason":"stop" if len(ids)<max_tokens or emitter.stopped else "length",
                "sampled":selector is not None,
                "mtp_used":self.runtime.mtp is not None and selector is None,
                "prefix_cache_hit":cache_hit,"prefix_tokens_reused":prefilled,
                "prompt_tokens_evaluated":len(prompt_ids)-prefilled}
        profile=self.runtime.profile_snapshot()
        if profile is not None:result["profile"]=profile["request"]
        with self.stats_lock:
            self.requests+=1;self.total_prompt_tokens+=len(prompt_ids)
            self.total_completion_tokens+=len(ids);self.total_inference_seconds+=seconds
            self.last_result=result.copy()
        print(f"PURE_ANE_REQUEST prompt={len(prompt_ids)} generated={len(ids)} "
              f"ttft={ttft:.3f}s seconds={seconds:.3f} tok_s={result['tokens_per_second']:.3f} "
              f"queue={result['queue_seconds']:.3f}s cache={'hit' if cache_hit else 'miss'} "
              f"reused={prefilled}",flush=True)
        return result


def _stops(value:Any)->list[str]:
    if value is None:return []
    if isinstance(value,str):return [value]
    if isinstance(value,list) and all(isinstance(x,str) for x in value):return value
    raise ValueError("stop must be a string or list of strings")


def _request_options(req:dict[str,Any],default_tokens:int)->dict[str,Any]:
    return {"max_tokens":int(req.get("max_tokens") or req.get("max_completion_tokens") or default_tokens),
            "temperature":float(req.get("temperature",0.0)),
            "top_p":float(req.get("top_p",1.0)),"top_k":int(req.get("top_k",0)),
            "repetition_penalty":float(req.get("repetition_penalty",1.0)),
            "seed":int(req.get("seed",0)),"stops":_stops(req.get("stop")),
            "prefix_cache":bool(req.get("prefix_cache",True))}


def _thinking_options(req:dict[str,Any],*,default:bool=True)->tuple[bool,str]:
    enabled=req.get("enable_thinking",default)
    if not isinstance(enabled,bool):raise ValueError("enable_thinking must be a boolean")
    effort=req.get("reasoning_effort","xhigh")
    if not isinstance(effort,str):raise ValueError("reasoning_effort must be a string")
    if enabled and effort not in _REASONING_INSTRUCTIONS:
        raise ValueError("reasoning_effort must be one of: low, medium, xhigh")
    return enabled,effort


def build_handler(service:PureAneService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version="HTTP/1.1";server_version="PureANE/0.1"
        def log_message(self,fmt:str,*args:Any)->None:print("  HTTP "+fmt%args,flush=True)
        def _json(self,obj:Any,status:int=200)->None:
            body=json.dumps(obj,separators=(",",":"),allow_nan=False).encode()
            self.send_response(status);self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body)));self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers();self.wfile.write(body)
        def _error(self,message:str,status:int=400)->None:
            self._json({"error":{"message":message,"type":"invalid_request_error"}},status)
        def do_OPTIONS(self)->None:
            self.send_response(204);self.send_header("Access-Control-Allow-Origin","*")
            self.send_header("Access-Control-Allow-Headers","Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods","GET, POST, OPTIONS");self.end_headers()
        def do_GET(self)->None:
            path=self.path.split("?",1)[0].rstrip("/")
            if path in ("","/health","/healthz"):
                self._json({"status":"ok","ready":True,"busy":service.lock.locked(),"model":service.args.name})
            elif path=="/v1/models":self._json({"object":"list","data":[service.model_info()]})
            elif path in ("/metrics","/v1/metrics"):self._json(service.metrics())
            else:self._error("not found",404)
        def do_POST(self)->None:
            try:
                length=int(self.headers.get("Content-Length","0"))
                if length>service.args.max_request_bytes:raise ValueError("request body is too large")
                req=json.loads(self.rfile.read(length) or b"{}")
                path=self.path.split("?",1)[0].rstrip("/")
                if path=="/v1/chat/completions":self._chat(req)
                elif path=="/v1/completions":self._completion(req)
                elif path=="/v1/benchmarks":self._benchmark(req)
                else:self._error("not found",404)
            except (ValueError,TypeError,json.JSONDecodeError) as exc:
                try:self._error(str(exc),400)
                except (BrokenPipeError,ConnectionResetError):pass
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception as exc:
                traceback.print_exc()
                try:self._error(str(exc),500)
                except (BrokenPipeError,ConnectionResetError):pass
        def _sse_start(self)->None:
            self.send_response(200);self.send_header("Content-Type","text/event-stream")
            self.send_header("Cache-Control","no-cache");self.send_header("Connection","close")
            self.send_header("Access-Control-Allow-Origin","*");self.end_headers();self.close_connection=True
        def _sse(self,obj:Any)->None:
            self.wfile.write(b"data: "+json.dumps(obj,separators=(",",":")).encode()+b"\n\n");self.wfile.flush()
        def _chat(self,req:dict[str,Any])->None:
            if req.get("n",1)!=1:raise ValueError("only n=1 is supported")
            enable_thinking,reasoning_effort=_thinking_options(req)
            functions,tool_required=_tool_functions(
                req.get("tools"),req.get("tool_choice","auto")
            )
            prompt=chat_prompt(
                req.get("messages",[]),enable_thinking,reasoning_effort,
                functions,tool_required
            )
            options=_request_options(req,service.args.max_tokens)
            service.validate(prompt,**options)
            cid="chatcmpl-"+uuid.uuid4().hex;stream=bool(req.get("stream",False));created=int(time.time())
            if stream:
                self._sse_start();self._sse({"id":cid,"object":"chat.completion.chunk","created":created,"model":service.args.name,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":None}]})
                def emit_content(delta:str)->None:
                    self._sse({"id":cid,"object":"chat.completion.chunk","created":created,"model":service.args.name,"choices":[{"index":0,"delta":{"content":delta},"finish_reason":None}]})
                def emit_reasoning(delta:str)->None:
                    self._sse({"id":cid,"object":"chat.completion.chunk","created":created,"model":service.args.name,"choices":[{"index":0,"delta":{"reasoning_content":delta},"finish_reason":None}]})
                # Tool XML must be parsed as a complete unit before exposing
                # OpenAI tool_call deltas, so tool-enabled streams buffer one
                # assistant turn. Ordinary text requests remain token-live.
                router=ReasoningStream(enable_thinking,emit_reasoning,emit_content)
                result=service.generate(
                    prompt,emit=None if functions else router.feed,**options
                )
                if not functions:router.finish()
                reasoning,visible=split_reasoning(result["text"],enable_thinking)
                content,calls=(split_tool_calls(visible)
                               if functions else (visible,[]))
                if functions and reasoning:emit_reasoning(reasoning)
                if functions and content:emit_content(content)
                if calls:
                    self._sse({"id":cid,"object":"chat.completion.chunk","created":created,"model":service.args.name,"choices":[{"index":0,"delta":{"tool_calls":[dict(index=i,**call) for i,call in enumerate(calls)]},"finish_reason":None}]})
                finish="tool_calls" if calls else result["finish_reason"]
                self._sse({"id":cid,"object":"chat.completion.chunk","created":created,"model":service.args.name,"choices":[{"index":0,"delta":{},"finish_reason":finish}],"usage":{"prompt_tokens":result["prompt_tokens"],"completion_tokens":result["completion_tokens"],"total_tokens":result["total_tokens"]}})
                self.wfile.write(b"data: [DONE]\n\n");self.wfile.flush();return
            result=service.generate(prompt,**options)
            reasoning,visible=split_reasoning(result["text"],enable_thinking)
            content,calls=(split_tool_calls(visible)
                           if functions else (visible,[]))
            message={"role":"assistant","content":content or (None if calls else "")}
            if enable_thinking:message["reasoning_content"]=reasoning
            if calls:message["tool_calls"]=calls
            finish="tool_calls" if calls else result["finish_reason"]
            metadata={k:v for k,v in result.items() if k not in ("text","prompt_tokens","completion_tokens","total_tokens","finish_reason")}
            metadata.update(enable_thinking=enable_thinking,
                            reasoning_effort=(reasoning_effort if enable_thinking else None))
            self._json({"id":cid,"object":"chat.completion","created":created,"model":service.args.name,"choices":[{"index":0,"message":message,"finish_reason":finish}],"usage":{"prompt_tokens":result["prompt_tokens"],"completion_tokens":result["completion_tokens"],"total_tokens":result["total_tokens"]},"pure_ane":metadata})
        def _completion(self,req:dict[str,Any])->None:
            prompt=req.get("prompt","")
            if not isinstance(prompt,str):raise ValueError("prompt must be a string")
            result=service.generate(prompt,**_request_options(req,service.args.max_tokens))
            self._json({"id":"cmpl-"+uuid.uuid4().hex,"object":"text_completion","created":int(time.time()),"model":service.args.name,"choices":[{"index":0,"text":result["text"],"finish_reason":result["finish_reason"]}],"usage":{"prompt_tokens":result["prompt_tokens"],"completion_tokens":result["completion_tokens"],"total_tokens":result["total_tokens"]},"pure_ane":result})
        def _benchmark(self,req:dict[str,Any])->None:
            runs=int(req.get("runs",3));warmup=int(req.get("warmup",1))
            if not 1<=runs<=100 or not 0<=warmup<=20:raise ValueError("runs must be 1..100 and warmup 0..20")
            enable_thinking,reasoning_effort=_thinking_options(req,default=False)
            prompt=(chat_prompt(req["messages"],enable_thinking,reasoning_effort)
                    if "messages" in req else str(req.get("prompt","Explain why persistent inference servers improve LLM benchmarking.")))
            options=_request_options(req,service.args.max_tokens)
            options["prefix_cache"]=bool(req.get("prefix_cache",False))
            for _ in range(warmup):service.generate(prompt,**options)
            measured=[service.generate(prompt,**options) for _ in range(runs)]
            rates=[x["tokens_per_second"] for x in measured]
            self._json({"object":"pure_ane.benchmark","model":service.args.name,"warmup":warmup,"runs":measured,"summary":{"runs":runs,"mean_tokens_per_second":statistics.fmean(rates),"median_tokens_per_second":statistics.median(rates),"min_tokens_per_second":min(rates),"max_tokens_per_second":max(rates),"mean_time_to_first_token":statistics.fmean(x["time_to_first_token"] for x in measured)}})
    return Handler


def serve(args:argparse.Namespace)->None:
    assert_standalone("server startup")
    print(f"PURE_ANE_SERVER loading {args.name} int{args.bits} context={args.context} mtp={args.mtp_draft} down_proj_parts={args.down_proj_parts}",flush=True)
    service=PureAneService(args);server=ThreadingHTTPServer((args.host,args.port),build_handler(service));server.daemon_threads=True
    print(f"PURE_ANE_SERVER_READY http://{args.host}:{args.port}/v1 model={args.name}",flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


def bench_client(args:argparse.Namespace)->None:
    payload={"max_tokens":args.tokens,"runs":args.runs,"warmup":args.warmup,
             "temperature":args.temperature,"prefix_cache":args.prefix_cache,
             "enable_thinking":args.thinking,
             "reasoning_effort":args.reasoning_effort}
    if args.raw_prompt:payload["prompt"]=args.prompt
    else:payload["messages"]=[{"role":"user","content":args.prompt}]
    request=urllib.request.Request(args.url.rstrip("/")+"/v1/benchmarks",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
    try:
        with urllib.request.urlopen(request,timeout=args.timeout) as response:data=json.load(response)
    except urllib.error.HTTPError as exc:raise SystemExit(exc.read().decode(errors="replace")) from exc
    print(json.dumps(data,indent=2))


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    s=sub.add_parser("serve")
    s.add_argument("--model",default=os.environ.get("Q38_MODEL","/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B"))
    s.add_argument("--engine-path",default=os.environ.get("Q38_ANE_ENGINE",str(_TOOLS.parent)))
    s.add_argument("--name",default="qwen3.8-27b-pure-ane");s.add_argument("--host",default="127.0.0.1");s.add_argument("--port",type=int,default=1240)
    s.add_argument("--bits",type=int,choices=(4,8,16),default=4);s.add_argument("--context",type=int,default=4096)
    s.add_argument("--mtp-draft",type=int,choices=(0,1,2),default=0);s.add_argument("--max-tokens",type=int,default=256)
    s.add_argument("--bake-cache",default=os.environ.get("Q38_ANE_BAKE_CACHE","auto"),
                   help="prequantized-weight cache directory (default: ~/Library/Caches/q38-pure-ane)")
    s.add_argument("--no-bake-cache",action="store_const",const="",dest="bake_cache",
                   help="disable the persistent prequantized-weight cache")
    s.add_argument("--profile-decode",action="store_true",
                   help="collect per-phase timings from the production model loop")
    s.add_argument("--down-proj-parts",type=int,choices=(1,4),default=4,
                   help="input-channel partitions for packed ANE down_proj")
    s.add_argument("--max-request-bytes",type=int,default=8*1024*1024)
    b=sub.add_parser("bench");b.add_argument("--url",default="http://127.0.0.1:1240")
    b.add_argument("--prompt",default="Reply with exactly: OK");b.add_argument("--tokens",type=int,default=16)
    b.add_argument("--raw-prompt",action="store_true")
    b.add_argument("--prefix-cache",action="store_true",
                   help="measure warm-prefix latency; cold-state runs are the default")
    b.add_argument("--thinking",action="store_true",
                   help="include Qwen thinking in benchmark generation")
    b.add_argument("--reasoning-effort",choices=("low","medium","xhigh"),
                   default="xhigh")
    b.add_argument("--runs",type=int,default=3);b.add_argument("--warmup",type=int,default=1)
    b.add_argument("--temperature",type=float,default=0.0);b.add_argument("--timeout",type=float,default=3600)
    args=parser.parse_args();serve(args) if args.command=="serve" else bench_client(args)


if __name__=="__main__":main()
