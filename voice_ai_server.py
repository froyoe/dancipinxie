import os
import json
import http.server
import socketserver
import urllib.request
import urllib.error
import cgi
import mimetypes
import re
import base64
import traceback


def _json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def _normalize_text(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_compact(s):
    return _normalize_text(s).replace(" ", "")


def _levenshtein(a, b):
    a = a or ""
    b = b or ""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[n]


def _best_window(expected, heard):
    e = _normalize_text(expected)
    h = _normalize_text(heard)
    if not e or not h:
        return {"best": h, "ratio": 0.0}
    if e == h:
        return {"best": h, "ratio": 1.0}
    e_tokens = e.split(" ")
    h_tokens = h.split(" ")
    w = max(1, len(e_tokens))
    cmp_e = _normalize_compact(e)
    best = h
    best_ratio = 0.0
    for i in range(0, len(h_tokens)):
        slice_ = " ".join(h_tokens[i : i + w])
        cmp_h = _normalize_compact(slice_)
        if not cmp_h:
            continue
        dist = _levenshtein(cmp_e, cmp_h)
        ratio = 1.0 - dist / max(len(cmp_e), len(cmp_h), 1)
        if ratio > best_ratio:
            best_ratio = ratio
            best = slice_
    return {"best": best, "ratio": float(best_ratio)}


def _judge(expected, transcript, tolerance):
    tol = 1
    try:
        tol = int(tolerance)
    except Exception:
        tol = 1
    threshold = 0.88
    if tol == 0:
        threshold = 0.95
    if tol == 2:
        threshold = 0.80
    exp_len = len(_normalize_compact(expected))
    if exp_len <= 4:
        threshold += 0.05
    m = _best_window(expected, transcript)
    ok = m["ratio"] >= threshold
    score = int(round(m["ratio"] * 100))
    return {"heard": m["best"], "ratio": m["ratio"], "score": score, "correct": bool(ok), "threshold": threshold}


def _multipart(boundary, fields, file_field):
    b = boundary.encode("utf-8")
    body = bytearray()
    for k, v in fields.items():
        body.extend(b"--" + b + b"\r\n")
        body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(v).encode("utf-8"))
        body.extend(b"\r\n")
    name, filename, content_type, data = file_field
    body.extend(b"--" + b + b"\r\n")
    body.extend(
        f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8")
    )
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.extend(data)
    body.extend(b"\r\n")
    body.extend(b"--" + b + b"--\r\n")
    return bytes(body)


def _openai_compat_base_url():
    base = (os.environ.get("ASR_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com").strip()
    return base.rstrip("/")


_RUNTIME_CFG = {
    "asr_provider": None,
    "asr_base_url": None,
    "asr_api_key": None,
    "asr_model": None,
    "asr_mode": None,
    "llm_judge": None,
    "llm_base_url": None,
    "llm_api_key": None,
    "llm_model": None,
}


def _get_cfg(name):
    v = _RUNTIME_CFG.get(name)
    if v is None:
        return None
    if isinstance(v, str) and not v.strip():
        return None
    return v

def _asr_config():
    provider = (_get_cfg("asr_provider") or os.environ.get("ASR_PROVIDER") or "openai_compatible").strip()
    key = (
        _get_cfg("asr_api_key")
        or os.environ.get("ASR_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    model = (
        _get_cfg("asr_model")
        or os.environ.get("ASR_MODEL")
        or os.environ.get("OPENAI_AUDIO_MODEL")
        or "whisper-1"
    ).strip() or "whisper-1"
    base_url = (_get_cfg("asr_base_url") or _openai_compat_base_url()).strip().rstrip("/")
    mode = (_get_cfg("asr_mode") or os.environ.get("ASR_MODE") or "transcriptions").strip()
    return {"provider": provider, "key": key, "model": model, "base_url": base_url, "mode": mode}


def _volcengine_asr_transcribe(audio_bytes, filename, content_type, key, model):
    # key format: "APPID TOKEN"
    parts = key.strip().split()
    if len(parts) < 2:
        raise RuntimeError("Volcengine ASR Key 格式错误，需为 'APPID TOKEN' (中间用空格隔开)")
    appid, token = parts[0], parts[1]
    cluster = model if model and model != "whisper-1" else "volc_auc_common"

    # Map content_type to Volcengine format
    fmt = "webm"
    if "wav" in content_type: fmt = "wav"
    elif "mp3" in content_type: fmt = "mp3"
    elif "ogg" in content_type: fmt = "ogg"
    elif "aac" in content_type: fmt = "aac"
    elif "m4a" in content_type: fmt = "m4a"
    elif "pcm" in content_type: fmt = "pcm"
    
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    
    url = "https://openspeech.bytedance.com/api/v1/vc/submit"
    payload = {
        "app": {
            "appid": appid,
            "token": token,
            "cluster": cluster
        },
        "user": {
            "uid": "spelling_bee_user"
        },
        "audio": {
            "format": fmt,
            "data": b64
        }
    }
    
    req = urllib.request.Request(
        url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer; {appid} {token}",
            "Content-Type": "application/json"
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", "ignore")
            data = json.loads(raw or "{}")
            if data.get("code") != 1000:
                 # 3000 is often partial success or processing
                 raise RuntimeError(f"Volcengine Error: {data.get('code')} {data.get('message')}")
            return (data.get("result") or [{}])[0].get("text", "") or ""
    except urllib.error.HTTPError as e:
         msg = e.read().decode("utf-8", "ignore")
         raise RuntimeError(f"Volcengine Request Failed: {e.code} {msg[:300]}")


def _llm_config():
    enabled = _get_cfg("llm_judge")
    if enabled is None:
        enabled = (os.environ.get("LLM_JUDGE") or "").strip() in ("1", "true", "TRUE", "yes", "YES")
    else:
        enabled = bool(enabled)
    key = (
        _get_cfg("llm_api_key")
        or _get_cfg("asr_api_key")
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("ASR_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    model = (_get_cfg("llm_model") or os.environ.get("LLM_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"
    base_url = (
        _get_cfg("llm_base_url")
        or _get_cfg("asr_base_url")
        or os.environ.get("LLM_BASE_URL")
        or os.environ.get("ASR_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com"
    ).strip().rstrip("/")
    return {"enabled": enabled, "key": key, "model": model, "base_url": base_url}


def _openai_compatible_transcribe(audio_bytes, filename, content_type, key, base_url, model):
    if not key:
        raise RuntimeError("ASR_API_KEY 未设置")
    boundary = "----spellingbee" + os.urandom(8).hex()
    fields = {"model": model, "response_format": "json"}
    file_field = ("file", filename or "audio.webm", content_type or "audio/webm", audio_bytes)
    body = _multipart(boundary, fields, file_field)
    req = urllib.request.Request(
        f"{base_url}/v1/audio/transcriptions",
        method="POST",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8", "ignore")
            data = json.loads(raw or "{}")
            return data.get("text", "") or ""
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "ignore")
        raise RuntimeError(f"ASR 请求失败: {e.code} {msg[:300]}")


def _openai_compatible_chat_asr_audio_url(audio_bytes, filename, content_type, key, base_url, model):
    if not key:
        raise RuntimeError("ASR_API_KEY 未设置")
    mime = content_type or "audio/webm"
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe the audio to plain text only."},
                    {"type": "audio_url", "audio_url": {"url": data_url}},
                ],
            }
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        method="POST",
        data=data,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8", "ignore")
            j = json.loads(raw or "{}")
            content = (((j.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
            return str(content).strip()
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "ignore")
        raise RuntimeError(f"ASR(chat) 请求失败: {e.code} {msg[:300]}")


def _openai_compatible_chat_json(key, base_url, model, system, user):
    if not key:
        raise RuntimeError("LLM_API_KEY 未设置")
    payload = {
        "model": model,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        method="POST",
        data=data,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8", "ignore")
            j = json.loads(raw or "{}")
            content = (((j.get("choices") or [{}])[0].get("message") or {}).get("content")) or "{}"
            return json.loads(content)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "ignore")
        raise RuntimeError(f"LLM 请求失败: {e.code} {msg[:300]}")


def _asr_transcribe(audio_bytes, filename, content_type):
    cfg = _asr_config()
    if cfg["provider"] == "volcengine":
        return _volcengine_asr_transcribe(
            audio_bytes=audio_bytes,
            filename=filename,
            content_type=content_type,
            key=cfg["key"],
            model=cfg["model"]
        )
    if cfg["provider"] != "openai_compatible":
        raise RuntimeError("ASR_PROVIDER 仅支持 openai_compatible 或 volcengine")
    if cfg.get("mode") == "chat_audio_url":
        return _openai_compatible_chat_asr_audio_url(
            audio_bytes=audio_bytes,
            filename=filename,
            content_type=content_type,
            key=cfg["key"],
            base_url=cfg["base_url"],
            model=cfg["model"],
        )
    return _openai_compatible_transcribe(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
        key=cfg["key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
    )


class Handler(http.server.SimpleHTTPRequestHandler):
    def _send_json(self, code, obj):
        data = _json_bytes(obj)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/ping"):
            asr = _asr_config()
            llm = _llm_config()
            engine = "none"
            if asr["key"]:
                engine = "groq" if "groq.com" in asr["base_url"] else "openai_compatible"
            return self._send_json(
                200,
                {
                    "ok": True,
                    "engine": engine,
                    "asr_provider": asr["provider"],
                    "asr_base_url": asr["base_url"],
                    "asr_model": asr["model"],
                    "asr_mode": asr.get("mode"),
                    "asr_has_key": bool(asr.get("key")),
                    "llm_judge": bool(llm["enabled"]),
                    "llm_base_url": llm["base_url"],
                    "llm_model": llm["model"],
                    "llm_has_key": bool(llm.get("key")),
                },
            )
        if self.path.startswith("/api/config"):
            return self._send_json(
                200,
                {
                    "ok": True,
                    "asr_provider": _get_cfg("asr_provider"),
                    "asr_base_url": _get_cfg("asr_base_url"),
                    "asr_model": _get_cfg("asr_model"),
                    "asr_mode": _get_cfg("asr_mode"),
                    "llm_judge": _get_cfg("llm_judge"),
                    "llm_base_url": _get_cfg("llm_base_url"),
                    "llm_model": _get_cfg("llm_model"),
                },
            )
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/config"):
            return self._handle_config()
        if self.path.startswith("/api/voice-score"):
            return self._handle_voice_score()
        return self._send_json(404, {"ok": False, "error": "not found"})

    def _handle_config(self):
        try:
            n = int(self.headers.get("content-length", "0") or "0")
        except Exception:
            n = 0
        if n <= 0 or n > 100_000:
            return self._send_json(400, {"ok": False, "error": "bad content-length"})
        raw = self.rfile.read(n).decode("utf-8", "ignore")
        try:
            obj = json.loads(raw or "{}")
        except Exception:
            return self._send_json(400, {"ok": False, "error": "invalid json"})
        if not isinstance(obj, dict):
            return self._send_json(400, {"ok": False, "error": "invalid payload"})
        allow = {
            "asr_provider": str,
            "asr_base_url": str,
            "asr_api_key": str,
            "asr_model": str,
            "asr_mode": str,
            "llm_judge": (bool, int),
            "llm_base_url": str,
            "llm_api_key": str,
            "llm_model": str,
        }
        for k, typ in allow.items():
            if k not in obj:
                continue
            v = obj.get(k)
            if v is None:
                _RUNTIME_CFG[k] = None
                continue
            if not isinstance(v, typ):
                continue
            if isinstance(v, str) and len(v) > 10_000:
                continue
            if k in ("llm_judge",):
                _RUNTIME_CFG[k] = bool(v)
            else:
                _RUNTIME_CFG[k] = v
        return self._send_json(200, {"ok": True})

    def _handle_voice_score(self):
        ctype, pdict = cgi.parse_header(self.headers.get("content-type", ""))
        if ctype != "multipart/form-data":
            return self._send_json(400, {"ok": False, "error": "content-type must be multipart/form-data"})
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("content-type")},
        )
        expected = form.getfirst("expected", "").strip()
        tolerance = form.getfirst("tolerance", "1")
        if not expected:
            return self._send_json(400, {"ok": False, "error": "missing expected"})
        if "audio" not in form:
            return self._send_json(400, {"ok": False, "error": "missing audio"})
        f = form["audio"]
        audio_bytes = f.file.read() if getattr(f, "file", None) else b""
        filename = getattr(f, "filename", None) or "audio.webm"
        content_type = getattr(f, "type", None) or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        if not audio_bytes:
            return self._send_json(400, {"ok": False, "error": "empty audio"})
        try:
            transcript = _asr_transcribe(audio_bytes, filename, content_type)
            judged = _judge(expected, transcript, tolerance)
            llm = _llm_config()
            llm_out = None
            if llm["enabled"]:
                system = "你是英语拼写考试的判卷老师。只输出JSON。"
                user = json.dumps(
                    {
                        "expected": expected,
                        "transcript": transcript,
                        "score": judged["score"],
                        "tolerance": tolerance,
                        "task": "判断孩子是否把目标单词读对了，允许常见口音与轻微吞音；如明显不是这个词则判错。",
                        "output_schema": {"correct": True, "reason": "简短原因"},
                    },
                    ensure_ascii=False,
                )
                llm_out = _openai_compatible_chat_json(
                    key=llm["key"],
                    base_url=llm["base_url"],
                    model=llm["model"],
                    system=system,
                    user=user,
                )
                if isinstance(llm_out, dict) and "correct" in llm_out:
                    judged["correct"] = bool(llm_out.get("correct"))
            return self._send_json(
                200,
                {
                    "ok": True,
                    "expected": expected,
                    "transcript": transcript,
                    **judged,
                    "llm": llm_out,
                },
            )
        except Exception as e:
            traceback.print_exc()
            return self._send_json(500, {"ok": False, "error": str(e)})


def main():
    port = int(os.environ.get("PORT", "8000"))
    with socketserver.TCPServer(("", port), Handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
