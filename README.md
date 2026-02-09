# Spelling Bee（拼写比赛训练网页）

## 语音作答：免费/低成本方案

网页端支持两种语音评分方式：

- 浏览器(快)：直接用浏览器自带语音识别，零配置，但对口音/噪声容错有限
- AI(更准)：录音上传到本地 `voice_ai_server.py`，由“ASR转写 + 容错判分(可选再加LLM判定)”输出正确/错误与得分

## 用免费/低成本 ASR + 大模型 API（推荐 Groq 免费额度）

本项目的后端实现为“OpenAI Compatible”调用方式，所以只要服务端兼容 OpenAI API 形态即可接入。

## 用 Qwen（DashScope / Model Studio）

启动 `voice_ai_server.py` 后，可以直接在网页“设置”里填写 DashScope 的 base_url / API Key / 模型，然后点“测试连接”。

- 国内（北京）：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- 国际（新加坡）：`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`

建议：
- 先把“语音评分方式”切到 **AI(更准)**，再配置
- 如果你只用 Qwen 来做“判卷”(更容错)，可开启“大模型判卷”，LLM 模型填 `qwen-plus` 或 `qwen-turbo`
- 如果你要用 Qwen 来做 ASR，需要把 ASR 调用方式切换为 `/v1/chat/completions（audio_url）`，并填写支持音频理解的模型名

### 1）启动 AI 评分服务（替代 http.server）

先停止你现在的：

```bash
python3 -m http.server 8000
```

改为启动：

```bash
PORT=8000 \
ASR_PROVIDER=openai_compatible \
ASR_BASE_URL=https://api.groq.com/openai \
ASR_API_KEY=你的GROQ_API_KEY \
ASR_MODEL=whisper-large-v3 \
python3 voice_ai_server.py
```

然后网页里：设置 → 语音评分方式 → 选择 **AI(更准)** → 测试连接。

### 2）可选：再加“大模型判卷”（更口语化容错）

如果想让“读音很像但转写文本不稳定”的情况更容易判对，可以打开 LLM 判卷：

```bash
LLM_JUDGE=1 \
LLM_BASE_URL=https://api.groq.com/openai \
LLM_API_KEY=你的GROQ_API_KEY \
LLM_MODEL=llama-3.1-70b-versatile \
```

（上述变量与上一段启动命令合并即可）

## 常见问题

- 为什么需要本地 `voice_ai_server.py`？
  - 纯静态网页无法安全地保存 API Key；本地服务负责保管 Key，并提供 `/api/voice-score` 给网页调用。
- iPad 上能用吗？
  - 浏览器支持录音权限即可；如果内置浏览器限制录音/跨域，建议用 Safari 打开本地网页。
