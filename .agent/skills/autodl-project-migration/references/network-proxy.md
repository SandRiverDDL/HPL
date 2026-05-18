# AutoDL Network And Codex Proxy

## Decision Rule

先判断当前执行环境在哪里：

- 如果当前 agent 运行在个人电脑，才考虑转发个人电脑本地代理端口，例如 `127.0.0.1:7890`。
- 如果当前 agent 运行在实验室服务器，个人电脑的 `7890` 不可见；不要假设能用。
- 如果实验室服务器能直连 OpenAI，而 AutoDL 不能，优先用 `ssh -R` remote dynamic SOCKS，让 AutoDL 通过实验室服务器出网。

## Preflight

在当前执行环境检查是否能直连 OpenAI：

```bash
curl -I -m 10 https://api.openai.com/v1/models
```

预期可用结果是 `HTTP/2 401`，表示网络通但未带 key。

在 AutoDL 上检查直连：

```bash
curl -I -m 15 https://api.openai.com/v1/models || true
```

如果 timeout，再配置反向 SOCKS。

## Remote Dynamic SOCKS

从实验室服务器发起，保持该 SSH 会话不断：

```bash
ssh -p <PORT> -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o StrictHostKeyChecking=accept-new \
  -R 127.0.0.1:17891 \
  root@<HOST>
```

含义：

```text
AutoDL 127.0.0.1:17891
-> ssh -R remote dynamic SOCKS
-> 实验室服务器出网
-> api.openai.com
```

注意：这里的 `-R 127.0.0.1:17891` 没有目标 host/port，是 OpenSSH 的 remote dynamic forwarding，远端监听 SOCKS 端口。

## Validation

在 AutoDL 上验证：

```bash
curl -I -m 20 --socks5-hostname 127.0.0.1:17891 https://api.openai.com/v1/models
```

成功标志：

```text
HTTP/2 401
www-authenticate: Bearer realm="OpenAI API"
```

`401` 是正常结果，说明网络通，只是没有 API key。

## Use For Codex

如果项目提供脚本，优先使用脚本写入或删除 AutoDL `.bashrc` 配置：

```bash
scripts/autodl/configure_codex_proxy.sh
scripts/autodl/remove_codex_proxy.sh
```

AutoDL 上启动 Codex 前设置：

```bash
export ALL_PROXY=socks5h://127.0.0.1:17891
export HTTPS_PROXY=socks5h://127.0.0.1:17891
export HTTP_PROXY=socks5h://127.0.0.1:17891
```

然后再启动 Codex 或其他需要访问 OpenAI 后端的命令。

## Common Pitfalls

- 不要把个人电脑的 `127.0.0.1:7890` 当成实验室服务器的端口。
- 普通 `ssh -R 127.0.0.1:17890:127.0.0.1:7890` 只适合当前执行环境本身有可用 HTTP/SOCKS 代理。
- 如果代理测试出现 `HTTP/1.1 200 Connection established` 后 TLS EOF，通常是代理端口或协议不匹配，不是 `ssh -R` 本身失败。
- AutoDL 学术加速适合 GitHub/HF 下载，不等于 OpenAI API 可用。

## AutoDL Academic Acceleration

下载 GitHub / HF 资源时可以临时开启：

```bash
source /etc/network_turbo
env | grep -i proxy
```

下载结束后关闭，避免影响正常网络：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

Notebook 中如果需要复用，要把 `source /etc/network_turbo` 得到的 proxy 环境变量写入 Python `os.environ`。
