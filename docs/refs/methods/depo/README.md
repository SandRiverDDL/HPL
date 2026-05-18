# DEPO

## 一句话

DEPO（Dual-Efficiency Preference Optimization）把 agent 效率显式拆成两个维度：每步 token 更少、完成任务步数更少，并用偏好优化同时提升任务表现和交互效率。

## 适合解决的问题

- agent 成功率尚可，但 Thought 太长、点击/交互步数太多。
- 需要降低推理成本和延迟。
- WebShop 这类既有 token 成本又有环境交互成本的任务。

## 不适合的问题

- 任务成功率还很低，过早压缩 reasoning 可能损害探索。
- 偏好数据无法区分“简洁有效”和“省略关键信息”。
- 需要完整透明长 CoT 作为调试依据。

## 关键结论

- DEPO 关注 dual-efficiency：step-level token efficiency 和 trajectory-level step efficiency。
- 论文报告 WebShop 和 BabyAI 上 token、step、performance 都有改善。
- 它适合作为 HPL 的效率增强方向，而不是直接替代 HPL。

## 对 HPL 的启发

- HPL group-level chosen/rejected 可以加入效率 tie-breaker。
- 同样成功时，偏好更短 group、更少 token、更少无效搜索。
- 可以把 group length curriculum 和 DEPO 的 step efficiency 结合。

## 资料来源

- paper: https://arxiv.org/abs/2511.15392
- project: https://opencausalab.github.io/DEPO/
