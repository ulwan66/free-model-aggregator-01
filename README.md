# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/ulwan66/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/ulwan66/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that answers OpenAI Chat Completions and Anthropic Messages requests from ten simulated free-tier providers. Each request goes to the best free model that meets its requirements — tools, vision, JSON mode, context length — chosen from the trial's quality table, speaking the provider's own wire (OpenAI, Anthropic, or Gemini), with failover to the next candidate when a provider refuses. Answers come back in the wire the client used, streamed or not as asked, with the provider's text whole, its usage counts, and its stop reason mapped. `node main.ts --list` prints the provider catalog the entry registers.

## Run it

```sh
node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=s3fge52m&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
