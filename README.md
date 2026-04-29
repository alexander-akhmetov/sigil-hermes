# hermes-plugin-sigil

Grafana AI Observability plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent). Records LLM calls and tool executions as Sigil generations and emits OTel traces + metrics.

## Install

```bash
pip install git+https://github.com/alexander-akhmetov/sigil-hermes
```

Install into the same Python environment hermes runs from (`which hermes` to check). Then enable the plugin in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - sigil
```

> Hermes's `plugins enable` CLI does not see pip-installed plugins yet — it only scans `~/.hermes/plugins/` and the bundled directory. Editing the YAML directly is the workaround.

## Configure

Two independent channels, OTel (metrics and traces) and Sigil (conversation data), you need to set both.
You can find the credentials and URLs in your Grafana account: `https://grafana.com/orgs/{org}`

```bash
# Generations → Sigil API (Conversations)
export HERMES_SIGIL_ENDPOINT="https://sigil-prod-<region>.grafana.net"
export HERMES_SIGIL_INSTANCE_ID="<sigil-instance-id>"

# You can find this token in your stack info -> "AI Observability" card
# in https://grafana.com/orgs/{org-id}/stacks/{stack-id}`
export HERMES_SIGIL_API_KEY="<sigil:write token>"

# Traces + metrics → Grafana Cloud OTLP gateway
export HERMES_SIGIL_OTLP_ENDPOINT="https://otlp-gateway-prod-<region>.grafana.net/otlp"
export HERMES_SIGIL_OTLP_INSTANCE_ID="<grafana-cloud-instance-id>"

# You can find this token in your stack info -> "OpenTelemetry" card
# in https://grafana.com/orgs/{org-id}/stacks/{stack-id}`
export HERMES_SIGIL_OTLP_TOKEN="<grafana-cloud-otlp-token>"

# Enable sending conversations content. Set to metadata_only to disable.
export HERMES_SIGIL_CONTENT_CAPTURE=full
```
### Optional

| Variable | Default | Description |
|---|---|---|
| `HERMES_SIGIL_AGENT_NAME` | `hermes` | `service.name` resource attribute on every span |
| `HERMES_SIGIL_SAMPLE_RATE` | `1.0` | Fraction of LLM and tool calls to record, `0.0`–`1.0` |
| `HERMES_SIGIL_CONTENT_CAPTURE` | `full` | `full` / `no_tool_content` / `metadata_only`. Defaults to `full` so tool args and results are visible — the SDK's own default is `no_tool_content`, which leaves agent conversations looking empty. |
| `HERMES_SIGIL_MAX_CHARS` | `12000` | Per-string truncation cap for redacted payloads |
| `HERMES_SIGIL_DEBUG` | `false` | Verbose plugin logs |
| `HERMES_SIGIL_OTEL_AUTO` | `true` | Set `false` if your application already installs a `TracerProvider` / `MeterProvider` |

## Verify

```bash
HERMES_SIGIL_DEBUG=true hermes
```

In `~/.hermes/logs/agent.log` you should see:

```
hermes-plugin-sigil: installed TracerProvider with OTLP HTTP exporter
hermes-plugin-sigil: installed MeterProvider with OTLP HTTP exporter
hermes-plugin-sigil: Sigil client initialized (generations=on, otlp=on)
```

Ask hermes anything, then check **Grafana Cloud → AI Observability → Conversations**.

## License

Apache-2.0.
