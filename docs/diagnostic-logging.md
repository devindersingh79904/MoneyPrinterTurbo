# Diagnostic Logging

In the WebUI, open **Settings → Interface Settings → Diagnostic logging**, choose a log level, enable
**Trace API request parameters** if needed, and click **Apply logging settings**. The settings take effect
immediately on the current service, including subsequent logs from tasks that are already running; all browser
sessions of the same service share the setting. Enabling "Hide Log" only hides the in-page log view and does not
turn off terminal output.

The API, CLI and WebUI share the same configuration:

```toml
[diagnostics]
log_level = "INFO"
log_requests = false
```

`log_level` accepts TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR and CRITICAL.
When omitted, the previous top-level `log_level` is used, defaulting to DEBUG. Editing the file by hand requires
a restart; "Apply" in the WebUI only updates the current process immediately and does not remotely change other
API or CLI processes.

Request tracing is an independent switch and can be enabled even at the ERROR level. It is off by default and records:

- The request ID, HTTP method, redacted URL and the length-limited request parameters.
- The response status (when available) and elapsed time, or the exception type on failure.
- LLM SDK call parameters. These are the SDK's inputs, not the raw JSON the SDK eventually sends;
  the SDK's internal automatic retries are not shown one by one. For streaming requests, the elapsed time
  only covers obtaining the response or stream object and excludes the subsequent download time.

It covers the project's explicit `requests` calls for materials, video, speech, publishing and sign-in, as well as
the OpenAI-compatible, Azure OpenAI, Qwen, Gemini and LiteLLM script-generation calls.
It does not capture the browser's own network traffic, inbound API request bodies, CLI subprocesses, the Edge TTS
WebSocket or other SDK-internal transports. The normal application log level and the log levels of third-party
SDKs are separate switches as well.

Request headers, response bodies and uploaded file contents are never logged. Keys, tokens, cookies, passwords and
signature fields are masked, the values of URL query parameters are hidden uniformly, and audio/video, base64 and
similar content is omitted. Raw unstructured request bodies are not logged. **Prompts and other business parameters
may appear in the logs**, so turn tracing off once troubleshooting is done and still check the logs for private
content before sharing them. Diagnostics never send logs externally and never issue, retry or replay additional requests.

CLI logs still go to stderr while stdout remains machine-readable JSON. WebUI task logs keep the existing
thread-ownership filtering and line limit, and switching the level does not remove the log handlers of running tasks.
