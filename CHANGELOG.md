# Changelog

All notable changes to `agentinc-sdk` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-05-13

### Added

- **AgentProtocol** — runtime-checkable protocol defining `run(input) -> AsyncIterator[AgentOutput]`
- **ToolProtocol** — runtime-checkable protocol defining `schema()` and `call()`
- **Schemas** — `AgentInput`, `AgentOutput`, `Message`, `ToolCall`, `ToolSchema` (Pydantic v2 models)
- **@tool decorator** — turns any sync/async function into a `ToolProtocol` with auto-generated JSON Schema
- **ToolWrapper** — class backing `@tool`, callable directly or via `ToolCall` dispatch
- **RawAdapter** — wraps any callable as `AgentProtocol`, auto-detecting 6 signature patterns
- **TenantContext** — multi-tenant isolation context, carried via `input.metadata["tenant"]`
- **A2A serve module** — `create_app()` and `serve()` helpers to expose agents over A2A (JSON-RPC 2.0 + SSE)
- **Examples** — echo, streaming, tool, OpenAI, Anthropic, LangChain, and CrewAI agents
- **Documentation** — single-page HTML docs with full API reference
