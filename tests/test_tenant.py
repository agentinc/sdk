from agentinc.sdk import TenantContext, AgentInput


def test_tenant_context():
    ctx = TenantContext(tenant_id="t-123", org_id="org-456")
    assert ctx.tenant_id == "t-123"
    assert ctx.org_id == "org-456"
    assert ctx.quotas == {}


def test_tenant_in_metadata():
    ctx = TenantContext(tenant_id="t-1", quotas={"max_tokens": 1000})
    inp = AgentInput(message="hi", metadata={"tenant": ctx.model_dump()})
    tenant = TenantContext.model_validate(inp.metadata["tenant"])
    assert tenant.tenant_id == "t-1"
    assert tenant.quotas["max_tokens"] == 1000
