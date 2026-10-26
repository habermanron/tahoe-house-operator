import pytest

from tahoe_operator.corpus import CorpusService


@pytest.mark.asyncio
async def test_keyword_fallback_finds_fixture_rule(monkeypatch):
    service = CorpusService()
    monkeypatch.setattr(service, "base_url", "")
    result = await service.search("bear box storm pickup", limit=3)
    assert result["mode"] == "keyword_fallback"
    assert result["chunks"]
    assert any("bear box" in chunk["text"].lower() for chunk in result["chunks"])
