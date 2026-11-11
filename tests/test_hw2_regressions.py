"""Границы регламента и ошибки модели, которые не покрывает заготовка ДЗ2."""
import asyncio
import json

import pytest
from pydantic import ValidationError

from desk.escalation import Draft, Signals, atriage_many, needs_human
from desk.structured import StructuredError
from tests.fakes import FakeLLM, text_reply
from tests.test_hw2_escalation import GOOD, NONE, SOURCE, draft


@pytest.mark.parametrize("field,value,expected", [
    ("refund", None, False), ("refund", 0, False),
    ("refund", 5000, False), ("refund", 5001, True),
    ("duplicate", None, False), ("duplicate", 0, False),
    ("duplicate", 15000, False), ("duplicate", 15001, True),
    ("fraud", True, True), ("threat", True, True),
    ("asks_human", True, True), ("tariff_pro", True, True),
    ("merchant_down", True, True), ("key_leak", True, True),
])
def test_escalation_boundaries(field, value, expected):
    assert needs_human(Signals(**{**NONE, field: value})) is expected


@pytest.mark.parametrize("changes", [
    {"category": "VIP"}, {"severity": 0}, {"severity": 6}, {"amount": -1},
    {"quote": ""}, {"quote": "выдуманная цитата"},
    {"payment_ids": ["88150"]}, {"payment_ids": ["P-99999"]},
    {"payment_ids": []}, {"signals": {**NONE, "refund": -1}},
    {"signals": {**NONE, "duplicate": -1}},
])
def test_draft_rejects_invalid_answers(changes):
    with pytest.raises(ValidationError):
        Draft.model_validate({**GOOD, **changes}, context={"source": SOURCE})


def test_draft_cannot_omit_payment_ids_from_source():
    body = {k: v for k, v in GOOD.items() if k != "payment_ids"}
    with pytest.raises(ValidationError):
        Draft.model_validate(body, context={"source": SOURCE})


def test_batch_repairs_category_and_quote_before_building_ticket():
    llm = FakeLLM([draft(category="VIP"), draft(quote="не было в тексте"), draft()])
    result = asyncio.run(atriage_many(llm, [SOURCE]))[0]
    assert result.category == "возвраты" and result.needs_human
    assert len(llm.calls) == 3
    assert "category" in llm.calls[1]["messages"][-1]["content"]
    assert "quote" in llm.calls[2]["messages"][-1]["content"]


def test_batch_keeps_order_limits_concurrency_and_isolates_errors():
    state = {"active": 0, "peak": 0}

    class DelayedLLM:
        async def achat(self, messages, **kw):
            source = next(m["content"] for m in reversed(messages)
                          if m["role"] == "user" and m["content"].startswith("<обращение>"))
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            try:
                await asyncio.sleep(0.02 if "медленно" in source else 0.001)
                if "сеть" in source:
                    raise ConnectionError("нет соединения")
                if "сломай" in source:
                    return text_reply("не JSON")
                body = {
                    **GOOD, "category": "другое", "payment_ids": [], "amount": None,
                    "quote": "обращение", "severity": 4 if "срочно" in source else 1,
                    "signals": {**NONE, "asks_human": "человек" in source},
                }
                return text_reply(json.dumps(body, ensure_ascii=False))
            finally:
                state["active"] -= 1

    texts = ["обращение медленно срочно человек", "обращение сломай",
             "обращение обычно", "обращение сеть"] * 3
    out = asyncio.run(atriage_many(DelayedLLM(), texts, concurrency=2))
    assert len(out) == len(texts)
    assert state == {"active": 0, "peak": 2}
    for i in range(0, len(out), 4):
        assert out[i].severity == 4 and out[i].needs_human is True
        assert isinstance(out[i + 1], StructuredError)
        assert out[i + 2].severity == 1 and out[i + 2].needs_human is False
        assert isinstance(out[i + 3], ConnectionError)


@pytest.mark.parametrize("concurrency", [0, -1])
def test_batch_rejects_invalid_concurrency(concurrency):
    with pytest.raises(ValueError, match="concurrency"):
        asyncio.run(atriage_many(FakeLLM([]), [SOURCE], concurrency=concurrency))
