# -*- coding: utf-8 -*-
"""Семинар 1. Замер кандидатов: разбор ответа, процентиль, параллельный прогон, строка матрицы"""
import asyncio

import pytest

from desk.bench import (
    CANDIDATES,
    EXAMPLES,
    Candidate,
    Row,
    error_report,
    parse_category,
    percentile,
    prompt,
    run_candidate,
    summarize,
)
from desk.data import tickets
from desk.llm import LLMError, Usage
from desk.stream import StreamResult


def test_parse_category():
    assert parse_category("Категория: Возвраты.") == "возвраты"
    assert parse_category("скорее платежи, хотя возможно и возвраты") == "платежи"
    assert parse_category("не знаю") is None


def test_percentile():
    assert percentile([], 0.95) == 0.0
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.5) == 5
    assert percentile(list(range(1, 101)), 0.95) == 95
    assert percentile([7.0], 0.95) == 7.0


def test_candidate_with_examples_builds_few_shot_prompt_without_dataset_leak():
    cand = next(c for c in CANDIDATES if c.name == "правила + примеры")
    row = {"text": "Новое обращение"}

    messages = prompt(cand, row)

    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[-1] == {"role": "user", "content": "Новое обращение"}
    assert cand.examples == EXAMPLES and len(EXAMPLES) == 2
    assert {text for text, _ in EXAMPLES}.isdisjoint(
        {row["text"] for row in tickets(None)}
    )


def test_candidate_without_examples_keeps_original_prompt_shape():
    cand = Candidate("обычный", "постановка")

    assert prompt(cand, {"text": "Обращение"}) == [
        {"role": "system", "content": "постановка"},
        {"role": "user", "content": "Обращение"},
    ]


class FakeStreamLLM:
    """Отвечает категорией из текста обращения; «сломай» даёт LLMError"""

    class cfg:
        extra_body = {"base": 1}

    def __init__(self):
        self.now = 0
        self.peak = 0
        self.bodies = []

    def variant(self, **changes):
        self.bodies.append(changes["extra_body"])
        return self

    async def astream(self, messages, **kw):
        self.now += 1
        self.peak = max(self.peak, self.now)
        await asyncio.sleep(0.01)
        self.now -= 1
        text = messages[-1]["content"]
        if "сломай" in text:
            raise LLMError("провайдер ответил 400")
        answer = text.split()[0]
        stop = "max_tokens" if "длинно" in text else "end_turn"
        return StreamResult(answer, stop, Usage(100, 5, 0, 0, 0.3, 0.001, 105, calls=1), 0.1, 0.3)


ROWS = [{"id": "T-%d" % i, "text": t, "gold": {"category": g}} for i, (t, g) in enumerate([
    ("платежи не проходят", "платежи"), ("доступ пропал", "доступ"), ("сломай всё", "тарифы"),
    ("возвраты длинно", "возвраты"), ("другое", "платежи")] * 3)]


def test_run_candidate_keeps_order_limits_concurrency_and_survives_failures():
    llm = FakeStreamLLM()
    cand = Candidate("тест", "постановка", body={"thinking": {"type": "enabled", "budget_tokens": 1024}})
    rows = asyncio.run(run_candidate(llm, cand, ROWS, concurrency=3))
    assert [r.id for r in rows] == [r["id"] for r in ROWS]
    assert llm.peak <= 3
    assert llm.bodies == [{"base": 1, "thinking": {"type": "enabled", "budget_tokens": 1024}}]
    assert [r.ok for r in rows[:5]] == [True, True, False, True, False]
    assert rows[2].failed and not rows[0].failed and rows[3].truncated
    assert (rows[0].expected, rows[0].predicted, rows[0].text) == (
        "платежи",
        "платежи",
        "платежи не проходят",
    )
    assert (rows[2].expected, rows[2].predicted) == ("тарифы", None)


def test_error_report_lists_counts_categories_and_ticket_text():
    rows = [
        Row("T-1", True, expected="платежи", predicted="платежи", text="Успех"),
        Row("T-2", False, expected="доступ", predicted="другое", text="Не войти"),
        Row("T-3", False, expected="тарифы", text="Ошибка API", failed=True),
    ]

    report = error_report("кандидат", rows)

    assert "кандидат: 2 ошибки из 3" in report
    assert "T-2 | доступ | другое | Не войти" in report
    assert "T-3 | тарифы | сбой | Ошибка API" in report


def test_row_keeps_existing_positional_arguments_compatible():
    row = Row("T-1", False, True, True, 1.5, 0.4)

    assert row.failed and row.truncated
    assert (row.latency_s, row.ttft_s) == (1.5, 0.4)


def test_summarize_counts_failures_as_errors():
    rows = [Row("a", True, latency_s=1.0, ttft_s=0.2, usage=Usage(100, 10, 0, 0, 1.0, 0.002, 110, calls=1)),
            Row("b", False, latency_s=3.0, ttft_s=0.4, truncated=True,
                usage=Usage(100, 10, 0, 0, 3.0, 0.002, 110, calls=1)),
            Row("c", False, failed=True)]
    s = summarize("кандидат", rows, flow_per_day=1000)
    assert s["точность"] == pytest.approx(1 / 3)
    assert (s["p50, с"], s["p95, с"], s["первый токен p50, с"]) == (1.0, 3.0, 0.2)
    assert s["входных на обращение"] == pytest.approx(100)
    assert s["взвешенных на обращение"] == pytest.approx(110)
    assert s["цена за 1000, у.е."] == pytest.approx(2.0)
    assert s["взвешенных в месяц, млн"] == pytest.approx(110 * 1000 * 30 / 1e6)
    assert s["цена в месяц, у.е."] == pytest.approx(0.002 * 1000 * 30)
    assert (s["обрезано"], s["сбоев"]) == (1, 1)


def test_candidates_are_consistent():
    thinking = [c for c in CANDIDATES if "thinking" in c.body]
    assert thinking and all(c.max_tokens > c.body["thinking"]["budget_tokens"] for c in thinking)
