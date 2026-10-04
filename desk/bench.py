# -*- coding: utf-8 -*-
"""Замер кандидатов

    python -m desk.bench --n 5     # пробный запуск
    python -m desk.bench --n 30    # сдаточный замер
    python -m desk.bench --n 30 --only "короткая постановка"

Кандидат это постановка задачи и добавка к запросу. Все кандидаты
проходят одни и те же обращения. Таблица печатается и сохраняется
в runs/s1_bench.md
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import RUNS
from .data import CATEGORIES, tickets
from .llm import LLM, LLMError, Usage

SHORT = (
    "Отнеси обращение в поддержку платёжного сервиса к одной категории: %s. "
    "Ответь одним словом." % ", ".join(CATEGORIES)
)

DETAILED = (
    SHORT
    + """
Правила:
- платежи: платёж не проходит, двойное списание, деньги не дошли, переводы, лимиты;
- возвраты: просьба вернуть деньги за покупку, статус возврата, спор через банк;
- доступ: вход, пароль, смена номера, блокировка, второй фактор;
- тарифы: комиссии, абонентская плата, смена тарифа, вывод выручки;
- интеграция: API, ключи, уведомления о платежах;
- другое: всё остальное; при сомнении выбирай «другое»."""
)

# Рассуждение. max_tokens кандидата должен быть больше budget_tokens
THINKING = {"thinking": {"type": "enabled", "budget_tokens": 1024}}

# Обращений в день для прогноза на месяц
FLOW_PER_DAY = 50_000

# Разобранные обращения на границах категорий. Примеры придуманы для задания
# и не встречаются в data/tickets.jsonl.
EXAMPLES = [
    (
        "Клиент не может завершить покупку: банк не присылает одноразовый "
        "код подтверждения для оплаты иностранной картой.",
        "платежи",
    ),
    (
        "Магазин уже перечислил покупателю возврат, но подтверждающая "
        "квитанция в кабинете так и не появилась.",
        "возвраты",
    ),
]


@dataclass
class Candidate:
    name: str
    system: str
    body: Dict[str, Any] = field(default_factory=dict)  # добавка к запросу
    max_tokens: int = 16
    examples: List[Tuple[str, str]] = field(default_factory=list)


CANDIDATES = [
    Candidate("короткая постановка", SHORT),
    Candidate("постановка с правилами", DETAILED),
    Candidate("правила + рассуждение", DETAILED, body=THINKING, max_tokens=2048),
    Candidate("правила + примеры", DETAILED, examples=EXAMPLES),
]


@dataclass
class Row:
    """Итог одного обращения"""

    id: str
    ok: bool  # категория совпала с эталоном
    failed: bool = False  # вызов не удался
    truncated: bool = False  # ответ оборван
    latency_s: float = 0.0
    ttft_s: Optional[float] = None
    usage: Usage = field(default_factory=Usage)
    expected: Optional[str] = None
    predicted: Optional[str] = None
    text: str = ""


def prompt(cand: Candidate, row: Dict[str, Any]) -> List[Dict[str, str]]:
    """Постановка кандидата, разобранные примеры и текст обращения"""
    messages = [{"role": "system", "content": cand.system}]
    for text, answer in cand.examples:
        messages.extend(
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": answer},
            ]
        )
    return messages + [{"role": "user", "content": row["text"]}]


def parse_category(text: str) -> Optional[str]:
    """Первая категория, которая встретилась в ответе"""
    low = text.lower()
    found = [(low.find(c), c) for c in CATEGORIES if c in low]
    return min(found)[1] if found else None


def percentile(values: List[float], q: float) -> float:
    """Процентиль q списка значений"""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(-(-q * len(ordered) // 1)) - 1))
    return ordered[idx]


async def run_candidate(
    llm: Any, cand: Candidate, rows: List[Dict[str, Any]], concurrency: int = 4
) -> List[Row]:
    """Прогоняет обращения через кандидата, не больше concurrency сразу

    Если вызов не удался, обращение засчитывается как ошибка,
    и замер идёт дальше
    """
    client = llm.variant(extra_body={**llm.cfg.extra_body, **cand.body})
    gate = asyncio.Semaphore(concurrency)

    async def one(row: Dict[str, Any]) -> Row:
        expected = row["gold"]["category"]
        async with gate:
            try:
                res = await client.astream(
                    prompt(cand, row), max_tokens=cand.max_tokens
                )
            except LLMError:
                return Row(
                    row["id"],
                    ok=False,
                    expected=expected,
                    text=row["text"],
                    failed=True,
                )
        predicted = parse_category(res.text)
        return Row(
            row["id"],
            predicted == expected,
            expected=expected,
            predicted=predicted,
            text=row["text"],
            truncated=res.truncated,
            latency_s=res.total_s,
            ttft_s=res.ttft_s,
            usage=res.usage,
        )

    return list(await asyncio.gather(*(one(r) for r in rows)))


def summarize(
    name: str, rows: List[Row], flow_per_day: int = FLOW_PER_DAY
) -> Dict[str, Any]:
    """Строка таблицы для кандидата

    Сбой считается ошибкой. Задержки и расход считаются по удачным вызовам
    """
    done = [r for r in rows if not r.failed]
    total = Usage()
    for r in done:
        total = total + r.usage
    per = lambda value: value / max(1, len(done))
    latencies = [r.latency_s for r in done]
    ttfts = [r.ttft_s for r in done if r.ttft_s is not None]
    return {
        "кандидат": name,
        "точность": sum(r.ok for r in rows) / max(1, len(rows)),
        "p50, с": percentile(latencies, 0.5),
        "p95, с": percentile(latencies, 0.95),
        "первый токен p50, с": percentile(ttfts, 0.5),
        "токенов на обращение": per(total.total_tokens),
        "входных на обращение": per(
            total.input_tokens + total.cache_read_tokens + total.cache_write_tokens
        ),
        "взвешенных на обращение": per(total.weighted),
        "цена за 1000, у.е.": 1000 * per(total.cost),
        "взвешенных в месяц, млн": per(total.weighted) * flow_per_day * 30 / 1e6,
        "цена в месяц, у.е.": per(total.cost) * flow_per_day * 30,
        "обрезано": sum(r.truncated for r in done),
        "сбоев": len(rows) - len(done),
    }


def table(rows: List[Dict[str, Any]]) -> str:
    """Таблица в формате Markdown"""
    heads = list(rows[0])
    fmt = lambda v: ("%.3f" % v if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    lines += ["| " + " | ".join(fmt(r[h]) for h in heads) + " |" for r in rows]
    return "\n".join(lines)


def error_report(name: str, rows: List[Row]) -> str:
    """Ошибки кандидата с эталоном, ответом модели и текстом обращения"""
    errors = [row for row in rows if not row.ok]
    count = len(errors)
    word = "ошибка" if count == 1 else "ошибки" if 2 <= count <= 4 else "ошибок"
    lines = ["### %s: %d %s из %d" % (name, count, word, len(rows))]
    if not errors:
        return lines[0] + "\n\nОшибок нет."

    lines += [
        "",
        "| обращение | эталон | ответ | текст |",
        "|---|---|---|---|",
    ]
    for row in errors:
        predicted = "сбой" if row.failed else row.predicted or "не распознано"
        text = row.text.replace("|", "\\|").replace("\n", " ")
        lines.append(
            "| %s | %s | %s | %s |"
            % (row.id, row.expected or "—", predicted, text)
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=30, help="сколько обращений из dev взять")
    ap.add_argument("--only", default=None, help="имя одного кандидата")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument(
        "--flow", type=int, default=FLOW_PER_DAY, help="обращений в день для прогноза"
    )
    args = ap.parse_args()
    rows = tickets("dev", args.n)
    llm = LLM(cache=False)  # задержку мерим без кэша
    chosen = [c for c in CANDIDATES if args.only in (None, c.name)]
    started = time.time()

    async def run_all() -> List[List[Row]]:
        return [
            await run_candidate(llm, cand, rows, args.concurrency) for cand in chosen
        ]

    results, diagnostics, spent = [], [], 0.0
    for cand, done in zip(chosen, asyncio.run(run_all())):
        spent += sum(r.usage.weighted for r in done)
        results.append(summarize(cand.name, done, args.flow))
        diagnostics.append(error_report(cand.name, done))
    report = table(results)
    print(report)
    footer = (
        "\nМодель %s, обращений %d, прогноз на %d обращений в день. "
        "Время замера %.0f с, потрачено взвешенных токенов: %.0f."
        % (llm.cfg.model, len(rows), args.flow, time.time() - started, spent)
    )
    print(footer)
    errors = "## Разбор ошибок\n\n" + "\n\n".join(diagnostics)
    print("\n" + errors)
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    (RUNS / "s1_bench.md").write_text(
        "# Замер кандидатов, %s\n\n%s\n%s\n\n%s\n"
        % (stamp, report, footer, errors),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
