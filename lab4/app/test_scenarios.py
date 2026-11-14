"""Автоматизована перевірка 10 промптів якості (розділи 1.4–1.5) та 8 тестових сценаріїв (розділ 1.7).

Запускає запити до локального OpenAI-сумісного сервера і зберігає звіт із відповідями моделі,
часом TTFT та швидкістю генерації (tokens/s) у файл Markdown.

Приклад запуску:
    python test_scenarios.py --model qwen3:1.7b --output quality_before.md
    python test_scenarios.py --model qwen3:0.6b --output quality_after.md
"""

import argparse
import time
from openai import OpenAI

SCENARIOS = [
    {
        "id": "Сценарій 1 (Типовий — Генерація карток із конспекту)",
        "system": "Ти — навчальний асистент. Створи 3 короткі картки «Питання — Відповідь» за текстом конспекту.",
        "prompt": (
            "Квантизація — це зменшення розрядності ваг моделі (наприклад, з 16 біт у fp16/bf16 до 8 або 4 біт у int8/int4). "
            "Формат bf16 має такий самий діапазон, як fp32 (8 біт експоненти), тому не переповнюється при великих числах, "
            "а fp16 має вищу точність (10 біт мантиси), але діапазон лише до 65504."
        ),
        "expected": "3 структуровані картки «Питання — Відповідь» про суть квантизації та різницю між bf16 і fp16.",
        "max_tokens": 350,
    },
    {
        "id": "Сценарій 2 (Типовий — Стислий конспект)",
        "system": "Виділи з тексту рівно 4 головні тези короткими маркованими пунктами українською мовою.",
        "prompt": (
            "Під час роботи локального сервера мовної моделі вимірюють три основні показники швидкості: "
            "TTFT (Time To First Token) — час від надсилання запиту до появи першого токена відповіді; "
            "TPOT або tokens/s — швидкість генерації наступних токенів для одного користувача; "
            "Throughput — сумарна кількість токенів за секунду для всіх паралельних клієнтів. "
            "Замість середнього арифметичного використовують перцентилі p50 та p95."
        ),
        "expected": "4 марковані пункти з точним поясненням TTFT, tokens/s, Throughput та перцентилів p50/p95.",
        "max_tokens": 300,
    },
    {
        "id": "Сценарій 3 (Типовий — Пояснення помилки в логах)",
        "system": "Коротко поясни причину помилки в логу та як її виправити в коді PyTorch/Transformers.",
        "prompt": (
            "ValueError: Bfloat16 is only supported on GPUs with compute capability of at least 8.0. "
            "Your Tesla T4 GPU has compute capability 7.5."
        ),
        "expected": "Пояснення, що T4 (Turing, 7.5) не має апаратної підтримки bf16 (потрібен Ampere 8.0+), і порада використати torch.float16 (fp16).",
        "max_tokens": 300,
    },
    {
        "id": "Сценарій 4 (Типовий — Генерація коду Python)",
        "system": "Напиши чистий код функції Python з коротким прикладом використання.",
        "prompt": "Напиши функцію percentile(values, q) на чистому Python для обчислення q-го перцентиля списку чисел.",
        "expected": "Робоча функція Python із сортуванням списку та лінійною інтерполяцією між сусідніми індексами.",
        "max_tokens": 300,
    },
    {
        "id": "Сценарій 5 (Складний — Багатокроковий розрахунок бюджету заліза)",
        "system": "Розв'яжи задачу покроково й дай точні числові відповіді.",
        "prompt": (
            "Ноутбук має 5.7 ГіБ вільної оперативної пам'яті DDR4-3200 з пропускною здатністю 51.2 ГБ/с. "
            "Ваги моделі qwen3:1.7b (Q4_K_M) займають 1.28 ГБ, а запас на KV-кеш і рантайм становить 30%. "
            "1) Скільки всього пам'яті потрібно моделі з запасом і чи вміститься вона? "
            "2) Яка теоретична стеля швидкості генерації (tokens/s) за формулою 51.2 / 1.28?"
        ),
        "expected": "1) 1.28 * 1.3 = 1.664 ГБ (вміщується у 5.7 ГіБ з запасом); 2) 51.2 / 1.28 = 40 tokens/s.",
        "max_tokens": 350,
    },
    {
        "id": "Сценарій 6 (Крайовий — Запит із хибною передумовою)",
        "system": "Ти — точний технічний асистент. Якщо у питанні користувача є фактична помилка, ввічливо виправ її.",
        "prompt": "Чому при встановленні temperature = 0 мовна модель починає максимально фантазувати і генерувати випадкові слова?",
        "expected": "Виправлення хибної передумови: при temperature = 0 випадковість вимикається (greedy decoding) і модель завжди обирає найімовірніший токен.",
        "max_tokens": 250,
    },
    {
        "id": "Сценарій 7 (Крайовий — Перевірка обов'язкового ліміту max_tokens)",
        "system": "Виконуй інструкцію користувача.",
        "prompt": "Напиши підряд усі числа від 1 до 500 словами українською мовою через кому.",
        "expected": "Генерація примусово й безпечно зупиняється на встановленому ліміті max_tokens=60 без зависання клієнта.",
        "max_tokens": 60,
    },
    {
        "id": "Сценарій 8 (Крайовий — Підрахунок літер після посимвольної розбивки промпту)",
        "system": "Порахуй уважно кожну літеру у списку й назви точне число.",
        "prompt": (
            "Порахуй, скільки разів літера «р» трапляється у послідовності: "
            "т - р - а - н - с - ф - о - р - м - е - р. Спочатку випиши кожну літеру з її номером, а потім дай підсумок."
        ),
        "expected": "Точний підрахунок після зміни промпту (розбиття слова дефісами на окремі токени): літера «р» трапляється 3 рази (позиції 2, 8, 12).",
        "max_tokens": 300,
    },
]


def run_single(client, model, system_prompt, user_prompt, max_tokens=300):
    start = time.perf_counter()
    first = None
    chunks = []
    usage = None
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
        seed=42,
        reasoning_effort="none",
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first is None:
                first = time.perf_counter()
            chunks.append(chunk.choices[0].delta.content)
        if chunk.usage:
            usage = chunk.usage
    end = time.perf_counter()
    text = "".join(chunks).strip()
    tokens = usage.completion_tokens if usage else max(1, len(text) // 4)
    ttft = (first - start) if first else (end - start)
    tps = (tokens - 1) / (end - first) if first and tokens > 1 and end > first else 0.0
    return text, ttft, end - start, tokens, tps


def main():
    parser = argparse.ArgumentParser(description="Перевірка тестових сценаріїв та 10 промптів якості.")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--output", default="scenarios_results.md")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="local", timeout=90.0)
    lines = [f"# Результати тестування моделі `{args.model}`\n"]

    for sc in SCENARIOS:
        print(f"Виконується {sc['id']}...")
        text, ttft, latency, tokens, tps = run_single(
            client, args.model, sc["system"], sc["prompt"], sc["max_tokens"]
        )
        lines.append(f"## {sc['id']}")
        lines.append(f"- **Промпт:** {sc['prompt']}")
        lines.append(f"- **Очікуваний результат:** {sc['expected']}")
        lines.append(f"- **Метрики:** TTFT = {ttft:.2f} с, затримка = {latency:.2f} с, токенів = {tokens}, швидкість = {tps:.1f} tokens/s")
        lines.append(f"- **Відповідь моделі:**\n\n```text\n{text}\n```\n")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Звіт збережено у {args.output}")


if __name__ == "__main__":
    main()
