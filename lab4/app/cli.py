"""Мінімалістичний CLI-застосунок «Навчальний AI-Помічник» (Лабораторна робота 4, розділ 1.6).

Працює з локальною моделлю через OpenAI-сумісний API (Ollama, llama-server, LM Studio).
Підтримує 4 режими:
  - flashcards  : перетворює конспект або тему на картки «Питання — Відповідь»
  - summary     : формує стислий конспект із 3–5 головних тез
  - code_review : пояснює причину помилки в логах або коді та пропонує виправлення
  - chat        : прямий діалог або інтерактивний цикл у терміналі

Інструкція запуску:
  1. Запустіть Ollama:
       ollama serve
  2. Завантажте модель:
       ollama pull qwen3:1.7b
  3. Встановіть клієнтську бібліотеку:
       pip install openai
  4. Приклади виклику:
       python cli.py --mode flashcards --prompt "Квантизація зменшує розрядність ваг з fp16 до int8 або int4..."
       python cli.py --mode summary --file notes.txt
       python cli.py --interactive
"""

import argparse
import sys
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

MODE_SYSTEM_PROMPTS = {
    "flashcards": (
        "Ти — навчальний асистент студента. Перетвори наданий конспект або тему на 3–5 лаконічних "
        "карток для повторення у форматі:\nКартка N:\nПитання: ...\nВідповідь: ...\n"
        "Відповідай українською мовою, точно за фактами тексту."
    ),
    "summary": (
        "Ти — навчальний асистент. Виділи з наданого тексту рівно 3–5 найважливіших тез "
        "короткими маркованими пунктами українською мовою без води."
    ),
    "code_review": (
        "Ти — технічний асистент з програмування та ML-інженерії. Коротко поясни причину помилки "
        "у наведеному логу чи коді та покажи виправлений варіант."
    ),
    "chat": "Ти — корисний і лаконічний AI-асистент. Відповідай чітко, грамотно українською мовою.",
}


def parse_args():
        parser = argparse.ArgumentParser(
        description="Навчальний AI-Помічник (OpenAI-сумісний локальний клієнт)."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="адреса OpenAI-сумісного API",
    )
    parser.add_argument(
        "--model",
        default="qwen3:1.7b",
        help="назва локальної моделі (напр. qwen3:1.7b або qwen3:0.6b)",
    )
    parser.add_argument(
        "--api-key", default="local", help="API-ключ (локально не перевіряється)"
    )
    parser.add_argument(
        "--mode",
        choices=["flashcards", "summary", "code_review", "chat"],
        default="flashcards",
        help="режим роботи помічника",
    )
    parser.add_argument("--prompt", help="текст запиту або нотаток")
    parser.add_argument(
        "--file", help="шлях до текстового файлу з конспектом або логом"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="температура семплінгу (за замовчуванням 0.2 для мінімізації галюцинацій)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=400,
        help="обов'язковий ліміт токенів відповіді (запобіжник від зациклення малої моделі)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default="none",
        help='режим міркування ("none" вимикає thinking для швидкого TTFT на CPU)',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="фіксація зерна генератора випадкових чисел",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="таймаут очікування відповіді у секундах",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="запустити інтерактивний чат у терміналі",
    )
    return parser.parse_args()


def stream_once(client, args, messages):
    """Виконує один стрімінговий запит з індикацією очікування, обробкою помилок і підрахунком метрик."""
    extra = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    if args.seed is not None:
        extra["seed"] = args.seed

    print(
        f"⏳ Очікування відповіді від моделі '{args.model}' ({args.base_url})...",
        file=sys.stderr,
        flush=True,
    )
    start = time.perf_counter()
    first = None
    usage = None
    had_reasoning = False
    collected = []

    try:
        stream = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=True,
            stream_options={"include_usage": True},
            **extra,
        )
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                extra_fields = delta.model_extra or {}
                if extra_fields.get("reasoning") or extra_fields.get(
                    "reasoning_content"
                ):
                    had_reasoning = True
                if delta.content:
                    if first is None:
                        first = time.perf_counter()
                    print(delta.content, end="", flush=True)
                    collected.append(delta.content)
            if chunk.usage:
                usage = chunk.usage
    except APITimeoutError:
        print(
            f"\n[Помилка таймауту]: Сервер не відповів за {args.timeout:.0f} с. "
            "Спробуйте меншу модель (qwen3:0.6b) або зменште --max-tokens.",
            file=sys.stderr,
        )
        return None
    except APIConnectionError:
        print(
            f"\n[Помилка з'єднання]: Не вдалося підключитися до {args.base_url}. "
            "Перевірте, що локальний сервер запущено (наприклад, `ollama serve`).",
            file=sys.stderr,
        )
        return None
    except APIStatusError as exc:
        print(
            f"\n[Помилка сервера HTTP {exc.status_code}]: Перевірте, чи завантажена модель "
            f"`ollama pull {args.model}`. Деталі: {exc.message}",
            file=sys.stderr,
        )
        return None
    except Exception as exc:
        print(f"\n[Непередбачена помилка]: {exc}", file=sys.stderr)
        return None

    end = time.perf_counter()
    print()

    full_text = "".join(collected).strip()
    if first is None or not full_text:
        print(
            "[Помилка відповіді]: Модель не повернула жодного видимого токена у полі content. "
            "Переконайтеся, що передано --reasoning-effort none або збільште --max-tokens.",
            file=sys.stderr,
        )
        return None

    ttft = first - start
    latency = end - start
    tokens = usage.completion_tokens if usage else max(1, len(full_text) // 4)
    tok_per_s = (tokens - 1) / (end - first) if tokens > 1 and end > first else 0.0

    print(
        f"--- Метрики: TTFT = {ttft:.2f} с | Повна затримка = {latency:.2f} с | "
        f"Токенів = {tokens} | Швидкість = {tok_per_s:.1f} tokens/s ---",
        file=sys.stderr,
    )
    if had_reasoning:
        print(
            "УВАГА: Модель міркувала перед відповіддю (TTFT включає час роздумів).",
            file=sys.stderr,
        )
    return full_text


def main():
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    system_prompt = MODE_SYSTEM_PROMPTS[args.mode]

    if args.interactive:
        print(
            f"=== Інтерактивний режим ({args.mode}) з моделлю {args.model}. Введіть 'exit' для виходу. ==="
        )
        history = [{"role": "system", "content": system_prompt}]
        while True:
            try:
                user_text = input("\nВи > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nЗавершення роботи.")
                break
            if not user_text:
                continue
            if user_text.lower() in ("exit", "quit", ":q"):
                break
            history.append({"role": "user", "content": user_text})
            answer = stream_once(client, args, history)
            if answer is not None:
                history.append({"role": "assistant", "content": answer})
            else:
                history.pop()
        return

    user_content = args.prompt
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            file_text = f.read().strip()
        user_content = (
            f"{user_content}\n\n{file_text}".strip() if user_content else file_text
        )

    if not user_content:
        user_content = (
            "Квантизація — це зменшення розрядності параметрів моделі з 16 біт (fp16/bf16) "
            "до 8 або 4 біт (int8/int4), що зменшує обсяг пам'яті та пришвидшує читання ваг із RAM."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    stream_once(client, args, messages)


if __name__ == "__main__":
    main()
