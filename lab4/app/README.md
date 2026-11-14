# Навчальний AI-Помічник (Лабораторна робота 4)

Мінімалістичний застосунок для роботи з локальною мовною моделлю через OpenAI-сумісний API (`Ollama` / `llama-server`). Перетворює лекційні нотатки на картки для повторення («Питання — Відповідь»), формує стислі конспекти, пояснює помилки в коді/логах і підтримує прямий діалог.

## Структура проєкту

```text
/lab4/
├── local_models_demo.ipynb   # Демо-ноутбук із відповідями на питання 0.1 та звітом 1.8
├── bench_results.csv         # Результати навантажувального тестування (before та after)
├── app/
│   ├── index.html            # Односторінковий вебзастосунок (HTML + CSS + JS, без зайвих фреймворків)
│   ├── cli.py                # Мінімалістичний CLI-клієнт на Python (бібліотека openai)
│   ├── test_scenarios.py     # Скрипт автоматичного прогону 8 тестових сценаріїв (розділ 1.7)
│   ├── prompts_10.txt        # 10 власних промптів для перевірки якості (розділи 1.4–1.5)
│   ├── bench.py              # Скрипт навантажувального тестування з демо-матеріалів
│   ├── stream_demo.py        # Скрипт одиничного стрімінгового запиту з демо-матеріалів
│   └── README.md             # Інструкція запуску та опис параметрів генерації
└── demo/
    └── README.md             # Папка для відео демонстрації та скріншотів роботи застосунку
```

## 1. Що потрібно встановити

1. **Рантайм [Ollama](https://ollama.com/download)** (для Linux на ноутбуці ASUS VivoBook M3502QA):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. **Python 3.10+** та пакет [`openai`](https://pypi.org/project/openai/) (для запуску `cli.py`, `bench.py`, `test_scenarios.py`):
   * **Рекомендовано (через віртуальне середовище `venv`, працює на всіх ОС включно з Ubuntu/Linux без помилки PEP 668 `externally-managed-environment`):**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate   # для Windows: .venv\Scripts\activate
     pip install openai
     ```
   * **Або швидко без створення середовища (якщо встановлено менеджер [`uv`](https://docs.astral.sh/uv/)):**
     ```bash
     uv run --with openai python3 cli.py --model qwen3:1.7b --interactive
     ```

## 2. Завантаження моделей

Запустіть сервер Ollama в окремому вікні термінала та завантажте початкову модель (`qwen3:1.7b` для вимірювання `before`) і оптимізовану легшу модель (`qwen3:0.6b` для вимірювання `after`):

```bash
ollama serve
```
У другому вікні термінала:
```bash
ollama pull qwen3:1.7b
ollama pull qwen3:0.6b
```

## 3. Проведення вимірювань (`before` та `after`) для `bench_results.csv`

Перейдіть у теку `/lab4/app`:

1. **Замір до оптимізації (`before`)** — модель `qwen3:1.7b` зі стандартними налаштуваннями сервера (`OLLAMA_NUM_PARALLEL=1`):
   ```bash
   python3 bench.py --base-url http://localhost:11434/v1 --model qwen3:1.7b --reasoning-effort none --prompts prompts_10.txt --label before --output ../bench_results.csv
   ```
2. **Оптимізація (`after`)** — перезапустіть сервер Ollama з 4 паралельними слотами та обмеженим контекстом (`OLLAMA_NUM_PARALLEL=4 OLLAMA_CONTEXT_LENGTH=2048 ollama serve`) і протестуйте оптимізовану конфігурацію (для `qwen3:1.7b` та легшої моделі `qwen3:0.6b`):
   ```bash
   python3 bench.py --base-url http://localhost:11434/v1 --model qwen3:1.7b --reasoning-effort none --prompts prompts_10.txt --label after_parallel4 --output ../bench_results.csv
   python3 bench.py --base-url http://localhost:11434/v1 --model qwen3:0.6b --reasoning-effort none --prompts prompts_10.txt --label after_0.6b --output ../bench_results.csv
   ```

## 4. Як запустити власний застосунок

### Варіант А: Найпростіший вебінтерфейс (`index.html`)
У теці `/lab4/app` запустіть вбудований HTTP-сервер Python (щоб браузер не блокував запити через CORS при відкритті з диска):
```bash
python3 -m http.server 8000 --bind 127.0.0.1
```
Відкрийте у браузері адресу: **`http://localhost:8000/index.html`**

### Варіант Б: Термінальний CLI-клієнт (`cli.py`) та автотести (`test_scenarios.py`)
```bash
# Генерація карток із конспекту у терміналі:
python3 cli.py --model qwen3:1.7b --mode flashcards --prompt "Квантизація зменшує розрядність ваг моделі..."

# Інтерактивний режим чату у терміналі:
python3 cli.py --model qwen3:1.7b --interactive

# Автоматичний прогін усіх 8 тестових сценаріїв розділу 1.7:
python3 test_scenarios.py --model qwen3:1.7b --output scenarios_results.md
```

## 5. Обґрунтування параметрів генерації (Розділ 1.6)

* **`temperature = 0.2`**: обрано низьку температуру, оскільки головна задача застосунку — точне структурування конспектів, формування карток «Питання — Відповідь» і розбір помилок у коді, де потрібна фактична точність і мінімум галюцинацій.
* **`max_tokens = 400`** (обов'язковий ліміт токенів відповіді): захищає застосунок від нескінченного зациклення малої моделі (repetition loop) та гарантує лаконічність карток і тез.
* **`reasoning_effort = "none"` (`think = False`)**: вимикає генерацію прихованого ланцюжка міркувань у моделях сімейства Qwen3, завдяки чому час до появи першого токена (`TTFT`) на процесорі AMD Ryzen 5 5600H скорочується з ~8–15 с до ~0.4–1.2 с.
* **Довжина контексту (`num_ctx = 2048`)**: достатня для обробки конспекту однієї лекції (до ~1200–1500 слів) і водночас економить оперативну пам'ять під KV-кеш при паралельних запитах.
