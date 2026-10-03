# ASSISTANT — Монгол хэлээр яагаад компьютер зүйл хийх / Speak Mongolian, the PC acts

**Төлөв / status (2026-09-28):** the pieces are written, and the server started successfully
once in a test (`selftest-server.log` shows TTS and STT both loaded). The full live run —
you talk, it transcribes, it acts, it talks back — has **not** been done yet with a real
microphone. Read the phrases below as "built and checked against the code", not as
"tested out loud".

## 1. What it does / Юу хийх вэ

You speak Mongolian. The assistant records you, sends the audio to a small local server on
your own computer, the server turns the audio into written Mongolian, that text is matched
against a list of known commands, the command runs (a file opens, a web page opens, a
program ends), and the answer is read out loud. Everything runs on this PC. Nothing goes
to the internet except the search page you ask for.

## 2. What it CANNOT do / Юу чадахгүй вэ (read this first)

1. **It is not instant.** No Mongolian speech recogniser works while you are still talking,
   so the real order is: **you talk → you stop and pause → it acts.** The listener waits
   about 0.65 s of silence before it decides you have finished, then it transcribes, then
   it acts, then it speaks. On the graphics card (GPU) recognition takes roughly 1-2 s;
   with the launcher that ships in this folder it is forced onto the CPU, so expect
   several seconds. Never call it "instant".
2. **It only knows the commands in the table below.** It is not a chatbot. A sentence with
   no command word gets "Ойлгохгүй байна" (did not understand).
3. **It does not answer questions.** "Internet search" opens a DuckDuckGo results page in
   your browser. You read the answer yourself; it is never read to you.
4. **"Close" really kills the program** — no "save your work?" question. Unsaved documents
   are lost.
5. **It is built for one person in a quiet room.** A TV or music gets transcribed as your
   command.
6. **The voice that answers you is not your own trained voice yet.** See section 6.

## 3. How to start it / Аажлуулах

Open PowerShell in this folder (`C:\Users\hi\Documents\mongolian-voice`). First time:

1. `START-SERVER.ps1` in one window. This is the speech engine: voice → text (Whisper
   medium) and text → a voice. It listens only on this computer at `127.0.0.1:8771`;
   nothing is published to the internet. **The first start takes a few minutes** — the
   Whisper model is being loaded into memory.
2. `START-ASSISTANT.ps1` in a second window.

After that the server stays up, so you only run `START-ASSISTANT.ps1` from then on (add
`-NoServer` to stop it starting the server for you). The launcher checks the server,
prints what it found, builds a list of your file names, then starts listening. Ctrl+C stops.

Extra flags go after the script name, e.g. `START-ASSISTANT.ps1 --dry-run`:

| Flag | What it does |
|---|---|
| `--dry-run` | says what it would do and does nothing: no file opens, no program ends |
| `--no-speak` | no spoken answer, text only in the window |
| `--verbose` | extra timing lines per step |
| `--device N` | use microphone number N instead of the Windows default |
| `--no-vad` | push-to-talk: treats every chunk as a command, no silence detection |
| `--roots "C:\Users\hi\Videos"` | also search that folder for files (repeatable) |

## 4. What you can say / Юу хэлж болох вэ

Say the command word plus a short name. Every sentence you say is printed in the window
before the action runs, so you can see exactly what it heard.

| What you want | Say in Mongolian (Cyrillic) | English |
|---|---|---|
| Open a file | Хөтөлөгчийн хуудгийг нээ · Төлөвлөгөө нээ · Гэрээ нээ | the bookmarks file / "Plan" / "Contract" |
| Open a folder | `data` нээ · `checkpoints` нээ | a folder inside this project folder |
| Open an app | Блокнот нээ · Ворд нээ · Эксэл нээ · Хөтөлөгч нээ · Тооцоологч нээ | Notepad / Word / Excel / browser / Calculator |
| Open a website | Ютюб нээ · Фейсбук нээ · Гитхаб нээ · Гэйл нээ · Википедиа нээ | YouTube / Facebook / GitHub / Gmail / MN Wikipedia |
| Close a program | Ворд хаа · Хөтөлөгч хаа · Блокнот хаа | ends it immediately, no save prompt |
| Search the internet | Монголын цаг агаарыг хай · Фермийн теорем үз | opens a DuckDuckGo page in your browser |
| Start a slideshow | Төлөвлөгөө танилцуул · Төлөвлөгөө слайд | slide show of that .pptx / .pdf |
| List matching files | Төлөвлөгөө файлууд байна · Excel файлууд байна | prints matching names in the window, not spoken |
| Stop listening | Сонсохыг зогсоо | stops and exits |
| What can I do | Тусламж | reads out the list of abilities |

Four rules that save you failures:

- **Files are searched only in** Documents, Desktop, Downloads and `OneDrive\Documents`.
  Anything else is invisible to it (use `--roots`).
- **For apps, use the plain name.** `Ворд нээ` works; `Ворд-ийг нээ` does not — that ending
  makes it look for a *file* with that exact name instead.
- **A slideshow** needs a real `.ppt`, `.pptx`, `.odp`, `.pdf` or `.key` file whose name
  matches. Say `Танилцуулга` alone and it asks you for a file name.
- **Do not pad the sentence.** "Төлөвлөгөө нээ" works; "Төлөвлөгөө файлыг нээж өгөөч" does
  not — the extra words stay in the name it searches for.

## 5. Teaching it new apps and websites / Шинэ программ, сайт зааварлах

Open `actions.py` and look at the tables at the top (lines 39-81). Each is a plain list of
`Mongolian word: what it means`. One line each, save, restart the assistant — that's all:

```python
APP_ALIASES = { "фаерфокс": "firefox", "firefox": "firefox", ... }   # "Фаерфокс нээ"
SITE_ALIASES = { "спорт": "https://www.mongolkhuree.mn", ... }       # "Спорт нээ"
CLOSEABLE = { "фаерфокс": "firefox", ... }                           # "Фаерфокс хаа"
```

- Add the word twice, Mongolian and English, so both work. If the program is not already
  in `APP_COMMANDS`, add it there too: `"firefox": "start \"\" firefox",`
- Copy the spelling from what the assistant prints in the window — it must match what
  Whisper writes in Cyrillic. A whole new *kind* of action is not a one-liner: that means
  editing `parse()` and `resolve()` in the same file.
- Extra folders need no edit: `.\venv\Scripts\python.exe assistant.py --roots "C:\Users\hi\Videos"`.
  The index holds at most 40,000 files and skips `.git`, `venv`, `node_modules`,
  `__pycache__`, `build`, `dist`, `target`, `.gradle`, `site-packages`. It stores file
  **names** only, never what is inside a file.

## 6. Known limits / Мэдэх ёстой хязгаарууд

1. **File matching is a guess.** Exact name → exact match. Part of the name → substring
   match. Otherwise it picks the name that looks most like what you said and tells you how
   sure it is ("~60% таамаглал"). It can open the wrong file.
2. **One speaker, quiet room.** The silence detector (it decides you finished after
   ~0.65 s of quiet) is fooled by a TV, music, or a second voice.
3. **GPU vs CPU.** `START-SERVER.ps1` deliberately sets `MN_TTS_DEVICE=cpu` and
   `MN_STT_DEVICE=cpu` so a training run keeps the graphics card. Delete those two lines
   when nothing is training and recognition gets much faster. The server picks the GPU by
   itself when more than about 2.2 GB of video memory is free.
4. **The speaking voice is Meta's MMS fallback, not a trained one.** There is no
   `piper_voice\*.onnx` and no `model\` folder, so the server loads `facebook/mms-tts-mon`
   (confirmed in `selftest-server.log`). It understands Mongolian and sounds flat.
5. **Voice training is stopped at epoch 0.** Newest checkpoint:
   `lightning_logs\version_1\checkpoints\epoch=0-step=1200.ckpt`. The README expects 15-30
   epochs before the voice is usable, and `run_till_5am.log` shows the overnight run never
   got going. **Do not run `EXPORT-VOICE.ps1` yet** — it would export a voice that has
   barely learned anything. Resume with `START-TRAIN.ps1` instead.
6. **A program that is not installed will not open,** and the assistant still says it is
   opening it (Word, Excel, PowerPoint need Office; `Код нээ` needs VS Code).

## 7. Troubleshooting / Асуудал гарвал

| What you see | What it means / what to do |
|---|---|
| `Чиглэлгүй дуу гарлаа` (no speech in that clip) | Too quiet, or the wrong mic. Check it is not muted (Settings → System → Sound → Input; on many laptops a keyboard key with a mic icon), then try `--device 1`. |
| `Ойлгохгүй байна` (did not understand) | The sentence had no command word. Say one phrase from the table in section 4. |
| `"X"-г олж чадсангүй` (could not find X) | The name after the command matched no file. Check the printed spelling, and check the file is in one of the four searched folders. |
| `(үйлдэл хийгдэхгүй / nothing was changed)` | It understood you but could not do it — usually the name matched nothing. |
| `voice server is not reachable`, exit code 2 | The server is not running. Run `START-SERVER.ps1`, wait for it to print `ready`, then start the assistant again. |
| `ERROR: voice server did not answer /health after 180 s` | The server took over 3 minutes. First start is slow; if it repeats, run `.\venv\Scripts\python.exe server.py` by hand in that window and read the last lines — it prints the real reason. |
| Every answer takes many seconds | Recognition is on the CPU. See section 6, item 3. |
| Cyrillic shows as `□□□` or `Р°Рѕ` | The console is not reading UTF-8. Run `chcp 65001` first, or just use `START-ASSISTANT.ps1`, which sets UTF-8 for you. |
| Nothing happens when I talk at all | Run `START-ASSISTANT.ps1 --verbose`: it prints each clip, its length, the transcription time and the device used. |

## Files / Файлүүд

`actions.py` = the brain (Mongolian sentence → what to do, plus the alias tables) ·
`assistant.py` = the loop and all the flags · `listener.py` = the microphone half ·
`server.py` = the speech engine on `127.0.0.1:8771` · `START-SERVER.ps1` and
`START-ASSISTANT.ps1` = the two launchers · `ASSISTANT.md` = this file.
