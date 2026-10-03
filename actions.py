"""Mongolian command -> local action.

Turns a sentence of Mongolian speech into something the machine actually does:
open a file / folder / website / app, close something, run a web search, start a
slideshow.  Pure text in, Action out -- no audio, no model, no network (except
the search fallback URL), so it is trivially testable.

    from actions import parse, run
    act = parse("Word-ийг нээ")
    run(act)
"""
from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

# Longest phrases first: "нээгээч" must win over "нээ" when both are substrings.
OPEN_WORDS = [
    "эхлээд", "нээгээч", "эхлүүл", "ажиллуул", "гарга", "үзүүл", "харуул",
    "шилжи", "соли", "нээ", "эхлэх", "заа", "сонс", "ау",
]
CLOSE_WORDS = ["хаагдах", "хаагаад", "хааж", "хаа", "тл", "зогсоо", "зогсоод"]
SEARCH_WORDS = ["хайж", "хайлт", "хай", "үз", "юу вэ", "гэж үз", "мэдээлэл"]
PRESENT_WORDS = ["танилцуул", "танилцуулга", "слайд", "презентац", "presentation"]
LIST_WORDS = ["жагсаалт", "лист", "файлууд", "юу байна", "юу байгаа"]
HELP_WORDS = ["туллах", "туслах", "туслаач", "тусламж", "юу хийж", "боломж",
              "команд"]
SLEEP_WORDS = ["зогсоо сонсох", "сонсохыг зогсоо", "сонсохгүй бол", "уншихад хээ"]

APP_ALIASES = {
    "ворд": "winword", "word": "winword", "docx": "winword",
    "эксэл": "excel", "excel": "excel",
    "пауэрпоинт": "powerpnt", "танилцуулгын программ": "powerpnt",
    "блокнот": "notepad", "notepad": "notepad",
    "тэмдэглэл": "notepad", "замет": "notepad",
    "файллах": "explorer", "проводник": "explorer", "файл manager": "explorer",
    "файллэгч": "explorer", "explorer": "explorer",
    "хэрэгсэл": "explorer", "хээрэгсэл": "explorer", "хэрэгслийн үзүүлэгч": "explorer",
    "комманд": "cmd", "cmd": "cmd", "терминал": "cmd", "powershell": "powershell",
    "код": "code", "кодод": "code", "vscode": "code", "vs code": "code",
    "хөтөлөгч": "msedge", "хром": "chrome", "browser": "msedge",
    "цагийн хуваарь": "charmap", "ажлын хуваарь": "charmap",
    "тооцоологч": "calc", "калькулятор": "calc",
}

# Which of those are real launchable commands vs. a process we must find and kill.
APP_COMMANDS = {
    "winword": "start \"\" winword", "excel": "start \"\" excel",
    "powerpnt": "start \"\" powerpnt", "notepad": "notepad",
    "explorer": "explorer", "cmd": "cmd", "powershell": "powershell",
    "code": "code", "msedge": "start \"\" msedge", "chrome": "start \"\" chrome",
    "charmap": "charmap", "calc": "calc",
}

SITE_ALIASES = {
    "ютюб": "https://youtube.com", "youtube": "https://youtube.com",
    "фейсбук": "https://facebook.com", "facebook": "https://facebook.com",
    " gmail ": "https://mail.google.com", "гэйл": "https://mail.google.com",
    "почм": "https://mail.google.com", "gmail": "https://mail.google.com",
    "гитхаб": "https://github.com", "github": "https://github.com",
    "википедиа": "https://mn.wikipedia.org",
    "учир": "https://www.google.com/search?q=site%3Audmb.mn",
    "хөгжүүлэгч": "https://mn.wikipedia.org",
    "цаг": "https://www.timeanddate.com/worldclock/mongolia",
}

CLOSEABLE = {
    "ворд": "winword", "word": "winword", "эксэл": "excel", "excel": "excel",
    "пауэрпоинт": "powerpnt", "блокнот": "notepad", "notepad": "notepad",
    "код": "code", "code": "code", "хөтөлөгч": "msedge", "хром": "chrome",
    "browser": "msedge", "файллах": "explorer", "проводник": "explorer",
    "комманд": "cmd", "cmd": "cmd", "powerpoint": "powerpnt", "teams": "teams",
}

SEARCH_URL = "https://duckduckgo.com/?q={}"

# Words that carry no search meaning once the command verb is stripped.
FILLER = {
    "байна", "байгаа", "юм", "үү", "тэр", "энэ", "дахь", "болох", "болоо",
    "бод", "хэрэг", "хэрэгтэй", "миний", "түүний", "хамгийн", "гэж",
    "миний", "хэний", "нэг", "хоёр", "гэж", "юу", "аль", "ыг", "ийм",
}

# trailing politeness verbs -- "блокнот нээж өг" means just "open notepad"
POLITE = ["өг", "хэл", "даа", "з", "юм", "ууд", "аа", "руу", "руугаар"]

DEFAULT_ROOTS = [
    Path.home() / "Documents",
    Path.home() / "Desktop",
    Path.home() / "Downloads",
    Path.home() / "OneDrive" / "Documents",
]
SLIDE_EXT = {".ppt", ".pptx", ".odp", ".pdf", ".key"}
INDEX_LIMIT = 40000


# --------------------------------------------------------------------------
# data types
# --------------------------------------------------------------------------

@dataclass
class Intent:
    name: str                      # open | close | search | present | list | help | sleep
    target: str = ""               # what to open / search for
    raw: str = ""                  # the sentence as heard
    note: str = ""                 # ambiguity / fallback explanation


@dataclass
class Action:
    kind: str                      # file | folder | url | app | search | process | none
    value: str = ""
    extra: str = ""
    reply: str = ""                # Mongolian sentence to speak back
    ok: bool = False               # a no-op must never read as success
    candidates: list = field(default_factory=list)


# --------------------------------------------------------------------------
# text normalisation + intent parsing
# --------------------------------------------------------------------------

def normalize(text: str) -> str:
    text = text.strip().lower().replace("ё", "е")
    # park URLs behind placeholders so punctuation stripping cannot turn
    # "example.com" into "example com"
    urls = re.findall(r"\b(?:https?://|www\.)\S+", text)
    for i, u in enumerate(urls):
        text = text.replace(u, f"\x00{i}\x00")
    text = re.sub(r"[\s,.:;!?\"'()\[\]]+", " ", text)
    for i, u in enumerate(urls):
        text = text.replace(f"\x00{i}\x00", u)
    return re.sub(r"\s+", " ", text).strip()


def _strip_words(text: str, words: list[str]) -> str:
    """Remove leading command vocabulary, longest match first.

    Mongolian glues a converb onto the verb ("нээ" -> "нээж", "хаа" -> "хаагаад"),
    so every verb also matches with one of those endings.
    """
    suffix = r"(?:ж|аад|ад|ууд|уу|аар|ийн|од|оод)?"
    for w in words:
        if text.startswith(w):
            rest = text[len(w):]
            m = re.match(suffix + r"(?=$| )", rest)
            if m:
                return rest[m.end():].strip()
    for w in words:
        text = re.sub(rf"(?:^| ){re.escape(w)}{suffix}(?=$| )", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_target(text: str) -> str:
    out = text
    for w in sorted(FILLER, key=len, reverse=True):
        out = re.sub(rf"(?:^| ){re.escape(w)}(?=$| )", " ", out)
    for w in POLITE:
        out = re.sub(rf"(?:^| ){re.escape(w)}(?=$| )", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    # grammatical endings, as whole suffixes -- never a bare character strip,
    # which would eat the first letter of a spoken name ("улаанбаатар" -> "лаанбаатар")
    return re.sub(r"(?:ыг|ийг|ыгхүү|ийн|ууд)$", "", out).strip(" ?!.,")


def parse(text: str) -> Intent | None:
    """Sentence -> Intent.  None when nothing in it is a command."""
    raw = text.strip()
    norm = normalize(text)
    if not norm:
        return None

    if any(w in norm for w in SLEEP_WORDS):
        return Intent("sleep", "", raw)

    if any(w in norm for w in HELP_WORDS):
        return Intent("help", "", raw)

    if any(w in norm for w in LIST_WORDS):
        body = _strip_words(norm, LIST_WORDS)
        return Intent("list", _clean_target(body), raw)

    if any(w in norm for w in PRESENT_WORDS):
        body = _strip_words(_strip_words(norm, PRESENT_WORDS), OPEN_WORDS)
        return Intent("present", _clean_target(body), raw)

    body = _strip_words(norm, OPEN_WORDS)
    if body != norm and body:
        return Intent("open", _clean_target(body), raw)

    body = _strip_words(norm, SEARCH_WORDS)
    if body != norm:
        return Intent("search", _clean_target(body), raw)

    body = _strip_words(norm, CLOSE_WORDS)
    if body != norm:
        # bare "зогсоо" = stop everything we could be running
        return Intent("close", _clean_target(body) or "бүх", raw)

    return None


# --------------------------------------------------------------------------
# file index
# --------------------------------------------------------------------------

class FileIndex:
    """Stem -> paths, built once and reused (mtime-checked so it self-heals)."""

    def __init__(self, roots: list[Path] | None = None, max_files: int = INDEX_LIMIT):
        # `roots or DEFAULT_ROOTS` would treat an explicitly empty list as
        # "use the defaults" and silently walk the whole user profile
        self.roots = [Path(r) for r in (DEFAULT_ROOTS if roots is None else roots)
                      if Path(r).exists()]
        self.max_files = max_files
        self._table: dict[str, list[str]] = {}
        self._stamp: tuple = ()
        self._n = 0

    def _root_stamp(self):
        return tuple(sorted((str(r), int(r.stat().st_mtime)) for r in self.roots
                            if r.exists()))

    def table(self, rebuild: bool = False) -> dict[str, list[str]]:
        stamp = self._root_stamp()
        if rebuild or stamp != self._stamp:
            table: dict[str, list[str]] = {}
            n = 0
            for root in self.roots:
                for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
                    dirnames[:] = [d for d in dirnames
                                   if d not in {"node_modules", ".git", "venv", ".venv",
                                                "__pycache__", "site-packages", ".gradle",
                                                "dist", "build", "target"}]
                    for fn in filenames:
                        n += 1
                        if n > self.max_files:
                            break
                        stem = Path(fn).stem.lower()
                        if not stem:
                            continue
                        table.setdefault(stem, []).append(str(Path(dirpath) / fn))
            self._table, self._stamp, self._n = table, stamp, n
        return self._table

    def find(self, query: str) -> tuple[list[str], float, str]:
        """-> (paths, score 0..1, how).  Empty list when nothing is close enough."""
        q = normalize(query).replace(" ", "")
        if not q:
            return [], 0.0, "empty"
        table = self.table()

        if q in table:
            return table[q], 1.0, "exact"
        joined: dict[str, list[str]] = {}
        for stem, paths in table.items():
            s = stem.replace(" ", "")
            if q in s or s in q:
                joined.setdefault(stem, []).extend(paths)
        if joined:
            best = max(joined, key=lambda s: len(s))
            return joined[best], 0.9, "partial"

        names = list(table)
        close = difflib.get_close_matches(q, names, n=3, cutoff=0.62)
        if close:
            best = close[0]
            return table[best], difflib.SequenceMatcher(None, q, best).ratio(), "fuzzy"
        return [], 0.0, "none"


# --------------------------------------------------------------------------
# resolution: intent -> concrete action
# --------------------------------------------------------------------------

def _search_url(query: str) -> str:
    return SEARCH_URL.format(urllib.parse.quote_plus(query))


def resolve(intent: Intent, index: FileIndex) -> Action:
    t = intent.target.strip()

    # these carry no target by design, so they must be answered before the
    # empty-target guard below would swallow them
    if intent.name == "help":
        return Action("help", reply=HELP_TEXT, ok=True)

    if intent.name == "sleep":
        return Action("none", value="sleep", reply="Сонсохыг зогсооё.", ok=True)

    if intent.name == "present" and not t:
        return Action("none", reply="Танилцуулгын файлыг нэгээр нэрлээрэй.",
                      ok=False)

    if not t:
        return Action("none", reply="Юуг нээхийг хэлээрэй.",
                      ok=False, candidates=[])

    if intent.name == "list":
        return Action("list", value=t, reply="Файлуудыг жагсааж байна.", ok=True)

    if intent.name == "search":
        return Action("search", value=_search_url(t),
                      reply=f"“{t}” гэж интернетэд хайж байна.", ok=True)

    if intent.name == "close":
        if t in ("бүх", "бүгд", "all"):
            return Action("process", value="__all__", ok=True,
                          reply="Нээлттэй бүх програмыг хаая.")
        proc = None
        for k, v in CLOSEABLE.items():
            if k in t:
                proc = v
                break
        if proc is None:
            paths, score, _ = index.find(t)
            proc = Path(paths[0]).name if paths else None
        if proc is None:
            return Action("none", reply=f"“{t}”-г юу гэж ойлгохгүй байна.",
                          ok=False)
        return Action("process", value=proc, ok=True,
                      reply=f"{proc} програмыг хаая.")

    # ---- open / present ----
    if intent.name == "present":
        paths, score, how = index.find(t)
        slide = None
        if paths:
            slide = next((p for p in paths
                          if Path(p).suffix.lower() in SLIDE_EXT), None)
        if slide is None and not t:
            return Action("none", reply="Танилцуулгын файлыг нэгээр нэрлээрэй.",
                          ok=False)
        if slide is None:
            return Action("search", value=_search_url(t + " танилцуулга"),
                          reply=f"“{t}” танилцуулга олдсонгүй. Интернетэд хайна.",
                          ok=False)
        return Action("slides", value=slide, ok=True,
                      reply=f"{Path(slide).name} танилцуулгыг нээж байна.")

    if re.match(r"^(https?://|www\.)", t):
        url = t if t.startswith("http") else "https://" + t
        return Action("url", value=url, reply=f"{url} гэж нээнэ.", ok=True)

    # apps are named exactly ("Ворд", "ворд-ийг"), sites are matched inside a
    # longer phrase ("фейсбук руу очих") -- and an exact app name has to beat a
    # site alias that merely appears somewhere in the sentence
    app = APP_ALIASES.get(t) or APP_ALIASES.get(re.sub(r"[-ийгийнгаар]*$", "", t).strip())
    if app:
        return Action("app", value=app, reply=f"{t} програмыг нээж байна.", ok=True)

    site = next((url for k, url in SITE_ALIASES.items() if k in t), None)
    if site:
        return Action("url", value=site, reply=f"{t} руу нээнэ.", ok=True)

    direct = Path(t)
    if direct.exists():
        kind = "folder" if direct.is_dir() else "file"
        return Action(kind, value=str(direct.resolve()), ok=True,
                      reply=f"{direct.name} нээж байна.")

    paths, score, how = index.find(t)
    if paths:
        extra = ""
        if len(paths) > 1:
            extra = f" ({len(paths)} ижил нэртэй, эхнийхийг нээлээ)"
        elif how != "exact":
            extra = f" (~{score:.0%} таамаглал)"
        return Action("file", value=paths[0], extra=extra, ok=True,
                      reply=f"{Path(paths[0]).name} нээж байна{extra}.")

    # a bare filename that exists nowhere -> try as a search anyway
    if " " in t and len(t) > 3:
        return Action("search", value=_search_url(t),
                      reply=f"“{t}” олдсонгүй. Интернетэд хайж байна.",
                      ok=False)
    return Action("none", reply=f"“{t}”-г олж чадсангүй.",
                  ok=False, candidates=[])


HELP_TEXT = ("Файл, хэрэгсэл, вэб сайт, программ нээх, хаах, "
             "интернетэд хайх, танилцуулга эхлүүлэх боломжтой.")


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def run(action: Action, dry_run: bool = False) -> Action:
    """Carry out the action.  Sets action.ok / appends to action.reply on failure.

    dry_run short-circuits BEFORE any side effect: it must never open a file,
    start an app, or kill a process.  Tagging an action after the fact is not
    a dry run -- that is how a test suite ends up launching the user's Word.
    A dry run reports the ok flag that resolve() already decided and never
    upgrades a "nothing to do" answer into a success.
    """
    passive = ("list", "none", "help")
    if action.kind not in passive + ("file", "folder", "url", "slides",
                                     "app", "search", "process"):
        action.ok = False
        action.reply = f"Тэгэх мөр үйлдэл байхгүй: {action.kind}"
    else:
        if action.kind == "list":
            action.extra = action.value
        if not dry_run and action.kind not in passive:
            _execute(action)
    if dry_run:
        action.extra = (action.extra + f" [DRY-RUN {action.kind}]").strip()
    return action


def _execute(action: Action) -> None:
    try:
        if action.kind in ("file", "folder"):
            os.startfile(action.value)
        elif action.kind in ("url", "search"):
            import webbrowser
            webbrowser.open(action.value)
        elif action.kind == "slides":
            subprocess.Popen(["start", "", action.value], shell=True)
        elif action.kind == "app":
            subprocess.Popen(APP_COMMANDS.get(action.value, action.value), shell=True)
        elif action.kind == "process":
            targets = (sorted(set(CLOSEABLE.values())) if action.value == "__all__"
                       else [action.value])
            if not any(_kill(proc) for proc in targets):
                action.ok = False
                action.reply += " Гэвч энэ программ ажиллаж байгаагүй бололтой."
    except Exception as e:                                   # noqa: BLE001
        action.ok = False
        action.reply = f"Алдаа: {e}"


def _kill(proc_name: str) -> bool:
    import psutil
    me = psutil.Process().pid
    killed = 0
    for p in psutil.process_iter(["name"]):
        try:
            if p.info["name"] and proc_name.lower() in p.info["name"].lower() \
                    and p.pid != me:
                p.terminate()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return killed > 0


# --------------------------------------------------------------------------
# one-call entry point
# --------------------------------------------------------------------------

class Assistant:
    """Stateful wrapper so callers keep one FileIndex instead of rebuilding it."""

    def __init__(self, roots: list[Path] | None = None):
        self.index = FileIndex(roots)

    def handle(self, text: str, dry_run: bool = False) -> tuple[Intent | None, Action]:
        intent = parse(text)
        if intent is None:
            return None, Action("none", reply="", ok=False)
        return intent, run(resolve(intent, self.index), dry_run=dry_run)


def which(name: str) -> str | None:
    return shutil.which(name)


if __name__ == "__main__":                                   # manual smoke test
    import sys
    a = Assistant()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        i, act = a.handle(line, dry_run="--dry" in sys.argv)
        print(f"> {line}\n  intent={i.name if i else None!r:9} "
              f"action={act.kind:7} {act.value}\n  -> {act.reply} {act.extra}")
