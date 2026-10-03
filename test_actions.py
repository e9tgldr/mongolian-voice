"""Unit tests for actions.py -- the Mongolian sentence -> action engine.

Run:  venv/Scripts/python.exe test_actions.py        (exit 1 if any test fails)

Safety contract this file keeps:
  * every file fixture is built inside a tempfile.TemporaryDirectory() and the
    CWD is moved there too, so FileIndex -- and the Path(t).exists() probes
    inside resolve() -- can only ever see the fixture, never the real profile.
  * run() is called with dry_run=True only, and behind stubs for every channel
    it can reach out through (os.startfile, subprocess.Popen, webbrowser.open,
    actions._kill).  No browser, editor, shell or process is started, inspected
    or killed from here.

Tests named test_bug_* pin behaviour that is wrong today.  They assert what the
code actually does, not what it should do, so they break loudly the moment
actions.py is fixed instead of quietly going green.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import webbrowser
from contextlib import contextmanager
from pathlib import Path

for _stream in ("stdout", "stderr"):
    getattr(sys, _stream).reconfigure(encoding="utf-8", errors="replace")

import actions
from actions import Action, Assistant, FileIndex, Intent, normalize, parse, resolve, run

FIXTURE_FILES = ("alpha_report.docx", "beta_slides.pptx", "тайлал.pdf")


class Failure(AssertionError):
    pass


def check(cond, msg):
    if not cond:
        raise Failure(msg)


def same_path(a, b) -> bool:
    return os.path.normcase(str(Path(a))) == os.path.normcase(str(Path(b)))


@contextmanager
def fixture():
    """Isolated file tree + CWD.  The real Documents/Desktop/Downloads are never
    walked, and relative Path() lookups inside resolve() see the fixture."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name in FIXTURE_FILES:
            (tmp / name).write_text("x", encoding="utf-8")
        (tmp / "projects").mkdir()
        cwd = Path.cwd()
        os.chdir(tmp)
        try:
            yield tmp
        finally:
            os.chdir(cwd)


@contextmanager
def no_side_effects():
    """Record every call run() tries to make instead of performing it."""
    calls = []
    real = (os.startfile, subprocess.Popen, webbrowser.open, actions._kill)
    os.startfile = lambda p: calls.append(("startfile", str(p)))
    subprocess.Popen = lambda *a, **k: calls.append(("Popen", a, k))
    webbrowser.open = lambda u, *a, **k: calls.append(("open_url", str(u)))
    actions._kill = lambda name: (calls.append(("kill", name)), True)[1]
    try:
        yield calls
    finally:
        os.startfile, subprocess.Popen, webbrowser.open, actions._kill = real


# --------------------------------------------------------------------------
# normalize()
# --------------------------------------------------------------------------

NORMALIZE_CASES = [
    ("", ""),
    ("    ", ""),
    ("  ВОРД   НЭЭЖ!  ", "ворд нээж"),
    ("Хөх тэнгэр, байна.", "хөх тэнгэр байна"),
    ("Тэр ёстой сайхан юм!!!", "тэр естой сайхан юм"),
    ("ЁРӨГСӨН байна", "ерөгсөн байна"),
    ("нээ, нээ.", "нээ нээ"),
    ("блокнот\tхаа", "блокнот хаа"),
    ("(танилцуулга) нээ", "танилцуулга нээ"),
]


def test_normalize():
    bad = []
    for raw, want in NORMALIZE_CASES:
        got = normalize(raw)
        if got != want:
            bad.append(f"{raw!r}: expected {want!r}, got {got!r}")
    check(not bad, "normalize() mismatches:\n    " + "\n    ".join(bad))


def test_normalize_joins_punctuation_and_case():
    got = normalize("  ВОРД,\tНЭЭЖ! ... ")
    check(got == "ворд нээж", f"expected 'ворд нээж', got {got!r}")
    check(normalize("ВОРД") == normalize("ворд"), "lowercasing is not idempotent")


def test_regress_normalize_keeps_urls():
    got = normalize("https://youtube.com")
    check(got == "https://youtube.com", f"normalised URL is {got!r}")
    check(normalize("www.ub.edu.mn") == "www.ub.edu.mn",
          f"normalised bare host is {normalize('www.ub.edu.mn')!r}")
    check(normalize("ворд, нээ.") == "ворд нээ",
          "ordinary punctuation handling regressed")


# --------------------------------------------------------------------------
# parse()
# --------------------------------------------------------------------------

PARSE_CASES = [
    # (sentence, expected intent name or None, expected target)
    ("Ворд нээ", "open", "ворд"),
    ("блокнот нээ", "open", "блокнот"),
    ("блокнот нээж өг", "open", "блокнот"),
    ("YouTube нээ", "open", "youtube"),
    ("IESLTS study pack нээ", "open", "ieslts study pack"),
    ("компани танилцуулга нээ", "present", "компани"),
    ("танилцуулга эхлүүл", "present", ""),
    ("зогсоо", "close", "бүх"),
    ("зогсооод өг", "close", "бүх"),
    ("блокнот хаа", "close", "блокнот"),
    ("сонсохыг зогсоо", "sleep", ""),
    ("туллах", "help", ""),
    ("тусламж", "help", ""),
    ("аудио файлууд", "list", "аудио"),
    ("хай", "search", ""),
    ("гэрэл унтраах гэж байна", None, None),
    ("маш сайхан өдөр байна", None, None),
    ("", None, None),
    ("   ", None, None),
]


def test_parse_table():
    bad = []
    for text, name, target in PARSE_CASES:
        got = parse(text)
        if name is None:
            if got is not None:
                bad.append(f"{text!r}: expected None, got ({got.name!r}, {got.target!r})")
            continue
        if got is None:
            bad.append(f"{text!r}: expected ({name!r}, {target!r}), got None")
        elif (got.name, got.target) != (name, target):
            bad.append(f"{text!r}: expected ({name!r}, {target!r}), "
                       f"got ({got.name!r}, {got.target!r})")
    check(not bad, "parse() mismatches:\n    " + "\n    ".join(bad))


def test_parse_keeps_raw_and_is_case_insensitive():
    i = parse("  ВОРД НЭЭЖ ӨГ ")
    check(i is not None and i.raw == "ВОРД НЭЭЖ ӨГ",
          f"raw should be the sentence with case kept, outer blanks stripped; "
          f"got {getattr(i, 'raw', None)!r}")
    check((i.name, i.target) == ("open", "ворд"), f"got ({i.name!r}, {i.target!r})")
    check(parse("ТУСЛАМЖ").name == "help", "uppercase Cyrillic command not understood")


def test_parse_empty_input_is_not_a_command():
    for text in ("", "   ", "\t\n"):
        check(parse(text) is None, f"parse({text!r}) should be None")


def test_regress_help_vocabulary_covers_real_spellings():
    for text in ("туслах", "ТУСЛАХ", "туслаач", "туллах", "тусламж"):
        got = parse(text)
        check(got is not None and got.name == "help",
              f"parse({text!r}) is {None if got is None else got.name!r}, expected 'help'")


def test_regress_clean_target_keeps_letters():
    cases = [("улаанбаатарын зураг нээ", "open", "улаанбаатарын зураг"),
             ("түр дуу харуул", "open", "түр дуу"),
             ("блокнот нээж өг", "open", "блокнот")]
    bad = []
    for text, name, target in cases:
        got = parse(text)
        if got is None or (got.name, got.target) != (name, target):
            bad.append(f"{text!r}: got {None if got is None else (got.name, got.target)!r}, "
                       f"expected ({name!r}, {target!r})")
    check(not bad, "clean_target mangled a name:\n    " + "\n    ".join(bad))


def test_regress_spoken_url_opens():
    with fixture() as tmp, no_side_effects():
        intent = parse("https://example.com нээ")
        check(intent is not None and intent.name == "open",
              f"expected an open intent, got {intent}")
        check(intent.target == "https://example.com", f"target is {intent.target!r}")
        action = resolve(intent, FileIndex([tmp]))
        check(action.kind == "url" and action.ok is True,
              f"expected a url action, got kind={action.kind!r} ok={action.ok!r}")
        action = run(action, dry_run=True)
        check(action.kind == "url" and "DRY-RUN" in action.extra,
              f"dry run not tagged: {action.extra!r}")


# --------------------------------------------------------------------------
# FileIndex()
# --------------------------------------------------------------------------

def test_fileindex_exact_hit():
    with fixture() as tmp:
        idx = FileIndex([tmp])
        for query, name in (("alpha_report", "alpha_report.docx"),
                            ("beta_slides", "beta_slides.pptx"),
                            ("тайлал", "тайлал.pdf")):
            paths, score, how = idx.find(query)
            check(how == "exact", f"{query!r}: how={how!r}, expected 'exact'")
            check(score == 1.0, f"{query!r}: score={score!r}, expected 1.0")
            check(len(paths) == 1, f"{query!r}: got {len(paths)} paths, expected 1")
            want = tmp / name
            check(Path(paths[0]).is_absolute(), f"{query!r}: path is not absolute")
            check(same_path(paths[0], want), f"{query!r}: got {paths[0]!r}, want {str(want)!r}")


def test_fileindex_fuzzy_near_miss():
    with fixture() as tmp:
        idx = FileIndex([tmp])
        paths, score, how = idx.find("alpha report")
        check(how == "fuzzy", f"how={how!r}, expected 'fuzzy'")
        check(0.0 < score < 1.0, f"fuzzy score {score!r} is not inside (0, 1)")
        check(same_path(paths[0], tmp / "alpha_report.docx"),
              f"got {paths[0]!r}, want alpha_report.docx")
        # a truncated stem is a 'partial' hit, also strictly between 0 and 1
        paths, score, how = idx.find("alpha_repor")
        check(how == "partial", f"how={how!r}, expected 'partial'")
        check(0.0 < score < 1.0, f"partial score {score!r} is not inside (0, 1)")
        check(same_path(paths[0], tmp / "alpha_report.docx"),
              f"got {paths[0]!r}, want alpha_report.docx")


def test_fileindex_misses():
    with fixture() as tmp:
        idx = FileIndex([tmp])
        for query, how in (("", "empty"), ("zzzzqqqxyz", "none")):
            paths, score, how_got = idx.find(query)
            check(how_got == how, f"{query!r}: how={how_got!r}, expected {how!r}")
            check(paths == [], f"{query!r}: got {paths!r}, expected []")
            check(score == 0.0, f"{query!r}: score={score!r}, expected 0.0")


def test_fileindex_stays_inside_its_roots():
    with fixture() as tmp:
        idx = FileIndex([tmp])
        table = idx.table()
        missing = {Path(f).stem for f in FIXTURE_FILES} - set(table)
        check(not missing, f"table is missing the stems {missing!r}")
        for stem, paths in table.items():
            for p in paths:
                check(Path(p).is_absolute(), f"{stem!r}: {p!r} is not absolute")
                check(tmp in Path(p).parents, f"{stem!r}: {p!r} escaped {str(tmp)!r}")


def test_regress_empty_roots_do_not_scan_profile():
    idx = FileIndex([])
    check(idx.roots == [], f"FileIndex([]) must have no roots, got {idx.roots!r}")
    check(idx.table() == {}, "an empty FileIndex must index nothing")
    check(FileIndex().roots, "FileIndex() with no argument must still use the defaults")


# --------------------------------------------------------------------------
# resolve()
# --------------------------------------------------------------------------

def test_resolve_url_target():
    with fixture() as tmp:
        a = resolve(Intent("open", "https://example.com/x"), FileIndex([tmp]))
        check((a.kind, a.value, a.ok) == ("url", "https://example.com/x", True),
              f"got kind={a.kind!r} value={a.value!r} ok={a.ok!r}")
        b = resolve(Intent("open", "www.example.com"), FileIndex([tmp]))
        check((b.kind, b.value) == ("url", "https://www.example.com"),
              f"got kind={b.kind!r} value={b.value!r}")


def test_resolve_site_alias():
    with fixture() as tmp:
        for alias, want in (("youtube", "https://youtube.com"),
                            ("фейсбук", "https://facebook.com"),
                            ("github", "https://github.com")):
            a = resolve(Intent("open", alias), FileIndex([tmp]))
            check((a.kind, a.value) == ("url", want),
                  f"{alias!r}: got kind={a.kind!r} value={a.value!r}, want url {want!r}")
            check(a.ok is True, f"{alias!r}: ok={a.ok!r}")


def test_resolve_app_alias():
    with fixture() as tmp:
        for alias, want in (("ворд", "winword"), ("блокнот", "notepad"),
                            ("эксэл", "excel"), ("хром", "chrome")):
            a = resolve(Intent("open", alias), FileIndex([tmp]))
            check((a.kind, a.value, a.ok) == ("app", want, True),
                  f"{alias!r}: got kind={a.kind!r} value={a.value!r} ok={a.ok!r}")
            check(want in actions.APP_COMMANDS,
                  f"{alias!r}: {want!r} has no entry in APP_COMMANDS, run() cannot launch it")


def test_resolve_existing_file():
    with fixture() as tmp:
        idx = FileIndex([tmp])
        # by stem, through the index
        a = resolve(Intent("open", "alpha_report"), idx)
        check((a.kind, a.ok) == ("file", True),
              f"got kind={a.kind!r} ok={a.ok!r}")
        check(same_path(a.value, tmp / "alpha_report.docx"),
              f"got {a.value!r}, want alpha_report.docx")
        # by full path, without the index
        full = str(tmp / "alpha_report.docx")
        b = resolve(Intent("open", full), FileIndex([tmp]))
        check((b.kind, b.ok) == ("file", True), f"got kind={b.kind!r} ok={b.ok!r}")
        check(same_path(b.value, full), f"got {b.value!r}, want {full!r}")
        # folder
        c = resolve(Intent("open", "projects"), FileIndex([tmp]))
        check((c.kind, c.ok) == ("folder", True), f"got kind={c.kind!r} ok={c.ok!r}")
        check(same_path(c.value, tmp / "projects"), f"got {c.value!r}, want projects/")


def test_resolve_present_picks_a_slide_file():
    with fixture() as tmp:
        idx = FileIndex([tmp])
        for query, name in (("beta_slides.pptx", "beta_slides.pptx"),
                            ("тайлал", "тайлал.pdf")):
            a = resolve(Intent("present", query), idx)
            check((a.kind, a.ok) == ("slides", True),
                  f"{query!r}: got kind={a.kind!r} ok={a.ok!r}")
            check(same_path(a.value, tmp / name), f"{query!r}: got {a.value!r}")
        # SLIDE_EXT has no ".docx": a Word file is not presentable, so it falls
        # back to a failed web search rather than erroring.
        b = resolve(Intent("present", "alpha_report.docx"), idx)
        check((b.kind, b.ok) == ("search", False),
              f"got kind={b.kind!r} ok={b.ok!r}")


def test_resolve_unknown_name():
    with fixture() as tmp:
        idx = FileIndex([tmp])
        search_prefix = actions.SEARCH_URL.split("{}")[0]
        single = resolve(Intent("open", "хонх"), idx)          # one word, no space
        check((single.kind, single.ok) == ("none", False),
              f"got kind={single.kind!r} ok={single.ok!r}")
        check(single.value == "", f"a 'none' action should carry no value, got {single.value!r}")
        multi = resolve(Intent("open", "гэрийн байшингийн зураг"), idx)   # >1 word
        check((multi.kind, multi.ok) == ("search", False),
              f"got kind={multi.kind!r} ok={multi.ok!r}")
        check(multi.value.startswith(search_prefix), f"search value is {multi.value!r}")


def test_resolve_close_maps_to_a_process():
    with fixture() as tmp:
        idx = FileIndex([tmp])
        a = resolve(Intent("close", "ворд"), idx)
        check((a.kind, a.value) == ("process", "winword"),
              f"got kind={a.kind!r} value={a.value!r}")
        b = resolve(Intent("close", "бүх"), idx)
        # "зогсоо" on its own means "stop everything"; it has to reach a real
        # action, not fall through to "I did not understand that"
        check((b.kind, b.value, b.ok) == ("process", "__all__", True),
              f"close('бүх'): got kind={b.kind!r} value={b.value!r} ok={b.ok!r}")


def test_regress_help_and_sleep_are_not_swallowed():
    with fixture() as tmp, no_side_effects():
        idx = FileIndex([tmp])
        help_action = resolve(parse("туслах"), idx)
        check(help_action.kind == "help" and help_action.ok is True,
              f"help resolved to {help_action.kind!r}/{help_action.ok!r}")
        check(len(help_action.reply) > 20, "help reply is empty")
        sleep_action = resolve(parse("сонсохыг зогсоо"), idx)
        check(sleep_action.value == "sleep" and sleep_action.ok is True,
              f"sleep resolved to {sleep_action.value!r}/{sleep_action.ok!r}")


# --------------------------------------------------------------------------
# run() -- dry_run only, every side-effecting channel stubbed out
# --------------------------------------------------------------------------

def test_safety_stubs_are_active():
    with no_side_effects():
        check(os.startfile.__module__ == __name__, "os.startfile is not stubbed")
        check(subprocess.Popen.__module__ == __name__, "subprocess.Popen is not stubbed")
        check(webbrowser.open.__module__ == __name__, "webbrowser.open is not stubbed")
        check(actions._kill.__module__ == __name__, "actions._kill is not stubbed")


def test_run_dry_run_marks_every_kind():
    with fixture() as tmp, no_side_effects() as calls:
        docx = str(tmp / "alpha_report.docx")
        pptx = str(tmp / "beta_slides.pptx")
        cases = [
            # (kind, value, ok in, ok out, expected recorded calls, expected extra)
            ("file", docx, True, True, [], "[DRY-RUN file]"),
            ("folder", str(tmp), True, True, [], "[DRY-RUN folder]"),
            ("url", "https://example.com", True, True, [], "[DRY-RUN url]"),
            ("search", "https://duckduckgo.com/?q=x", True, True, [], "[DRY-RUN search]"),
            ("slides", pptx, True, True, [], "[DRY-RUN slides]"),
            ("app", "winword", True, True, [], "[DRY-RUN app]"),
            ("process", "winword", True, True, [], "[DRY-RUN process]"),
            ("list", "аудио", True, True, [], "аудио [DRY-RUN list]"),
            ("none", "", False, False, [], "[DRY-RUN none]"),
            ("help", "", True, True, [], "[DRY-RUN help]"),
        ]
        bad = []
        for kind, value, ok_in, ok_out, want_calls, want_extra in cases:
            calls.clear()
            a = run(Action(kind, value=value, reply="r", ok=ok_in), dry_run=True)
            if a.ok is not ok_out:
                bad.append(f"{kind}: ok={a.ok!r}, expected {ok_out!r}")
            if calls != want_calls:
                bad.append(f"{kind}: calls={calls!r}, expected {want_calls!r}")
            if a.extra != want_extra:
                bad.append(f"{kind}: extra={a.extra!r}, expected {want_extra!r}")
            if a.reply != "r" and kind != "bogus":
                bad.append(f"{kind}: dry_run rewrote reply to {a.reply!r}")
        check(not bad, "run(dry_run=True) mismatches:\n    " + "\n    ".join(bad))


def test_run_dry_run_rejects_unknown_kind():
    with no_side_effects() as calls:
        a = run(Action("teleport", value="x", reply="r"), dry_run=True)
        check(a.ok is False, f"unknown kind should fail, ok={a.ok!r}")
        check(a.reply == "Тэгэх мөр үйлдэл байхгүй: teleport", f"reply={a.reply!r}")
        check(calls == [], f"unknown kind reached {calls!r}")
        check(a.extra == "[DRY-RUN teleport]", f"extra={a.extra!r}")


def test_run_dry_run_touches_nothing_on_disk():
    with fixture() as tmp, no_side_effects():
        before = sorted(str(p.relative_to(tmp)) for p in tmp.rglob("*"))
        a = run(Action("file", value=str(tmp / "alpha_report.docx"),
                       reply="r", extra="note", ok=True), dry_run=True)
        after = sorted(str(p.relative_to(tmp)) for p in tmp.rglob("*"))
        check(after == before, f"dry run changed the file tree: {before} -> {after}")
        check(a.value == str(tmp / "alpha_report.docx"), "dry_run rewrote .value")
        check(a.extra == "note [DRY-RUN file]", f"extra={a.extra!r}")
        check(a.ok is True, f"ok={a.ok!r}")


def test_regress_dry_run_never_fires():
    for kind, value in (("file", "alpha_report.docx"), ("folder", "."),
                        ("url", "https://example.com"),
                        ("search", "https://duckduckgo.com/?q=x"),
                        ("slides", "beta_slides.pptx"),
                        ("app", "winword"), ("process", "winword")):
        with no_side_effects() as calls:
            action = run(Action(kind, value, ok=True), dry_run=True)
        check(calls == [],
              f"dry_run {kind} still fired {calls!r} -- dry run must never "
              f"open a file, start an app, or kill a process")
        check(f"DRY-RUN {kind}" in action.extra,
              f"dry_run {kind} not tagged: {action.extra!r}")


def test_regress_none_action_defaults_to_not_ok():
    check(Action("none").ok is False,
          "a no-op Action must not default to ok=True")
    for kind in ("file", "url", "app", "search", "folder", "slides", "process"):
        with no_side_effects():
            check(run(Action(kind, "x", ok=True), dry_run=True).ok is True,
                  f"dry-run {kind} should report ok=True")
    # a dry run must not turn a failed resolution into a success
    with no_side_effects():
        check(run(Action("none", ok=False), dry_run=True).ok is False,
              "dry run upgraded a no-op to ok=True")


# --------------------------------------------------------------------------
# Assistant.handle() -- dry_run, stubs on
# --------------------------------------------------------------------------

def test_assistant_handle_dry_run():
    with fixture() as tmp, no_side_effects() as calls:
        a = Assistant([tmp])
        cases = [
            ("alpha_report нээ", ("open", "alpha_report"), "file"),
            ("alpha report нээ", ("open", "alpha report"), "file"),
            ("блокнот нээ", ("open", "блокнот"), "app"),
            ("youtube нээ", ("open", "youtube"), "url"),
            ("танилцуулга эхлүүл", ("present", ""), "none"),   # BUG, see below
            ("сонсохыг зогсоо", ("sleep", ""), "none"),        # BUG, see below
        ]
        bad = []
        for text, want_intent, want_kind in cases:
            calls.clear()
            intent, action = a.handle(text, dry_run=True)
            got_intent = None if intent is None else (intent.name, intent.target)
            if got_intent != want_intent:
                bad.append(f"{text!r}: intent {got_intent!r} != {want_intent!r}")
            if action.kind != want_kind:
                bad.append(f"{text!r}: kind {action.kind!r} != {want_kind!r}")
            if f"[DRY-RUN {action.kind}]" not in action.extra:
                bad.append(f"{text!r}: extra {action.extra!r} has no DRY-RUN tag")
        check(not bad, "Assistant.handle() mismatches:\n    " + "\n    ".join(bad))
        # "сонсохыг зогсоо" carries no target, but it must still reach a real
        # action instead of being swallowed by the empty-target guard
        _, action = a.handle("сонсохыг зогсоо", dry_run=True)
        check(action.value == "sleep" and action.ok is True,
              f"sleep: got value={action.value!r} ok={action.ok!r}")
        # "танилцуулга эхлүүл" has no file name, so it must ask for one
        _, action = a.handle("танилцуулга эхлүүл", dry_run=True)
        check(action.ok is False and "нэрлээрэй" in action.reply,
              f"nameless present: got ok={action.ok!r} reply={action.reply!r}")


def test_assistant_handle_unknown_sentence():
    with fixture() as tmp, no_side_effects() as calls:
        a = Assistant([tmp])
        intent, action = a.handle("гэрэл унтраах гэж байна", dry_run=True)
        check(intent is None, f"expected no intent, got {intent}")
        check((action.kind, action.ok, action.reply) == ("none", False, ""),
              f"got kind={action.kind!r} ok={action.ok!r} reply={action.reply!r}")
        check(calls == [], f"an unparsed sentence reached {calls!r}")


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:                                    # noqa: BLE001
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"PASS {t.__name__}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed"
          + (f" -- FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
