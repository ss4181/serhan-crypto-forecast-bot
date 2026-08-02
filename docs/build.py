"""Render docs/index.md into a single self-contained docs/index.html.

    python docs/build.py

Keeping one source and generating the other stops the two copies from drifting.
The output embeds its own styling and loads nothing from the network, so the
file works offline, from a USB stick, or attached to an e-mail.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "index.md"
OUTPUT = HERE / "index.html"

STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0 auto; padding: 2.5rem 1.25rem 6rem; max-width: 46rem;
  font: 16px/1.65 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  color: #1b1b1b; background: #fff;
  -webkit-text-size-adjust: 100%;
}
h1 { font-size: 1.9rem; line-height: 1.2; margin: 0 0 .5rem; }
h2 {
  font-size: 1.3rem; margin: 3rem 0 .75rem; padding-top: 1.25rem;
  border-top: 1px solid #e2e2e2;
}
h3 { font-size: 1.05rem; margin: 2rem 0 .5rem; }
p, ul, ol { margin: 0 0 1rem; }
li { margin-bottom: .3rem; }
a { color: #0b5cad; }
code {
  font: .875em/1.5 ui-monospace, SFMono-Regular, Consolas, monospace;
  background: #f2f2f2; padding: .1em .35em; border-radius: 3px;
}
pre {
  background: #f7f7f7; border: 1px solid #e5e5e5; border-radius: 4px;
  padding: .9rem 1rem; overflow-x: auto;
}
pre code { background: none; padding: 0; }
blockquote {
  margin: 1.25rem 0; padding: .1rem 0 .1rem 1rem;
  border-left: 3px solid #c8c8c8; color: #444;
}
/* Wide tables scroll inside their own box rather than the page. */
.table-wrap { overflow-x: auto; margin: 0 0 1.25rem; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { padding: .45rem .7rem; border-bottom: 1px solid #e5e5e5; text-align: left; }
th { font-weight: 600; background: #f7f7f7; white-space: nowrap; }
tbody tr:last-child td { border-bottom: none; }
hr { border: none; border-top: 1px solid #e2e2e2; margin: 2.5rem 0; }
footer { margin-top: 4rem; font-size: .85rem; color: #666; }
@media (prefers-color-scheme: dark) {
  body { color: #e6e6e6; background: #16181c; }
  h2, hr { border-color: #2e323a; }
  a { color: #79b8ff; }
  code { background: #23262c; }
  pre { background: #1c1f24; border-color: #2e323a; }
  th { background: #1c1f24; }
  th, td { border-color: #2e323a; }
  blockquote { border-color: #3a3f48; color: #b8bcc4; }
  footer { color: #9aa0a8; }
}
@media print {
  body { max-width: none; padding: 0; color: #000; background: #fff; }
  h2 { page-break-after: avoid; }
  .table-wrap, table { page-break-inside: avoid; }
}
"""

PAGE = """<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{style}</style>
</head>
<body>
{body}
<footer>
Bu sayfa <code>docs/index.md</code> dosyasından üretilmiştir
(<code>python docs/build.py</code>). Kaynak değişirse yeniden üretin.
</footer>
</body>
</html>
"""


def front_matter_title(text: str) -> tuple[str, str]:
    """Strip the Jekyll front matter and take the title out of it."""
    match = re.match(r"\A---\n(.*?)\n---\n", text, flags=re.S)
    if not match:
        return "Trade Lab", text
    found = re.search(r'^title:\s*"?(.+?)"?\s*$', match.group(1), flags=re.M)
    title = found.group(1) if found else "Trade Lab"
    return title, text[match.end() :]


def main() -> int:
    if not SOURCE.exists():
        print(f"{SOURCE} bulunamadi", file=sys.stderr)
        return 1
    title, body_markdown = front_matter_title(SOURCE.read_text(encoding="utf-8"))
    body = markdown.markdown(body_markdown, extensions=["tables", "sane_lists"])
    body = body.replace("<table>", '<div class="table-wrap"><table>')
    body = body.replace("</table>", "</table></div>")
    OUTPUT.write_text(
        PAGE.format(title=html.escape(title), style=STYLE, body=body),
        encoding="utf-8",
    )
    print(f"{OUTPUT} yazildi ({OUTPUT.stat().st_size:,} bayt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
