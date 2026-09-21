"""Export the local article as editable WordPress blocks and a draft REST payload."""
from pathlib import Path
import json
import re
import markdown

ROOT = Path(__file__).resolve().parents[1]
BLOG = ROOT / 'docs/blog'
stem = 'local-coding-agents-16gb-gpu'
source = (BLOG / (stem + '.md')).read_text(encoding='utf-8')
title, content = source.split('\n', 1)
title = title.removeprefix('# ').strip()
rendered = markdown.markdown(content, extensions=['tables'])
blocks = re.findall(r'<p>.*?</p>|<h2>.*?</h2>|<h3>.*?</h3>|<table>.*?</table>|<ul>.*?</ul>|<ol>.*?</ol>|<blockquote>.*?</blockquote>', rendered, re.S)
assert re.sub(r'\s+', '', ''.join(blocks)) == re.sub(r'\s+', '', rendered), 'Unconverted content'
output = []
for block in blocks:
    tag = re.match(r'<(\w+)', block).group(1)
    if tag == 'p':
        output.append('<!-- wp:paragraph -->\n' + block + '\n<!-- /wp:paragraph -->')
    elif tag in ('h2', 'h3'):
        attrs = '' if tag == 'h2' else ' {"level":3}'
        block = block.replace('<' + tag + '>', '<' + tag + ' class="wp-block-heading">', 1)
        output.append('<!-- wp:heading' + attrs + ' -->\n' + block + '\n<!-- /wp:heading -->')
    elif tag == 'table':
        output.append('<!-- wp:table -->\n<figure class="wp-block-table">' + block + '</figure>\n<!-- /wp:table -->')
    elif tag in ('ul', 'ol'):
        attrs = '' if tag == 'ul' else ' {"ordered":true}'
        block = block.replace('<' + tag + '>', '<' + tag + ' class="wp-block-list">', 1)
        block = block.replace('<li>', '<!-- wp:list-item -->\n<li>').replace('</li>', '</li>\n<!-- /wp:list-item -->')
        output.append('<!-- wp:list' + attrs + ' -->\n' + block + '\n<!-- /wp:list -->')
    else:
        output.append('<!-- wp:html -->\n' + block + '\n<!-- /wp:html -->')
html = '\n\n'.join(output) + '\n'
excerpt = 'I tested local coding-agent configurations on a 16 GB GPU. Here are the useful picks, tool failures, memory limits, and reasons nothing qualified.'
payload = {'status': 'draft', 'title': title, 'slug': stem, 'excerpt': excerpt, 'content': html}
(BLOG / (stem + '.wordpress.html')).write_text(html, encoding='utf-8')
(BLOG / (stem + '.wordpress-post.json')).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
notes = {
    'target_site': 'https://sosuke.com',
    'publication_state': 'local_draft_only',
    'suggested_categories': ['AI', 'LLM', 'Performance', 'Testing'],
    'suggested_tags': ['local LLM', 'coding agents', 'benchmarks', 'llama.cpp', 'quantization', 'MTP', 'DFlash'],
    'title': title, 'slug': stem, 'excerpt': excerpt,
    'wordpress_editor': 'Paste the .wordpress.html file into the WordPress block editor Code editor, then switch back to Visual editor. Enter the title separately.',
    'rest_payload': 'The .wordpress-post.json file is a draft POST payload. Category and tag IDs are intentionally omitted because site IDs have not been looked up.',
    'report_attachment': 'output/pdf/local-coding-agent-benchmark-report.pdf is available locally. Upload it separately before adding a public download link.',
    'evidence_dates_utc': '2026-09-18 to 2026-09-21',
    'review': 'Screening samples and early-stopped coding denominators are explicitly labeled. No private repository links, machine account paths, or credentials are included in the article.',
}
(BLOG / (stem + '.publication.json')).write_text(json.dumps(notes, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
assert html.count('<!-- wp:table -->') == 3
assert '<script' not in html.lower()
assert len(excerpt) <= 160
print(json.dumps({'title': title, 'words': len(re.findall(r'\b[\w-]+\b', source)), 'blocks': len(blocks),
                  'tables': 3, 'excerpt_characters': len(excerpt), 'directory': str(BLOG)}))

