"""Create the evidence-based findings PDF without running benchmark workloads."""
from pathlib import Path
from datetime import datetime, timezone
from xml.sax.saxutils import escape
import hashlib
import json
import sqlite3
import textwrap

from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Preformatted, Flowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import letter
from pypdf import PdfReader, PdfWriter
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / 'tmp/pdfs'
OUT = ROOT / 'output/pdf'
OUT.mkdir(parents=True, exist_ok=True)
ORIG = json.loads((TMP / 'original-audit.json').read_text())
FOCUS = json.loads((TMP / 'focused-audit.json').read_text())
PDF = OUT / 'local-coding-agent-benchmark-report.pdf'
NOW = datetime.now(timezone.utc)
NAVY = HexColor('#173044')
TEAL = HexColor('#087E8B')
INK = HexColor('#253847')
MUTED = HexColor('#5D6D77')
PALE = HexColor('#EDF4F6')
ORANGE = HexColor('#B86414')
RED = HexColor('#A33B38')
GREEN = HexColor('#167164')
W = 528

pdfmetrics.registerFont(TTFont('Segoe', 'C:/Windows/Fonts/segoeui.ttf'))
pdfmetrics.registerFont(TTFont('SegoeBold', 'C:/Windows/Fonts/segoeuib.ttf'))
pdfmetrics.registerFont(TTFont('SegoeItalic', 'C:/Windows/Fonts/segoeuii.ttf'))
pdfmetrics.registerFontFamily('Segoe', normal='Segoe', bold='SegoeBold', italic='SegoeItalic', boldItalic='SegoeBold')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle('BodyCustom', fontName='Segoe', fontSize=9.3, leading=13.4, textColor=INK, spaceAfter=9))
styles.add(ParagraphStyle('SmallCustom', fontName='Segoe', fontSize=7.9, leading=10.7, textColor=MUTED, spaceAfter=7))
styles.add(ParagraphStyle('TitleCustom', fontName='SegoeBold', fontSize=25, leading=30, textColor=NAVY, spaceAfter=14))
styles.add(ParagraphStyle('SectionCustom', fontName='SegoeBold', fontSize=20, leading=25, textColor=NAVY, spaceAfter=13))
styles.add(ParagraphStyle('SubCustom', fontName='SegoeBold', fontSize=11.7, leading=16, textColor=TEAL, spaceBefore=7, spaceAfter=7))
styles.add(ParagraphStyle('Eyebrow', fontName='SegoeBold', fontSize=8, leading=11, textColor=TEAL, spaceAfter=7))
styles.add(ParagraphStyle('CellCustom', fontName='Segoe', fontSize=8.1, leading=10.8, textColor=INK))
styles.add(ParagraphStyle('CellSmall', fontName='Segoe', fontSize=7.1, leading=9.5, textColor=INK))
styles.add(ParagraphStyle('CellHead', fontName='SegoeBold', fontSize=7.6, leading=10.3, textColor=white))
styles.add(ParagraphStyle('CodeCustom', fontName='Courier', fontSize=7.15, leading=9.8, textColor=INK, backColor=PALE, borderPadding=8, spaceAfter=10))
story = []


def p(text, style='BodyCustom'):
    return Paragraph(text, styles[style])


def body(text):
    story.append(p(text))


def sub(text):
    story.append(p(text, 'SubCustom'))


def note(text):
    story.append(p(text, 'SmallCustom'))


def page(title, label, first=False):
    if not first:
        story.append(PageBreak())
    story.append(p(label.upper(), 'Eyebrow'))
    story.append(p(title, 'TitleCustom' if first else 'SectionCustom'))


def table(headers, rows, widths, small=False):
    cellstyle = 'CellSmall' if small else 'CellCustom'
    data = [[p(escape(str(h)), 'CellHead') for h in headers]]
    data += [[p(escape(str(x)).replace('\n', '<br/>'), cellstyle) for x in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0),NAVY), ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),7), ('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),7), ('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[white, PALE]),
        ('LINEBELOW',(0,0),(-1,0),0.7,TEAL),
    ]))
    story.append(t)
    story.append(Spacer(1,10))


def box(text, color=PALE):
    t = Table([[p(text)]], colWidths=[W])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),color),('LEFTPADDING',(0,0),(-1,-1),12),
                           ('RIGHTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,-1),10),
                           ('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.append(t)
    story.append(Spacer(1,12))


def fmt(value, digits=2):
    return 'not measured' if value is None else f'{value:.{digits}f}'


def rng(data):
    return 'not measured' if not data or not data.get('n') else f"{data['min']:.2f}-{data['max']:.2f}"


def pick(model, engine='upstream-llama-nightly'):
    return next(c for c in ORIG['inventory'] if c['model_id'] == model and c['engine_id'] == engine
                and c['speculation'] == 'off' and c['regular']['attempted'] > 0)


Q = pick('qwen36-27b-iq2')
QB = pick('qwen36-27b-iq2','bundled-llama')
OX = pick('oxcoder9b-q4km')
QS = pick('qwen38-27b-iq3')
SO = pick('signoffour35ba3b-iq2m')
FAMILY_IDS = ['devstral-small2-24b-q3','ministral3-14b-reasoning-q4','gptoss20b-q4ks',
              'gemma4-12b-qat-q4','gemma4-e4b-q6','nemotron3nano-4b-q6']
FAMILY_LABELS = ['Devstral Small 2 24B Q3_K_L','Ministral 3 14B Reasoning Q4_K_M','GPT-OSS 20B Q4_K_S',
                 'Gemma 4 12B QAT Q4_0','Gemma 4 E4B Q6_K','Nemotron 3 Nano 4B Q6_K']
FAMILY_RESULTS = {i:json.loads((ROOT / 'artifacts/families-16k' / i / 'results.json').read_text()) for i in FAMILY_IDS}


def quality_errors(result):
    rows = result.get('results', [])
    # Focused exports retain the latest attempt per job.
    results = [r.get('result',r) for r in rows]
    completed = [r for r in results if r.get('kind') == 'quality' and r.get('group') == 'regular' and r.get('valid') is True]
    return sum(r.get('reason') == 'malformed_or_unpermitted_tool_stream' for r in completed)


class Bars(Flowable):
    def __init__(self, items, maximum=16.110, height=200, labelwidth=185, suffix='GiB'):
        Flowable.__init__(self)
        self.items = items
        self.maximum = maximum
        self.width = W
        self.height = height
        self.labelwidth = labelwidth
        self.suffix = suffix
    def draw(self):
        c = self.canv
        chartw = self.width - self.labelwidth - 77
        rowh = self.height / len(self.items)
        for i,(label,value,color,right) in enumerate(self.items):
            y = self.height - (i+1)*rowh + 7
            c.setFont('Segoe',8)
            c.setFillColor(INK)
            c.drawString(0,y+4,label)
            c.setFillColor(PALE)
            c.roundRect(self.labelwidth,y,chartw,16,3,fill=1,stroke=0)
            c.setFillColor(color)
            c.roundRect(self.labelwidth,y,max(2,chartw*min(value/self.maximum,1)),16,3,fill=1,stroke=0)
            c.setFillColor(INK)
            c.setFont('SegoeBold',8)
            c.drawString(self.labelwidth+chartw+7,y+4,right)


page('Local coding-agent benchmark\nFindings and practical picks', 'Decision report | RTX 4060 Ti 16 GB', True)
note('Measured September 18-21, 2026 (UTC). Report compiled ' + NOW.strftime('%Y-%m-%d %H:%M UTC') + '.')
box('<b>No configuration qualified as a verified coding-agent winner at 64K or in the completed 16K family queue.</b> The picks below are useful starting points for supervised work or further investigation, with explicit limits.')
table(['Pick','Why I would keep it','Measured limit'],[
 ['1. Qwen3.6 27B i1-IQ2_M','Best observed coding quality; relatively few tool failures. Upstream for reliability, bundled for faster diagnostic fresh action.','28/36 coding passes. Long tasks: 4/8 completed; eliminated before qualification.'],
 ['2. OxCoder 9B Q4_K_M / upstream','Best measured fast fallback with clean tool use across its completed coding tasks.','11/21 coding passes; 0/21 tool errors. Not qualified.'],
 ['3. GPT-OSS 20B / interface research','Its basic native file tools worked. Patch-format incompatibility is a concrete issue worth isolating.','0/4 coding passes; all four failed the required patch format. Capacity parser also failed one seed.'],
 ['4. Ternary Bonsai 2 27B / memory research','Exceptional weight-size reduction and demonstrated 64K capacity.','Slow fresh 64K response; 3/5 completed smoke passes, 2 tool failures. Stopped by user.'],
],[120,234,174])
body('If I had to use a local assistant today, I would start with <b>Qwen3.6 IQ2_M</b> for work where correctness matters most, or <b>OxCoder 9B</b> when responsiveness and a functioning tool loop matter more. Both need tests and human review. Neither earns the original 64K agent label.')
sub('What changed during the search')
body('Reducing context to 16K allowed some larger quantizations to fit, but it did not produce a qualifying agent. The new family queue finished cleanly. GPT-OSS hit the four-run tool-error cutoff; the other five models reached the regular coding elimination threshold.')
note('Evidence scope: 1,093 recorded attempts and 63 registered configurations. The original ranking export contains 932 attempts and 50 configurations; 13 later configurations are covered separately. Attempt counts include retries, diagnostics, and interrupted records, not 1,093 distinct coding tasks.')


page('What the benchmark actually required', 'Measurement and grading')
table(['Requirement','Original 64K contract','Separate 16K experiments'],[
 ['Context / fully templated input','65,536 / 61,440 tokens','16,384 / 12,288 tokens'],
 ['Output reserve','4,096 tokens','4,096 tokens'],
 ['Regular coding','At least 27/36, with complete coverage','Same frozen 27/36 gate'],
 ['Context coding','At least 9/12 long tasks','Same 9/12 gate at 12,288-token target'],
 ['Capacity','All nine marker values across three seeded requests','Same marker gate at the smaller input target'],
 ['Final stability','20 valid fresh + 20 valid cached measurements; isolated demo','Same stability and demo requirements'],
 ['Fit','Record real GPU and system memory','Full GPU layer offload, f16 KV, at least 512 MiB measured free headroom'],
],[112,207,209])
sub('Three speed metrics with different meanings')
body('<b>Fresh TTFT / first stream:</b> time from request dispatch to the first model output event. That event can be reasoning. <b>First useful action:</b> time to a completely parsed, schema-valid code/tool action. This is the primary responsiveness measure. <b>Decode throughput:</b> output tokens divided by verified decode time; it does not include the full cost of processing a fresh long prompt.')
body('Fresh measurements reset the model context and verify input/cache counters. Cached timing requires a verified stable-prefix priming procedure and is a separate secondary metric. Three diagnostic fresh samples do not establish the required 20-sample p95 or stability.')
sub('Tools and grading')
body('Python and TypeScript fixtures use deterministic seeds, public tests, hidden checks, and gold implementations. Agents can inspect and modify only permitted source files. The tool interface requires native calls and the advertised argument schemas. The patch tool requires a numbered unified diff. Malformed or unpermitted calls fail that coding run.')
body('The controller already allows iteration within a fixture: up to 16 tool turns, 32,768 generated tokens in aggregate, and a 900-second task deadline. Coding failures therefore measure the final task outcome after permitted interaction, rather than a single untested answer.')
note('The latest family cutoff stops after four coding runs when two or more contain tool-call errors. That cutoff is separate from the regular early elimination threshold of ten failed coding tasks.')


page('Original 64K search: quality beat headline speed', 'Measured results')
chosen = [Q,QB,QS,OX,SO] + [c for c in ORIG['inventory'] if c['model_id'] == 'ornith15-9b-q4' and c['engine_id']=='bundled-llama' and c['regular']['attempted']>0][:1]
rows=[]
for c in chosen:
    rows.append([c['model_id']+'\n'+c['engine_id'].replace('-llama-nightly',''),
                 f"{c['regular']['passed']}/{c['regular']['attempted']}",
                 f"{c['tool_errors_all_completed_regular']}/{c['regular']['attempted']}",
                 rng(c['fresh_screen_first_action']),
                 fmt(c['sustained_64k_throughput'].get('median'))])
table(['Model / engine','Coding\npass/run','Tool errors\nper run','Fresh useful\naction (s)','Exact sustained\ntok/s'],rows,[184,59,78,99,108],True)
note('Coding denominators are completed valid fixture runs, not the 36-task target. Useful-action ranges have three screen samples per listed configuration. Sustained throughput is one exact full-context sample per listed configuration. None is a qualifying final p95.')
sub('Qwen3.6 IQ2_M: closest to the original quality gate')
body('Both upstream and bundled engines passed <b>28/36 regular tasks (77.8%)</b> and all nine 64K markers. Long coding passed <b>4/8 completed tasks</b>; four failures made 9/12 impossible. Upstream failed the required single-final-action contract on all four failed long runs; bundled had three such failures and one timeout. This is a long-task tool-contract failure, not evidence that capacity exceeded GPU memory.')
body(f"Upstream had <b>1 tool error in 36 coding runs</b>; bundled had <b>2 in 36</b>. Upstream fresh first-stream range was {rng(Q['fresh_screen_first_stream'])} seconds and useful action {rng(Q['fresh_screen_first_action'])} seconds. Bundled was faster in its small screen: first stream {rng(QB['fresh_screen_first_stream'])}, useful action {rng(QB['fresh_screen_first_action'])} seconds.")
sub('OxCoder: a faster assistant with a weaker final solution rate')
body(f"OxCoder 9B Q4_K_M on upstream passed <b>11/21 tasks (52.4%)</b>, with <b>zero tool errors in 21 completed runs</b>. Its fresh useful action was {rng(OX['fresh_screen_first_action'])} seconds and sustained decode {fmt(OX['sustained_64k_throughput']['median'])} tok/s. Ten deterministic test failures eliminated it before long coding and final stability.")
sub('Why I would not pick the throughput leader')
body('SignOfFour 35B-A3B IQ2_M reached 59.64 tok/s, but passed only 10/20 completed tasks and had seven tool errors. Qwen3.8 IQ3_S passed 15/25, but all ten recorded coding failures were tool-stream failures. Those outcomes outweigh synthetic or decode speed under the stated priorities.')


page('Completed 16K family queue', 'Installed files | no new family downloads')
rows=[]
for ident,label in zip(FAMILY_IDS,FAMILY_LABELS):
    d=FAMILY_RESULTS[ident]
    n=d['regular']['attempted']
    rows.append([label,f"{d['regular']['passed']}/{n}",
                 str(quality_errors(d)),f"{d['first_pass']['tool_errors']}/4",
                 'Four-run stop' if d['first_pass']['stop'] else 'Coding eliminated'])
table(['Installed model','Coding\npass/run','All tool\nerrors','First-four\ntool errors','Outcome'],rows,[199,66,65,79,119],True)
note('All-tool counts use completed valid regular coding records. A coding run is counted once if its final failure was a tool error, even when earlier native calls succeeded. Coding and tool-error counts overlap.')
body('<b>All six models fit the 16K memory gate.</b> Devstral, Ministral, both Gemma models, and Nemotron passed all nine capacity markers. GPT-OSS passed the first two seeded capacity requests (six marker values); the third seed failed its native output parser.')
body('The queue exited cleanly, with no active job or benchmark model server remaining. Devstral, Ministral, both Gemma models, and Nemotron accumulated ten regular task failures, making the 27/36 target mathematically unreachable. They did not proceed to long coding, final 20+20 timing, or the demo.')
sub('What these results do and do not mean')
body('Gemma E4B had the best completed pass fraction in this small family queue, <b>5/15 (33.3%)</b>. That is still far below the required quality gate. The installed Nemotron tested here was <b>Nano 4B Q6_K</b>; this is not a verdict on larger Nemotron checkpoints. The GPT-OSS file was the requested <b>20B Q4_K_S</b> model.')
body('The Gemma 12B test used exactly <font face="Courier" size="7.5">F:/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf</font>. Muse/Glimmer was skipped at the user\'s request.')
box('<b>GPT-OSS needs a precise interpretation.</b> Its four failed coding runs successfully listed and read source files, then sent Codex-style patch envelopes. The frozen tool required unified diffs and rejected those arguments. This is a tool-contract failure of the tested configuration; it is not evidence that every tool response was null.')
note('Source: artifacts/families-16k/queue-status.json, per-model results.json and native turn logs; audit snapshots are embedded in this PDF.')


page('The fast 16K models still failed as agents', 'Clean diagnostic speed | tools versus accuracy')
speed_rows=[]
for ident,label in zip(FAMILY_IDS,FAMILY_LABELS):
    f=FOCUS['families_16k'][ident]
    fresh=f['fresh_screen_clean']
    speeds=[r['decode_tok_s'] for r in f['throughput_screen'] if r.get('clean_decode_timing') and r.get('valid')]
    speed_rows.append([label,fmt(fresh['ttft_seconds'].get('median'),3),
        fmt(fresh['useful_action_seconds'].get('median'),3),
        fmt(speeds[0],3) if speeds else 'not measured'])
table(['Model','Fresh TTFT\nmedian (s)','Fresh useful\naction median (s)','Native decode\ntok/s'],speed_rows,[242,89,107,90],True)
note('Fresh full-input action screens: three clean samples, 12,288 templated input tokens. Native throughput screens: one 256-output-token sample per shown result. These are structured-action diagnostics, not first source edits or the final 20-sample p95. GPT-OSS was stopped before these speed screens.')
body('Gemma E4B and Nemotron reached approximately 50-54 output tok/s and single-digit fresh useful-action seconds at 16K. That speed was real for the diagnostic workload, but final coding pass rates and tool failures were too weak to recommend either as an unattended agent. Do not compare these smaller-prompt times directly with the 61,440-token original search.')
sub('Tool-error share across completed coding runs')
chart=[]
for ident,label in zip(FAMILY_IDS,['Devstral 24B','Ministral 14B','GPT-OSS 20B','Gemma 12B QAT','Gemma E4B','Nemotron 4B']):
    d=FAMILY_RESULTS[ident]
    count=quality_errors(d)
    n=d['regular']['attempted']
    chart.append((label,100*count/n,RED if count/n>=0.5 else ORANGE,f'{count}/{n}'))
story.append(Bars(chart,maximum=100,height=192,labelwidth=160,suffix='%'))
note('Bar length is the fraction of completed coding runs ending in a malformed/unpermitted tool failure. The first-four cutoff used only the initial four runs, so later tool failures remain visible here. GPT-OSS errors were unsupported patch syntax after working native read tools.')
body('Reliable tools make a feedback loop possible. These regular fixtures already gave models an opportunity to inspect files, patch code, run public tests, and revise within the bounded task budget. Low final coding accuracy cannot be dismissed as having had no chance to iterate.')


page('The actual 16K memory boundary', 'Quantization experiments')
table(['Model / weight file','File size\nGB decimal','Measured free\nheadroom MiB','16K verdict'],[
 ['Qwen3.8 27B installed IQ3_S','12.04','3,448','Fits; initial capacity probe passed'],
 ['Qwen3.8 27B UD-Q4_K_M','16.46','290','Rejected; full-prompt request timed out'],
 ['Qwen3.8 27B UD-Q4_K_S','15.36','364','Rejected; full-prompt request timed out'],
 ['Qwen3.8 27B UD-IQ4_XS','14.25','1,354','Fits; all nine markers; user stopped coding'],
 ['Qwen3.6 27B IQ4_XS','15.44','312','Rejected; full-prompt request timed out'],
 ['Qwen3.6 27B UD-Q3_K_XL','14.47','1,164','Fits; all nine markers; 8/18 coding passes'],
],[230,73,107,118],True)
body('The largest tested Qwen3.6 file that met the specific 16K fit rule was <b>UD-Q3_K_XL</b>: 8/18 coding passes, nine tool failures and one ordinary coding failure. The fitting Qwen3.8 IQ4_XS trial had 6/11 completed passes, four tool failures and one coding failure before the user stopped it. Neither result improves the evidence for a qualified agent.')
sub('Why disk size did not predict headroom')
body('GGUF size is a disk representation. CUDA weight buffers, KV cache, recurrent state where applicable, compute buffers, runtime allocations, and desktop activity all contribute to residency. For Qwen3.6 IQ4_XS, the CUDA model buffer was about 14,032 MiB; at 16K its f16 KV cache added 1,024 MiB, plus about 150 MiB recurrent state and 138 MiB compute buffer.')
body('For Qwen3.6 Q3_K_XL, the CUDA weight buffer fell to about 13,110 MiB. That reduction restored sufficient measured headroom. A smaller file can still sit near the memory limit if the remaining runtime allocations consume the difference.')
sub('The reserved-memory correction')
body('NVML reported 16,380 MiB total, but used plus free equaled about 16,110 MiB, leaving 270 MiB reserved. Fit now subtracts that reserve and uses the smaller of reported free memory and available memory minus peak usage. The earlier total-minus-used calculation overstated headroom.')
note('Headroom is measured for the recorded probe and background workload. It is not a promise that a changing desktop workload cannot consume it. Shared WDDM allocation is reported as a diagnostic delta; it alone does not prove weight spill or paging.')


page('Shorter context helps fit and fresh prefill', 'Interpretation of the 16K pivot')
box('<b>Context is measured in tokens, not kilobytes.</b> The configured window must hold fully templated input and output together. The benchmark uses 61,440 + 4,096 at 64K, or 12,288 + 4,096 at 16K.')
sub('There are two useful effects')
body('<b>More room for weights:</b> a smaller context window reduces the KV footprint for these settings, which can enable a higher quantization or additional draft weights to remain on the GPU. <b>Less fresh input to process:</b> a genuinely smaller prompt also reduces prefill work. The second benefit exists even when the model weights and quantization remain unchanged.')
body('Merely changing the configured maximum while sending the same short prompt does not establish a fresh-latency improvement. Conversely, a model can fit a large configured window and still take a long time to process a nearly full, uncached input.')
sub('The measured counterexample to file-size intuition')
body('A roughly 10.00 GB Qwen3.6 IQ2_M file could load at 64K with f16 KV and pass capacity. A 15.44 GB Qwen3.6 IQ4_XS file failed the 16K headroom rule. That is consistent: the higher-precision weights consumed the memory recovered from the smaller cache, along with runtime overhead.')
sub('Higher quantization did not establish higher agent quality')
body('The new 16K trials changed both the context regime and weight representation. They were not a randomized identical-settings quality comparison. Their measured result is narrower: the fitting Qwen3.6 Q3_K_XL stack failed the regular coding gate, and the Qwen3.8 IQ4_XS stack was stopped before full coverage. There is no measured crossover where those tested stacks became qualified coding agents.')
sub('Capacity is not long-context coding')
body('Retrieving nine markers confirms that the prompt reached the model and it can extract the specified evidence. Long coding adds reasoning about distributed source facts and producing a correct final action. Qwen3.6 IQ2_M passed retrieval capacity yet failed enough long coding tasks to eliminate it.')
note('The user explicitly authorized separate 16K experiments. The original 64K grading contract remains intact, and no 16K result is relabeled as a 64K success.')


page('Ternary Bonsai: real memory result, limited agent evidence', 'Experimental compression')
table(['Observation','Measured result'],[
 ['Checkpoint / representation','prism-ml/Ternary-Bonsai-2-27B-gguf, PTQ1_0'],
 ['Downloaded weight size','5,946,648,928 bytes (5.95 GB decimal)'],
 ['Required runtime','Prism ternary fork b10709, commit 9a9394a895b96003ca842a6041cb28ac49a108f7'],
 ['GPU / context','All 65 layers offloaded; configured 65,536 tokens'],
 ['64K capacity','Three fresh full-prompt requests; all nine marker values passed'],
 ['Measured peak GPU use','10,426-10,554 MiB in full-context capacity records'],
 ['Coding smoke','3/5 completed passes; 2 malformed/unpermitted tool-stream failures'],
 ['Stop status','Stopped by user; sixth smoke record interrupted, not a completed failure'],
],[158,370])
table(['Fresh input','First model output (s)','First useful action (s)','Decode tok/s'],[
 ['16,384 tokens / small probe','40.665','54.089','27.487'],
 ['61,440 tokens / 64K, n=3','167.475-169.217','187.510-192.085','21.461-21.477'],
],[160,139,145,84],True)
note('Both Bonsai rows used a configured 65,536-token window; the small probe had 16,384 input tokens. These are capacity-request measurements, not the regular timing workload or a qualifying sustained decode comparison. No cached latency was measured. The three full-context capacity intervals had no flagged foreign GPU overlap.')
body('Bonsai delivered an unusually small weight file and comfortable memory residency at 64K. It did <b>not</b> deliver a fast fresh useful action on this host: nearly full uncached input needed about three minutes before the valid capacity action. The regular coding, long coding, stability, and demo contracts were not completed.')
body('I would retain it as a memory-efficiency research candidate, especially if a compatible tool/template configuration can be tested. I would not choose it as the everyday coding agent based on this evidence.')
note('Pinned model revision: 6ed5e12bf84b7a63069882c91dd9e9218647d17b. Weight SHA256: 53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3. The primary native llama.cpp engine was not substituted for its required custom kernel support.')


page('MTP and DFlash: real drafting, no qualifying improvement', 'Speculative decoding')
table(['Target KV','Mode','64K\nmarkers','Clean capacity\ntimings','Coding\nsmoke','Valid fresh\ntimings'],[
 ['q8_0 / q8_0','Off','9/9','3/3','4/6','3'],
 ['q8_0 / q8_0','MTP','9/9','1/3','3/6','0'],
 ['q8_0 / q8_0','DFlash','9/9','0/3','5/6','0'],
 ['q4_0 / q4_0','Off','9/9','3/3','3/6','3'],
 ['q4_0 / q4_0','MTP','9/9','2/3','3/6','0'],
 ['q4_0 / q4_0','DFlash','9/9','1/3','3/6','0'],
],[111,84,61,105,74,93],True)
body('The target was Qwen3.6 27B IQ2_M. Native counters confirmed real drafting: <b>MTP accepted 90.4-92.1%</b> of drafted tokens across capacity records; <b>DFlash accepted 58.4-58.9%</b>. MTP used 1.680 GB Q4_0 heads and width two. DFlash used a 1.849 GB Q8_0 drafter and comparison width seven. Draft cache was q8_0/q8_0.')
body('Clean auxiliary capacity decoding reached <b>24.22-25.49 tok/s with MTP</b> and <b>36.71 tok/s for one DFlash q4-cache sample</b>. This is a promising decode signal with small sample counts, not a demonstrated fresh-action improvement or coding qualification.')
body('All six arms passed retrieval capacity. None of the speculative arms passed its complete screen. Foreign GPU overlap made many speculative timing intervals diagnostic only; those values are excluded from clean speed comparisons. The arms also failed required coding/action screening gates, so a capacity pass cannot establish a usable agent.')
sub('What I conclude')
body('Speculation remains a plausible decoding optimization, but this run did not verify that it improves the desired fresh useful-action metric. It adds weight and KV memory pressure, and the model still has to process a long fresh prompt before decoding. A higher output tok/s value cannot by itself rescue delayed or invalid tools.')
sub('What I would test if this search continues')
body('First select a target configuration with reliable tools and adequate quality. Then compare off versus MTP versus DFlash under the same target weights, prompt, sampling, and quiet GPU conditions, preserving native acceptance counters and memory measurements. Only surviving arms should receive full regular/long qualification and final 20+20 timing.')
note('Source: artifacts/speculation-measured-results.json/.md and speculation-comparison-plan.json. The report labels these as screening results, not full coding qualification. Exact arm settings, counters, attempt IDs, and launch arguments are retained in the embedded focused audit and local raw export.')


page('Engine coverage and barriers', 'Availability is different from losing')
table(['Engine / version','Observed local outcome','What remains unproven'],[
 ['Upstream llama.cpp b11040\n5b335f413e4f...','Ran the strongest Qwen3.6 baseline, OxCoder, speculative arms, and all six new 16K family models.','No model passed the complete quality/stability contract.'],
 ['Bundled llama.cpp b10472\n7a556b8f9...','Matched Qwen3.6 regular quality; faster three-sample fresh-action screen.','No full-context coding winner or final p95.'],
 ['CUDA TurboQuant fork\n8ed935a092ee...','Isolated CUDA runtime exercised; compressed-cache and control profiles entered the registry.','No complete qualifying tool/coding/long/stability result.'],
 ['ik_llama.cpp','Container preflight failed to load libcuda.so.1.','No comparable successful model benchmark.'],
 ['ExLlamaV3 / TabbyAPI','ExLlamaV3 1.5.0+cu128.torch2.9.0; Tabby 53da7919d4e...; model startup lacked preprocessor_config.json.','Performance and quality are unavailable for that stack.'],
 ['vLLM v0.29.0','Recorded Docker storage headroom barrier; probe budget later exhausted.','No successful comparable local endpoint measurement.'],
 ['SGLang v0.5.19','Recorded Docker storage headroom barrier; probe budget later exhausted.','No successful comparable local endpoint measurement.'],
 ['LM Studio','Authenticated comparison was attempted; a pending load could not be confirmed complete. Later probe budget exhausted.','No qualified LM Studio measurement; not an authentication-unavailable label.'],
 ['Ollama','Comparison encountered unresolved owned managed-load state; probe budget later exhausted.','No qualifying comparative agent result.'],
 ['Prism ternary fork\nb10709 / 9a9394a895...','Ran Bonsai full-offload 64K capacity and five completed coding smokes.','User stopped before qualification.'],
],[138,230,160],True)
body('Many earlier attempts were delayed by an unresolved benchmark-owned managed load. The controller refused to start another GPU workload while ownership was uncertain. Those availability records do not make an engine a measured performance loser.')
note('Discovery versions are pinned historical leads, not a claim that each engine\'s newest release was installed or fully benchmarked. Exact binary, package, source and container pins are recorded in the source artifacts. No driver, global environment, WSL, clocks, or personal repositories were changed to force a comparison.')


page('Failure diagnosis and confidence limits', 'Read the failure before blaming the model')
sub('GPT-OSS: patch dialect and a separate native parser issue')
body('The first four GPT-OSS coding runs emitted native list_files and read_file calls successfully. The next apply_patch calls supplied JSON-valid arguments containing Codex-style <font face="Courier">*** Begin Patch</font> / <font face="Courier">*** Update File</font> envelopes. The frozen executor requires <font face="Courier">---</font> / <font face="Courier">+++</font> numbered unified hunks and rejected that format.')
body('In a separate capacity request, the native Harmony parser failed to parse a final constrained-tool output. Two earlier seeded requests succeeded. These are concrete integration failures in the tested stack. An isolated prompt/template/parser experiment could be useful, but must rerun the same fixtures and grading rules; it cannot silently credit unsupported patches as passes.')
sub('Missing counters were not necessarily tokenizer defects')
body('Earlier Devstral and Ministral labels such as small_context_token_count_mismatch followed requests that timed out without final token counters. GPT-OSS also had an engine error that ended the stream before usage appeared. The label describes failed evidence validation; the underlying error is needed to diagnose the cause. Both Devstral and Ministral later passed all nine capacity markers at 16K.')
sub('Historical harness issues were identified and corrected')
body('A directory spelling bug rejected native list_files arguments containing <font face="Courier">src/</font> while accepting <font face="Courier">src</font>. The source-directory validation was repaired under a new protocol identity, and subsequent tests preserved traversal restrictions. Historical records remain visible rather than being rewritten as successes.')
body('A prolonged Qwen3.5 reasoning stream was investigated and found to be actively producing reasoning, rather than a stalled HTTP read or demonstrated OOM. Fast first reasoning did not mean fast first useful code. Interrupted attempts remain incomplete, and raw byte/event counts were not converted into token throughput.')
sub('Limits that materially affect the recommendations')
body('No candidate reached the complete 20 fresh + 20 cached + demo qualification sequence. Many engines were unavailable or bounded out. Some coding runs were stopped early by mathematical elimination, others by the user. Quantization/context comparisons changed more than one setting. GPU overlap and incomplete attribution make flagged timing diagnostic. This is a finite local search, not a universal model leaderboard.')
note('Several older focused results.md files retain a generic Qwen3.8 heading even for Qwen3.6 wrappers. This PDF uses JSON configuration identities and verified model paths, not those inherited headings.')


page('My picks and the next useful experiment', 'Judgment grounded in these measurements')
sub('1. Qwen3.6 27B IQ2_M: quality-first supervised baseline')
body('It is the only tested baseline in this search that completed and passed the regular 27/36 threshold: 28/36 on both native engines. Its upstream tool-error rate was 1/36. I would retain upstream as the reliability reference and bundled as the faster candidate from the small fresh screen. The speed difference is not a final p95 ranking. Its failed long coding means I would keep supplied repository context focused and verify changes with tests.')
sub('2. OxCoder 9B Q4_K_M / upstream: faster, repairable loop')
body('Zero tool errors in 21 completed coding tasks is especially relevant to the stated preference that broken tools are worse than wrong code. Its 37.62-37.81-second fresh full-context useful-action screen and 31.79 tok/s sustained sample were substantially more responsive than the large dense Qwen baseline. It still failed ten deterministic coding tasks. I would use it for supervised small edits and debugging, not unattended acceptance of changes.')
sub('3. GPT-OSS 20B: best concrete interface-repair lead')
body('The tested configuration failed the four-run rule and should remain disqualified. Still, the saved calls point to a specific patch-format mismatch after working file tools, rather than an absence of native tool ability. A pinned configuration that clearly follows the advertised unified-diff contract, with compatible native output parsing, deserves a small four-run retest before spending another full sweep.')
sub('4. Bonsai: keep for memory research, not responsiveness')
body('Its 5.95 GB file and demonstrated 64K GPU residency make it informative. Its nearly three-minute fresh full-context useful action and two tool failures in five completed smokes make it a poor current everyday coding pick. Better compression was real; a better agent was not established.')
sub('What I would pause or deprioritize')
body('I would not repeat the same failing configurations. Qwen3.8 had repeated malformed/unpermitted calls across tested quantizations. The new family models failed regular quality well before long/stability testing. MTP and DFlash should wait for a reliable target and quiet timing intervals. Unavailable engines need a concrete barrier fix before another performance comparison.')
box('<b>Highest-value continuation:</b> a bounded GPT-OSS interface retest, or an iteration-focused OxCoder experiment, using the same hidden tests and an explicit new configuration identity. Broader engine and model discovery remains possible within the shared budgets, but there is no measured justification for restarting the whole search.')


page('Reproducing the two practical baselines', 'Exact measured launch settings')
body('These are recorded native launch settings, not a claim that an agent endpoint is currently running or qualified. Native model servers expose inference; the benchmark controller supplies tool execution, workspace restrictions, and grading.')
sub('Qwen3.6 IQ2_M / upstream b11040')
args = Q['launch_argv']
def show_command(argv):
    # A PowerShell splat avoids quoting ambiguity in the printed reproduction recipe.
    lines = ['$engineArgs = @(']
    for i, token in enumerate(argv[1:]):
        value = '"' + token.replace('\\','/').replace('"','\\"') + '"'
        lines.append('  '+value+(',' if i < len(argv)-2 else ''))
    lines += [')', '& "'+argv[0].replace('\\','/')+'" @engineArgs']
    story.append(Preformatted('\n'.join(lines), styles['CodeCustom']))
# Compact exact argument pairs, with quoted model path separately to keep the recipe on one page.
binary = args[0].replace('\\','/')
modelpath = Q['model_path'].replace('\\','/')
cmd = '\n'.join([
 '$model = "'+modelpath+'"',
 '$engineArgs = @(',
 '  "--model", $model, "--ctx-size", "65536", "--parallel", "1",',
 '  "--n-gpu-layers", "99", "--flash-attn", "on",',
 '  "--cache-type-k", "f16", "--cache-type-v", "f16",',
 '  "--batch-size", "2048", "--ubatch-size", "512",',
 '  "--threads", "16", "--threads-batch", "16",',
 '  "--host", "127.0.0.1", "--port", "38201", "--jinja",',
 '  "--no-context-shift", "--device", "CUDA0", "--verbosity", "4",',
 '  "--cors-origins", "http://127.0.0.1:38201",',
 '  "--spec-type", "none", "--metrics", "--slots"',
 ')',
 '& "'+binary+'" @engineArgs',
])
story.append(Preformatted(cmd,styles['CodeCustom']))
note('Long filesystem lines in the recipe may wrap in the viewer; the exact recorded argv array is embedded in original-audit.json. Launch only one model on the GPU and confirm the port is free.')
sub('OxCoder 9B Q4_K_M / the same upstream engine')
body('Use its recorded model file below with the same displayed upstream context/cache/server settings; the embedded audit retains the full exact argv for this configuration.')
story.append(p(escape(OX['model_path'].replace('\\','/')), 'CodeCustom'))
body('Request sampling for both references: temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence_penalty 0, repeat_penalty 1, with recorded deterministic seeds. Reasoning mode default; speculative decoding off.')
note('Qwen upstream config: '+Q['configuration_id']+'. Model SHA256: '+Q['model_sha256']+'.')
note('OxCoder config: '+OX['configuration_id']+'. Model SHA256: '+OX['model_sha256']+'.')


page('Host, budgets, provenance, and raw evidence', 'Audit trail')
table(['Item','Recorded value'],[
 ['GPU / driver','NVIDIA RTX 4060 Ti 16 GB; driver 591.86; total 16,380 MiB; available used+free about 16,110 MiB'],
 ['CPU / memory','AMD Threadripper 1950X, 16 cores / 32 threads; 128 GiB system RAM'],
 ['Execution clock','First runtime probe: 2026-09-18 17:46:39 UTC. Immutable deadline: 2026-09-25 17:46:39 UTC. 168-hour elapsed cap.'],
 ['New model weight ledger','166,804,714,750 bytes across 23 acquired reservations. 133,195,285,250 bytes remain below the 300 GB decimal cap.'],
 ['Downloads / single GPU','User-selected E:/modelmadness; installed files reused before new weights. GPU trials sequential.'],
 ['Harness verification','Original speculation harness: 588 tests, one skip. Expanded 16K parent: 597 tests, one skip; focused wrapper scope checks passed.'],
 ['Source snapshot','Private GitHub repository barretts/local-llm-benchmark; initial source commit 6df0b36044255269656bcaaa1bcc615a1923c128.'],
 ['End of queue','All six family workers exit code 0; no active job; no benchmark model server. Final observed GPU baseline about 435 MiB at 33 C.'],
],[145,383],True)
sub('Primary local evidence')
for text in [
 '<b>Original search:</b> artifacts/ranking.json, recommendation.md, results.json, results.csv and report.html. Original export cutoff: '+ORIG['original_export_latest_finished_utc']+'.',
 '<b>Live durable record:</b> state/benchmark.sqlite3. The report read it with a read-only SQLite connection; it did not restart or change the benchmark.',
 '<b>16K families:</b> artifacts/families-16k/queue-status.json and each model\'s results.json, selection.json, plan.json and worker logs.',
 '<b>Weight fit trials:</b> artifacts/qwen38-16k/, qwen36-16k/, and qwen36-q3kxl-16k/. <b>Bonsai:</b> artifacts/bonsai-2-27b/ plus its saved database configuration.',
 '<b>Speculation:</b> artifacts/speculation-measured-results.json and speculation-comparison-plan.json. <b>Engine probes:</b> engine-manifest.json, engine-discovery.json and runtime/container pin files.',
 '<b>Raw streams and grading:</b> .logs/ and the local fixture/grader artifacts. These retain actual usage counters, SSE events, execution outcomes, resource samples and exact launch metadata.'
]:
    note(text)
body('Two compact JSON audit snapshots are attached to this PDF. They include configuration identities, versions, hashes, launch argv, per-run outcomes, and sample counts. Raw multi-megabyte streams and the complete database remain local. Credentials are excluded.')
note('External Reddit speed reports were discovery leads only. Different GPUs, short inputs, cache restoration, or decode-only claims do not establish this machine\'s fresh full-context useful-action latency. No external headline speed is used to choose a winner in this report.')


# Full original inventory: readable tables with short unique IDs.
all_configs = ORIG['inventory']
for start in range(0,len(all_configs),13):
    part = all_configs[start:start+13]
    page('Original configuration inventory', f'Appendix | rows {start+1}-{start+len(part)} of {len(all_configs)}')
    rows=[]
    for c in part:
        cap = f"{c['capacity_seed_successes']}/3"
        n=c['regular']['attempted']
        tps = c['sustained_64k_throughput'].get('median')
        profile = c['engine_id'].replace('upstream-llama-nightly','upstream').replace('bundled-llama','bundled').replace('turboquant-cuda','TurboQuant')
        profile += '\n'+c['cache_k']+'/'+c['cache_v']
        if c['speculation'] not in ['off',None]:
            profile += '\n'+str(c['speculation'])
        if c['reasoning'] not in ['default',None]:
            profile += '\nreasoning '+str(c['reasoning'])
        rows.append([c['configuration_id'][:8]+'\n'+c['model_id'],profile,
                     f"{c['regular']['passed']}/{n}\nT={c['tool_errors_all_completed_regular']}",
                     f"{c['long']['passed']}/{c['long']['attempted']}",
                     cap,fmt(tps)])
    table(['ID / model','Engine / KV\nprofile','Coding\nT=tool errors','Long\npass/run','Capacity\nrequests','tok/s'],rows,[184,111,69,55,58,51],True)
    note('All rows are disqualified. Coding and long denominators are completed valid tasks; 0/0 means unmeasured, not zero accuracy. Capacity requests require three marker values each: 3/3 requests is all nine values. Throughput is only shown when the original report retained an exact sustained full-context sample. Reasoning and speculation variations are distinct configurations.')
    note('The complete 64-character IDs, exact settings, failure reasons, versions, hashes, sample counts and argv are in the attached original audit. The 13 later registered configurations are covered in the main report and focused audit. This inventory is not sorted as a qualified ranking.')

class NumberedCanvas(canvas.Canvas):
    def __init__(self,*args,**kwargs):
        canvas.Canvas.__init__(self,*args,**kwargs)
        self.saved=[]
    def showPage(self):
        self.saved.append(dict(self.__dict__))
        self._startPage()
    def save(self):
        count=len(self.saved)
        for state in self.saved:
            self.__dict__.update(state)
            self.setStrokeColor(PALE)
            self.line(42,35,570,35)
            self.setFont('Segoe',7.2)
            self.setFillColor(MUTED)
            self.drawString(42,22,'LOCAL CODING-AGENT BENCHMARK | MEASURED FINDINGS')
            self.drawRightString(570,22,f'{self._pageNumber} / {count}')
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

doc=SimpleDocTemplate(str(PDF),pagesize=letter,rightMargin=42,leftMargin=42,
    topMargin=39,bottomMargin=47,title='Local coding-agent benchmark: findings and practical picks',
    author='Local benchmark evidence review',subject='Measured results and provisional picks; no qualifying 64K or 16K winner')
doc.build(story,canvasmaker=NumberedCanvas)

# Attach redacted, scoped evidence rather than large raw log archives.
writer=PdfWriter()
writer.clone_document_from_reader(PdfReader(str(PDF)))
for name in ['original-audit.json','focused-audit.json']:
    writer.add_attachment(name,(TMP/name).read_bytes())
attached=TMP/'report-attached.pdf'
with attached.open('wb') as handle:
    writer.write(handle)
attached.replace(PDF)

render=TMP/'render'
render.mkdir(exist_ok=True)
pdf=pymupdf.open(str(PDF))
texts=[]
for i,page_obj in enumerate(pdf):
    texts.append(page_obj.get_text())
    pix=page_obj.get_pixmap(matrix=pymupdf.Matrix(1.25,1.25),alpha=False)
    pix.save(str(render/f'page-{i+1:02}.png'))
assert len(pdf)>=12
assert 'No configuration qualified' in texts[0]
assert len(PdfReader(str(PDF)).attachments)==2
sha=hashlib.sha256(PDF.read_bytes()).hexdigest()
(TMP/'pdf-validation.json').write_text(json.dumps({'pdf':str(PDF),'pages':len(pdf),'sha256':sha,
    'attachments':2,'rendered_pages':len(pdf),'page_characters':[len(t) for t in texts]},indent=2))
(TMP/'extracted-text.txt').write_text('\n\n'.join(texts),encoding='utf-8')
print(json.dumps({'pdf':str(PDF),'pages':len(pdf),'bytes':PDF.stat().st_size,'sha256':sha}))
