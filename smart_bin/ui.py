"""
BinGenie - the browser demo page.

Kept out of `app.py` so the server stays readable: this module is only the
document (markup + CSS + a few hundred lines of vanilla JS).  No CDN, no build
step, no framework - the device is often offline and everything here must work
from a single HTTP response.

Design notes
------------
* The page answers three questions in order: what did I give it, what did it
  decide, and *why* - the last one matters more than the first two, because the
  whole argument of the project is that the decision is auditable.
* Four compartments, four accents.  The accent for a result is the colour of the
  compartment the item goes to, so the screen reads like the physical bin.
* The confidence gauge carries a tick at the operating threshold, so the
  fail-safe rule ("below 0.40 an item is refused") is something you can see
  rather than something you have to be told.
* Every state is explicit: empty, loading, result, error, and "no samples"
  (a checkout with no dataset still has a working demo).
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>BinGenie · on-device waste segregation</title>
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<style>
  :root {
    --bg: #0a0f0d;
    --bg-soft: #0f1614;
    --panel: #121b18;
    --panel-2: #16211d;
    --line: #22302b;
    --line-soft: #1a2621;
    --text: #e8f3ee;
    --text-dim: #9db3aa;
    --text-faint: #6d8479;
    --green: #34d399;
    --amber: #fbbf24;
    --orange: #fb923c;
    --red: #f87171;
    --accent: var(--green);
    --radius: 18px;
    --shadow: 0 18px 40px -28px rgba(0, 0, 0, .95);
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  }

  * { box-sizing: border-box; }

  html, body { height: 100%; }

  body {
    margin: 0;
    background:
      radial-gradient(1100px 620px at 12% -12%, rgba(52, 211, 153, .10), transparent 60%),
      radial-gradient(900px 520px at 100% 0%, rgba(251, 191, 36, .07), transparent 55%),
      var(--bg);
    color: var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Inter, sans-serif;
    font-size: 15px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }

  h1, h2, h3 { margin: 0; font-weight: 650; letter-spacing: -.01em; }
  p { margin: 0; }
  button { font: inherit; }
  a { color: var(--green); text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* ---------------------------------------------------------------- top bar */
  .topbar {
    position: sticky; top: 0; z-index: 20;
    display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
    padding: 14px clamp(16px, 4vw, 40px);
    background: rgba(10, 15, 13, .82);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--line-soft);
  }
  .brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
  .mark {
    width: 38px; height: 38px; flex: none; border-radius: 12px;
    background: linear-gradient(150deg, #14532d, #0b3b24);
    border: 1px solid #1d6a44;
    display: grid; place-items: center;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, .08);
  }
  .mark svg { width: 22px; height: 22px; }
  .brand h1 { font-size: 19px; letter-spacing: .01em; }
  .brand h1 em { font-style: normal; color: var(--green); }
  .brand .tagline { font-size: 12px; color: var(--text-faint); }

  .topbar-spacer { flex: 1 1 auto; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .chip {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 6px 11px; border-radius: 999px;
    background: var(--panel); border: 1px solid var(--line);
    font-size: 12px; color: var(--text-dim); white-space: nowrap;
  }
  .chip strong { color: var(--text); font-weight: 600; }
  .chip.link:hover { border-color: var(--green); text-decoration: none; }
  .dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--green);
    box-shadow: 0 0 0 3px rgba(52, 211, 153, .16);
  }
  .dot.warn { background: var(--amber); box-shadow: 0 0 0 3px rgba(251, 191, 36, .16); }

  /* ------------------------------------------------------------------ layout */
  .wrap {
    max-width: 1240px; margin: 0 auto;
    padding: clamp(16px, 3vw, 30px) clamp(16px, 4vw, 40px) 56px;
    display: grid; gap: 20px;
  }
  .columns { display: grid; grid-template-columns: minmax(340px, 1.02fr) minmax(360px, 1fr); gap: 20px; align-items: start; }
  @media (max-width: 980px) { .columns { grid-template-columns: 1fr; } }

  .panel {
    background: linear-gradient(180deg, rgba(255, 255, 255, .022), transparent 42%), var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 18px;
  }
  .panel-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
  .panel-head h2 { font-size: 13px; letter-spacing: .12em; text-transform: uppercase; color: var(--text-dim); }
  .panel-head .step { color: var(--text-faint); font-variant-numeric: tabular-nums; }
  .panel-head .hint { margin-left: auto; font-size: 12px; color: var(--text-faint); text-align: right; }
  h3 { font-size: 13px; color: var(--text-dim); letter-spacing: .02em; }

  /* ------------------------------------------------------------------- stage */
  .stage {
    position: relative; display: grid; place-items: center;
    min-height: 268px; padding: 18px;
    border: 1.5px dashed #2b4038; border-radius: 14px;
    background: var(--bg-soft);
    cursor: pointer; overflow: hidden;
    transition: border-color .15s ease, background .15s ease, transform .15s ease;
  }
  .stage:hover, .stage:focus-visible { border-color: #3c6b56; background: #101a16; outline: none; }
  .stage.hover { border-color: var(--green); background: #0e1f18; }
  .stage.has-image { border-style: solid; border-color: var(--line); padding: 10px; }
  .stage img {
    max-width: 100%; max-height: 360px; border-radius: 10px; display: block;
    object-fit: contain;
  }
  .stage-empty { text-align: center; display: grid; gap: 8px; justify-items: center; padding: 12px; }
  .stage-empty .icon {
    width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center;
    background: #14211c; border: 1px solid var(--line); margin-bottom: 2px;
  }
  .stage-empty .icon svg { width: 22px; height: 22px; stroke: var(--text-dim); }
  .stage-empty strong { color: var(--text); }
  .stage-empty .hint { font-size: 12.5px; color: var(--text-faint); max-width: 34ch; }
  .linkbtn {
    background: none; border: 0; padding: 0; color: var(--green);
    cursor: pointer; text-decoration: underline; text-underline-offset: 2px;
  }

  .stage-loading {
    position: absolute; inset: 0; display: grid; place-content: center; gap: 10px;
    justify-items: center; background: rgba(10, 15, 13, .74); backdrop-filter: blur(2px);
    font-size: 13px; color: var(--text-dim); z-index: 3;
  }
  .spinner {
    width: 22px; height: 22px; border-radius: 50%;
    border: 2px solid rgba(52, 211, 153, .25); border-top-color: var(--green);
    animation: spin .7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ---------------------------------------------------------------- controls */
  .controls { display: flex; gap: 9px; flex-wrap: wrap; margin-top: 14px; }
  .btn {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 9px 14px; border-radius: 11px; cursor: pointer;
    background: var(--panel-2); color: var(--text);
    border: 1px solid var(--line);
    font-size: 13.5px; font-weight: 550;
    transition: border-color .15s ease, background .15s ease, transform .06s ease;
  }
  .btn:hover { border-color: #375b4c; background: #18251f; }
  .btn:active { transform: translateY(1px); }
  .btn:focus-visible { outline: 2px solid var(--green); outline-offset: 2px; }
  .btn svg { width: 15px; height: 15px; stroke: currentColor; fill: none; stroke-width: 1.7; }
  .btn[disabled] { opacity: .45; cursor: not-allowed; }
  .btn-primary {
    background: linear-gradient(180deg, #1c6b48, #15543a);
    border-color: #249063; color: #eafff4;
  }
  .btn-primary:hover { background: linear-gradient(180deg, #22855a, #17614a); border-color: #2fae7a; }

  /* --------------------------------------------------------------- threshold */
  .threshold { margin-top: 16px; padding: 13px 14px; border-radius: 13px; background: var(--bg-soft); border: 1px solid var(--line-soft); }
  .threshold-top { display: flex; align-items: baseline; gap: 8px; }
  .threshold-top label { font-size: 13px; color: var(--text); font-weight: 550; }
  .threshold-top .hint { font-size: 12px; color: var(--text-faint); margin-left: auto; }
  .threshold-row { display: flex; align-items: center; gap: 12px; margin-top: 8px; }
  input[type="range"] {
    -webkit-appearance: none; appearance: none; flex: 1;
    height: 4px; border-radius: 99px; background: #253830; cursor: pointer;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%;
    background: var(--green); border: 3px solid #0d1512; box-shadow: 0 0 0 1px var(--green);
  }
  input[type="range"]::-moz-range-thumb {
    width: 14px; height: 14px; border-radius: 50%; background: var(--green);
    border: 3px solid #0d1512; box-shadow: 0 0 0 1px var(--green);
  }
  input[type="range"]:focus-visible { outline: 2px solid var(--green); outline-offset: 4px; }
  output {
    font-family: var(--mono); font-size: 13px; min-width: 42px; text-align: right;
    color: var(--text); font-variant-numeric: tabular-nums;
  }

  /* ----------------------------------------------------------------- samples */
  .samples { margin-top: 18px; }
  .samples-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
  .filters { display: flex; gap: 6px; flex-wrap: wrap; margin-left: auto; }
  .filter {
    padding: 4px 9px; border-radius: 999px; cursor: pointer;
    background: transparent; border: 1px solid var(--line);
    color: var(--text-faint); font-size: 11.5px; text-transform: capitalize;
  }
  .filter[aria-pressed="true"] { background: #17322a; border-color: #2c5a48; color: var(--text); }
  .filter:hover { color: var(--text); border-color: #375b4c; }

  .gallery {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(64px, 1fr));
    gap: 8px; max-height: 190px; overflow-y: auto; padding-right: 2px;
  }
  .gallery button {
    position: relative; padding: 0; border: 1px solid var(--line); border-radius: 10px;
    overflow: hidden; cursor: pointer; background: #0d1512; aspect-ratio: 1;
    transition: border-color .14s ease, transform .1s ease;
  }
  .gallery button img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .gallery button:hover { border-color: var(--green); transform: translateY(-1px); }
  .gallery button:focus-visible { outline: 2px solid var(--green); outline-offset: 2px; }
  .gallery button[aria-current="true"] { border-color: var(--green); box-shadow: 0 0 0 1px var(--green); }
  .gallery .cap {
    position: absolute; inset: auto 0 0 0; padding: 2px 4px;
    font-size: 9.5px; letter-spacing: .02em; text-transform: capitalize;
    background: linear-gradient(transparent, rgba(0, 0, 0, .82));
    color: #cfe6dc; text-align: left;
  }

  /* ------------------------------------------------------------------ result */
  .result { min-height: 250px; display: grid; }

  .result-empty { display: grid; align-content: center; gap: 14px; padding: 8px 4px; }
  .result-empty .lead { color: var(--text-dim); font-size: 14px; max-width: 44ch; }
  .steps { list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }
  .steps li { display: grid; grid-template-columns: 24px 1fr; gap: 10px; align-items: start; font-size: 13px; color: var(--text-faint); }
  .steps li b { color: var(--text-dim); font-weight: 600; }
  .steps .n {
    width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center;
    border: 1px solid var(--line); font-family: var(--mono); font-size: 11px; color: var(--text-dim);
  }

  .verdict {
    --accent: var(--green);
    display: grid; gap: 16px; padding: 18px;
    border-radius: 15px;
    border: 1px solid var(--accent-line, var(--line));
    background:
      radial-gradient(520px 200px at 0% 0%, var(--accent-glow, transparent), transparent 70%),
      var(--bg-soft);
    animation: rise .22s ease-out;
  }
  @keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

  .verdict-top { display: flex; align-items: flex-start; gap: 14px; flex-wrap: wrap; }
  .verdict-class { font-size: clamp(26px, 5vw, 34px); font-weight: 700; letter-spacing: -.02em; line-height: 1.05; color: var(--accent); text-transform: uppercase; }
  .verdict-sub { font-size: 13.5px; color: var(--text-dim); margin-top: 5px; }
  .verdict-sub b { color: var(--text); font-weight: 600; }
  .bin-badge {
    margin-left: auto; display: grid; gap: 2px; justify-items: center;
    padding: 9px 14px; border-radius: 12px; min-width: 96px;
    border: 1px solid var(--accent-line, var(--line));
    background: var(--accent-fill, #0b1210);
    font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--text-dim);
  }
  .bin-badge strong { font-size: 15px; letter-spacing: .04em; color: var(--accent); }

  .gauge { display: grid; gap: 6px; padding-top: 20px; }
  .gauge-track {
    position: relative; height: 10px; border-radius: 99px;
    background: #1b2a25; overflow: visible;
  }
  /* the tick's caption sits above the track; the padding on .gauge reserves it */
  .gauge-fill { height: 100%; border-radius: 99px; background: var(--accent); transition: width .45s cubic-bezier(.22, .61, .36, 1); }
  .gauge-tick {
    position: absolute; top: -4px; bottom: -4px; width: 2px; border-radius: 2px;
    background: #e8f3ee; opacity: .75;
  }
  .gauge-tick::after {
    content: "threshold"; position: absolute; top: -17px; left: 50%; transform: translateX(-50%);
    font-size: 9.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--text-faint); white-space: nowrap;
  }
  .gauge-legend { display: flex; justify-content: space-between; font-size: 11.5px; color: var(--text-faint); font-family: var(--mono); }

  .refused {
    display: flex; align-items: center; gap: 9px; padding: 9px 12px; border-radius: 11px;
    background: rgba(248, 113, 113, .09); border: 1px solid rgba(248, 113, 113, .32);
    font-size: 12.5px; color: #ffd7d7;
  }
  .refused svg { width: 15px; height: 15px; flex: none; stroke: var(--red); fill: none; stroke-width: 1.8; }
  .ok-note { font-size: 12.5px; color: var(--text-faint); }

  .facts { display: flex; gap: 8px; flex-wrap: wrap; }
  .fact {
    padding: 5px 10px; border-radius: 9px; background: #0d1512; border: 1px solid var(--line-soft);
    font-size: 11.5px; color: var(--text-dim); font-family: var(--mono);
  }
  .fact b { color: var(--text); font-weight: 600; }

  .reasons { list-style: none; margin: 0; padding: 0; display: grid; gap: 5px; }
  .reasons li { display: grid; grid-template-columns: 14px 1fr; gap: 8px; font-size: 12.5px; color: var(--text-dim); }
  .reasons li::before { content: "→"; color: var(--accent); }

  /* ----------------------------------------------------------------- ranking */
  .ranking { margin-top: 16px; }
  .bars { list-style: none; margin: 10px 0 0; padding: 0; display: grid; gap: 8px; }
  .bars li { display: grid; grid-template-columns: 74px 1fr 46px; gap: 10px; align-items: center; font-size: 12.5px; }
  .bars .name { color: var(--text-dim); text-transform: capitalize; overflow: hidden; text-overflow: ellipsis; }
  .bars .track { height: 7px; border-radius: 99px; background: #1b2a25; overflow: hidden; }
  .bars .fill { height: 100%; border-radius: 99px; background: #35564a; transition: width .45s cubic-bezier(.22, .61, .36, 1); }
  .bars li[data-top="true"] .name { color: var(--text); font-weight: 600; }
  .bars li[data-top="true"] .fill { background: var(--accent); }
  .bars .pct { text-align: right; font-family: var(--mono); font-size: 11.5px; color: var(--text-faint); }

  /* -------------------------------------------------------------- model card */
  .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
  .metric { padding: 14px; border-radius: 13px; background: var(--bg-soft); border: 1px solid var(--line-soft); }
  .metric .k { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--text-faint); }
  .metric .v { font-size: 21px; font-weight: 650; margin-top: 5px; letter-spacing: -.02em; }
  .metric .s { font-size: 11.5px; color: var(--text-faint); margin-top: 2px; }
  .metric.hero .v { color: var(--green); }

  .card-grid { display: grid; grid-template-columns: 1.15fr 1fr; gap: 18px; margin-top: 16px; align-items: start; }
  @media (max-width: 820px) { .card-grid { grid-template-columns: 1fr; } }

  .perclass { list-style: none; margin: 10px 0 0; padding: 0; display: grid; gap: 7px; }
  .perclass li { display: grid; grid-template-columns: 78px 1fr 62px; gap: 10px; align-items: center; font-size: 12px; }
  .perclass .name { color: var(--text-dim); text-transform: capitalize; }
  .perclass .track { height: 6px; border-radius: 99px; background: #1b2a25; overflow: hidden; }
  .perclass .fill { height: 100%; border-radius: 99px; background: #3c6b56; }
  .perclass .val { text-align: right; font-family: var(--mono); font-size: 11px; color: var(--text-faint); }

  .notes { font-size: 12.5px; color: var(--text-faint); display: grid; gap: 8px; }
  .notes b { color: var(--text-dim); font-weight: 600; }
  code {
    font-family: var(--mono); font-size: 11.5px; color: #bfe6d4;
    background: #0d1512; border: 1px solid var(--line-soft);
    padding: 1px 5px; border-radius: 6px;
  }

  /* --------------------------------------------------------------------- log */
  .log-wrap { overflow-x: auto; margin-top: 6px; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line-soft); white-space: nowrap; }
  th { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--text-faint); font-weight: 500; }
  td { color: var(--text-dim); font-variant-numeric: tabular-nums; }
  tbody tr:hover td { background: #101a16; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 10.5px; letter-spacing: .04em; text-transform: uppercase; }

  .empty-note { font-size: 12.5px; color: var(--text-faint); padding: 10px 0; }

  footer {
    max-width: 1240px; margin: 0 auto; padding: 0 clamp(16px, 4vw, 40px) 40px;
    font-size: 12px; color: var(--text-faint); display: grid; gap: 6px;
  }
  footer b { color: var(--text-dim); }

  .sr-only {
    position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
    overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0;
  }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: .001ms !important; transition-duration: .001ms !important; }
  }
</style>
</head>
<body>

<header class="topbar">
  <div class="brand">
    <span class="mark" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="#7ef0bb" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <path d="M5 8h14l-1.2 11.2a2 2 0 0 1-2 1.8H8.2a2 2 0 0 1-2-1.8Z"/>
        <path d="M9 8V5.8A1.8 1.8 0 0 1 10.8 4h2.4A1.8 1.8 0 0 1 15 5.8V8"/>
        <path d="M9.5 12v5M14.5 12v5"/>
      </svg>
    </span>
    <div>
      <h1>Bin<em>Genie</em></h1>
      <p class="tagline">On-device waste classification &middot; four-stream routing</p>
    </div>
  </div>
  <div class="topbar-spacer"></div>
  <div class="chips">
    <span class="chip" id="statusChip"><span class="dot" id="statusDot"></span><span id="statusText">connecting…</span></span>
    <span class="chip" id="backendChip" hidden><strong id="backendName">model</strong></span>
    <a class="chip link" href="#model">Model card</a>
  </div>
</header>

<div class="wrap">
  <div class="columns">

    <!-- ------------------------------------------------------------ input -->
    <section class="panel" aria-labelledby="h-present">
      <div class="panel-head">
        <h2 id="h-present"><span class="step">01</span> Present an item</h2>
        <span class="hint">processed on this machine</span>
      </div>

      <div class="stage" id="stage" role="button" tabindex="0"
           aria-label="Drop an image, paste from the clipboard, or press Enter to choose a file">
        <div class="stage-empty" id="stageEmpty">
          <span class="icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5"/>
              <path d="M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15"/>
            </svg>
          </span>
          <p><strong>Drop an image here</strong>, paste with <code>Ctrl</code>+<code>V</code>, or <span class="linkbtn">browse files</span></p>
          <p class="hint">JPEG or PNG, one object per frame. Nothing is uploaded anywhere.</p>
        </div>
        <img id="preview" alt="Item to classify" hidden>
        <div class="stage-loading" id="loading" hidden><span class="spinner" aria-hidden="true"></span>classifying…</div>
      </div>

      <div class="controls">
        <button type="button" class="btn btn-primary" id="browse">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h5l2 2h9v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2Z"/></svg>
          Choose image
        </button>
        <button type="button" class="btn" id="camera">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 8.5A2.5 2.5 0 0 1 5.5 6h1.2a2 2 0 0 0 1.7-1l.4-.7A1.6 1.6 0 0 1 10.2 3h3.6a1.6 1.6 0 0 1 1.4.8l.4.7a2 2 0 0 0 1.7 1h1.2A2.5 2.5 0 0 1 21 8.5v8A2.5 2.5 0 0 1 18.5 19h-13A2.5 2.5 0 0 1 3 16.5Z"/><circle cx="12" cy="12.2" r="3.3"/></svg>
          Camera
        </button>
        <button type="button" class="btn" id="random">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 3h5v5"/><path d="M21 3 13 11"/><path d="M8 21H3v-5"/><path d="M3 21l8-8"/><path d="M16 21h5v-5"/><path d="M14 14l7 7"/></svg>
          Random test image
        </button>
        <button type="button" class="btn" id="rerun" hidden>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2.6 6.1"/><path d="M20 4v7h-7"/></svg>
          Classify again
        </button>
        <input type="file" id="file" accept="image/*" hidden>
        <input type="file" id="fileCamera" accept="image/*" capture="environment" hidden>
      </div>

      <div class="threshold">
        <div class="threshold-top">
          <label for="thr">Confidence threshold</label>
          <span class="hint" id="thrHint">below this, an item is refused to RESIDUAL &middot; the product value is 0.40</span>
        </div>
        <div class="threshold-row">
          <input type="range" id="thr" min="20" max="99" step="1" value="40"
                 aria-describedby="thrHint">
          <output id="thrVal" for="thr">0.40</output>
        </div>
      </div>

      <div class="samples">
        <div class="samples-head">
          <h3>Held-out test images</h3>
          <div class="filters" id="filters" role="group" aria-label="Filter samples by class"></div>
        </div>
        <div class="gallery" id="gallery" aria-busy="true"></div>
        <div class="empty-note" id="galleryNote" hidden>No test images in this checkout - the classifier still works, the samples are simply not on disk.</div>
      </div>
    </section>

    <!-- ----------------------------------------------------------- result -->
    <section class="panel" aria-labelledby="h-decision">
      <div class="panel-head">
        <h2 id="h-decision"><span class="step">02</span> Decision</h2>
        <span class="hint" id="resultMeta"></span>
      </div>

      <div class="result" id="result">
        <div class="result-empty">
          <p class="lead">Present an item and the verdict appears here, together with the rule that produced it.</p>
          <ol class="steps">
            <li><span class="n">1</span><span><b>Camera.</b> The frame is resized to 128&times;128 and normalised exactly as during training and evaluation.</span></li>
            <li><span class="n">2</span><span><b>Classifier.</b> MobileNetV3-Small returns a probability for each of the six dataset classes.</span></li>
            <li><span class="n">3</span><span><b>Rule layer.</b> Class maps to one of four compartments; if the top probability is below the threshold, or two classes are too close, the item is refused to RED instead of guessed.</span></li>
          </ol>
        </div>
      </div>

      <div class="ranking" id="ranking" hidden>
        <h3>Class probabilities</h3>
        <ul class="bars" id="bars"></ul>
      </div>
    </section>
  </div>

  <!-- -------------------------------------------------------------- model card -->
  <section class="panel" id="model" aria-labelledby="h-model">
    <div class="panel-head">
      <h2 id="h-model"><span class="step">03</span> Model card</h2>
      <span class="hint" id="measuredOn"></span>
    </div>

    <div class="metrics" id="metrics"></div>

    <div class="card-grid">
      <div>
        <h3>Per-class recall</h3>
        <ul class="perclass" id="perclass"></ul>
      </div>
      <div class="notes" id="notes"></div>
    </div>
  </section>

  <!-- ------------------------------------------------------------------- log -->
  <section class="panel" aria-labelledby="h-log">
    <div class="panel-head">
      <h2 id="h-log">Decision log</h2>
      <span class="hint">one row per classified item, written to <code>logs/bingenie_decisions.csv</code></span>
    </div>
    <div class="log-wrap">
      <table>
        <thead>
          <tr><th>Time</th><th>Item</th><th>Compartment</th><th>Confidence</th><th>Outcome</th><th>Source</th></tr>
        </thead>
        <tbody id="logBody">
          <tr><td colspan="6" class="empty-note">nothing classified yet in this session</td></tr>
        </tbody>
      </table>
    </div>
  </section>
</div>

<footer>
  <div><b>Honest demo.</b> Numbers are measured, not illustrative: the accuracy figures come from the artefact this page is actually serving, scored on the published TrashNet test split that no model saw during training.</div>
  <div><b>Known limits.</b> TrashNet is studio photography on a white background with documented near-duplicates between splits; field accuracy on a real bin will be lower. A sixth of the classes (<code>trash</code>) has 29 test images, so its recall has a wide confidence interval.</div>
</footer>

<div class="sr-only" id="live" role="status" aria-live="polite"></div>

<script>
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // Hex, not CSS variables: the result panel needs tinted borders
  // ("#34d39955"), and `var(--green)55` is not a colour the browser can parse.
  const STREAM_COLOUR = {
    biodegradable: "#34d399",
    recyclable: "#fbbf24",
    electronic: "#fb923c",
    residual: "#f87171"
  };
  const tint = (hex, alpha) => hex + alpha;
  const STREAM_LABEL = {
    biodegradable: "Wet / organic",
    recyclable: "Dry recyclable",
    electronic: "Electronic / hazardous",
    residual: "Residual / hazardous"
  };

  const state = {
    image: null,          // data URL or sample URL
    sourceLabel: "",
    samples: [],
    filter: "all",
    threshold: 0.40,
    busy: false,
    lastResult: null
  };

  /* ------------------------------------------------------------- utilities */
  const pct = (value, digits = 1) => (value * 100).toFixed(digits);
  const esc = (text) => String(text).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
  const colourFor = (stream) => STREAM_COLOUR[stream] || "#9db3aa";
  const announce = (text) => { $("live").textContent = text; };

  let runToken = 0;

  /* --------------------------------------------------------------- images */
  function setImage(source, label, { classify = true } = {}) {
    state.image = source;
    state.sourceLabel = label || "";
    const preview = $("preview");
    preview.src = source;
    preview.hidden = false;
    preview.onload = () => {
      $("stageEmpty").hidden = true;
      $("stage").classList.add("has-image");
    };
    $("stageEmpty").hidden = true;
    $("stage").classList.add("has-image");
    markActiveSample(source);
    if (classify) run();
  }

  function loadFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      showError("That file is not an image. JPEG and PNG are supported.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setImage(reader.result, file.name);
    reader.onerror = () => showError("The browser could not read that file.");
    reader.readAsDataURL(file);
  }

  function markActiveSample(source) {
    document.querySelectorAll("#gallery button").forEach((button) => {
      button.setAttribute("aria-current", button.dataset.url === source ? "true" : "false");
    });
  }

  /* ------------------------------------------------------------- classify */
  async function run() {
    if (!state.image || state.busy) return;
    const token = ++runToken;
    state.busy = true;
    $("loading").hidden = false;
    $("result").setAttribute("aria-busy", "true");
    $("resultMeta").textContent = "";
    try {
      const response = await fetch("api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: state.image, threshold: state.threshold, name: state.sourceLabel })
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || ("HTTP " + response.status));
      }
      const data = await response.json();
      if (token !== runToken) return;           // a newer request already won
      state.lastResult = data;
      render(data);
      refreshLog();
    } catch (error) {
      if (token !== runToken) return;
      showError(error.message || String(error));
    } finally {
      if (token === runToken) {
        state.busy = false;
        $("loading").hidden = true;
        $("result").removeAttribute("aria-busy");
      }
    }
  }

  /* --------------------------------------------------------------- render */
  function render(data) {
    const d = data.decision;
    const colour = colourFor(d.stream);
    const refused = d.to_residual;
    const confident = pct(d.confidence);

    $("result").innerHTML = `
      <div class="verdict" style="--accent:${colour};--accent-line:${tint(colour, "55")};--accent-glow:${tint(colour, "1f")};--accent-fill:${tint(colour, "14")}">
        <div class="verdict-top">
          <div>
            <div class="verdict-class">${esc(d.label)}</div>
            <div class="verdict-sub">
              ${esc(STREAM_LABEL[d.stream] || d.stream)} &middot; slot ${d.slot} of the carousel &middot;
              <b>${confident}%</b> confident
            </div>
          </div>
          <div class="bin-badge">
            <span>compartment</span>
            <strong>${esc(d.compartment)}</strong>
          </div>
        </div>

        <div class="gauge">
          <div class="gauge-track">
            <div class="gauge-fill" style="width:${Math.max(2, d.confidence * 100)}%"></div>
            <div class="gauge-tick" style="left:${state.threshold * 100}%"></div>
          </div>
          <div class="gauge-legend"><span>0.00</span><span>top-1 probability</span><span>1.00</span></div>
        </div>

        ${refused ? `
        <div class="refused">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 8v5"/><path d="M12 16.5v.5"/><circle cx="12" cy="12" r="9"/></svg>
          <span><b>Refused, not guessed.</b> The fail-safe sent this item to the RED compartment; the classifier's own label is shown above but was not acted on.</span>
        </div>` : `
        <div class="ok-note">Above the ${state.threshold.toFixed(2)} threshold with a ${pct(d.margin)} margin over the runner-up, so the verdict was acted on.</div>`}

        <ul class="reasons">
          ${d.reasons.map((reason) => `<li><span>${esc(reason)}</span></li>`).join("")}
        </ul>

        <div class="facts">
          <span class="fact"><b>${confident}%</b> confidence</span>
          <span class="fact"><b>${pct(d.margin)}%</b> margin</span>
          <span class="fact"><b>${data.latency_ms.toFixed(0)}</b> ms</span>
          <span class="fact">${esc(data.runtime || "CPU")}</span>
          <span class="fact">${d.used_fusion ? "sensor fusion used" : "vision only"}</span>
        </div>
      </div>`;

    renderRanking(data.ranking, d, colour);
    $("resultMeta").textContent = state.sourceLabel
      ? state.sourceLabel + " · " + data.checkpoint
      : data.checkpoint;
    announce(`${d.label} to the ${d.compartment} compartment, ${confident} percent confident.`);
  }

  function renderRanking(ranking, decision, colour) {
    $("ranking").hidden = false;
    $("bars").style.setProperty("--accent", colour);
    $("bars").innerHTML = ranking.map(([name, probability], index) => `
      <li data-top="${index === 0}">
        <span class="name">${esc(name)}</span>
        <span class="track"><span class="fill" style="width:${(probability * 100).toFixed(1)}%"></span></span>
        <span class="pct">${pct(probability)}%</span>
      </li>`).join("");
  }

  function showError(message) {
    $("ranking").hidden = true;
    $("result").innerHTML = `
      <div class="verdict" style="--accent:#f87171;--accent-line:#f8717155;--accent-glow:#f871711f;--accent-fill:#f8717114">
        <div class="verdict-top">
          <div>
            <div class="verdict-class" style="font-size:22px">Could not classify</div>
            <div class="verdict-sub">${esc(message)}</div>
          </div>
        </div>
        <div class="facts"><span class="fact">try a smaller JPEG, or <b>Classify again</b></span></div>
      </div>`;
    announce("Error: " + message);
  }

  /* --------------------------------------------------------------- samples */
  function renderGallery() {
    const gallery = $("gallery");
    const shown = state.filter === "all"
      ? state.samples
      : state.samples.filter((sample) => sample.label === state.filter);

    if (!state.samples.length) {
      gallery.innerHTML = "";
      $("galleryNote").hidden = false;
      gallery.setAttribute("aria-busy", "false");
      return;
    }
    $("galleryNote").hidden = true;
    gallery.innerHTML = shown.map((sample) => `
      <button type="button" data-url="${esc(sample.url)}" data-label="${esc(sample.label)}"
              aria-current="false" title="${esc(sample.label)} - held-out test image">
        <img src="${esc(sample.url)}" alt="test image of ${esc(sample.label)}" loading="lazy">
        <span class="cap">${esc(sample.label)}</span>
      </button>`).join("");
    gallery.setAttribute("aria-busy", "false");
    markActiveSample(state.image);
  }

  function renderFilters() {
    const labels = [...new Set(state.samples.map((sample) => sample.label))].sort();
    const options = ["all", ...labels];
    $("filters").innerHTML = options.map((label) => `
      <button type="button" class="filter" data-filter="${esc(label)}"
              aria-pressed="${label === state.filter}">${esc(label)}</button>`).join("");
  }

  /* ------------------------------------------------------------ model card */
  function renderModelCard(info) {
    const metrics = info.metrics || {};
    const cards = [
      { key: "top-1 accuracy", value: metrics.top1_accuracy_pct, sub: metrics.top1_ci_text || "held-out test split", hero: true },
      { key: "routing accuracy", value: metrics.routing_accuracy_pct, sub: "correct compartment" },
      { key: "macro F1", value: metrics.macro_f1_pct, sub: "six classes, balanced" },
      { key: "top-3 accuracy", value: metrics.top3_accuracy_pct, sub: "right answer in top three" },
      { key: "refused to residual", value: metrics.residual_rate_pct, sub: "fail-safe, not a guess" },
      { key: "test images", value: metrics.images, sub: "never seen while training" }
    ];
    $("metrics").innerHTML = cards.filter((card) => card.value !== undefined && card.value !== null).map((card) => `
      <div class="metric${card.hero ? " hero" : ""}">
        <div class="k">${esc(card.key)}</div>
        <div class="v">${esc(card.value)}</div>
        <div class="s">${esc(card.sub)}</div>
      </div>`).join("");

    const perClass = metrics.per_class || [];
    $("perclass").innerHTML = perClass.map((row) => `
      <li>
        <span class="name">${esc(row.label)}</span>
        <span class="track"><span class="fill" style="width:${(row.recall * 100).toFixed(1)}%"></span></span>
        <span class="val">${pct(row.recall)}% <span style="opacity:.6">n=${row.support}</span></span>
      </li>`).join("") || '<li class="empty-note">per-class metrics unavailable</li>';

    $("notes").innerHTML = `
      <div><b>Artefact.</b> <code>${esc(info.checkpoint)}</code> &middot; ${esc(info.model)} &middot; ${esc(info.imgsz)}&nbsp;px input &middot; ${esc(info.backend)}</div>
      <div><b>Classes.</b> ${info.labels.map((label) => `<code>${esc(label)}</code>`).join(" ")}</div>
      <div><b>Measured on.</b> ${esc((info.metrics && info.metrics.measured_on) || "the PyTorch checkpoint")}, published TrashNet test partition.</div>
      <div><b>Why the second number is higher.</b> Five of the six classes share the yellow dry-recyclable compartment, so a plausible class confusion usually still reaches the correct bin.</div>
      ${info.note ? `<div><b>Scope.</b> ${esc(info.note)}</div>` : ""}`;

    $("measuredOn").textContent = (info.metrics && info.metrics.measured_on)
      ? "measured on " + info.metrics.measured_on
      : "";

    $("backendChip").hidden = false;
    $("backendName").textContent = info.backend || "model loaded";
    $("statusText").textContent = info.backend_kind === "onnx" ? "ONNX Runtime" : "PyTorch";
  }

  /* ------------------------------------------------------------------- log */
  async function refreshLog() {
    try {
      const response = await fetch("api/decisions");
      const data = await response.json();
      const rows = (data.rows || []).slice(-8).reverse();
      if (!rows.length) return;
      $("logBody").innerHTML = rows.map((row) => {
        const stream = String(row.stream || "").toLowerCase();
        const colour = colourFor(stream);
        const confidence = Number(row.confidence);
        const refused = String(row.to_residual).toLowerCase() === "true";
        return `<tr>
          <td>${esc((row.timestamp || row.time || "").replace("T", " ").replace("+00:00", "").slice(0, 19))}</td>
          <td>${esc(row.label || "—")}</td>
          <td><span class="tag" style="color:${colour};border:1px solid ${tint(colour, "55")}">${esc(row.compartment || stream || "—")}</span></td>
          <td>${Number.isFinite(confidence) ? pct(confidence) + "%" : "—"}</td>
          <td>${refused ? "refused → residual" : "accepted"}</td>
          <td>${esc(row.source || "—")}</td>
        </tr>`;
      }).join("");
    } catch (error) { /* the log is a convenience; never break the page for it */ }
  }

  /* ----------------------------------------------------------------- events */
  $("browse").addEventListener("click", () => $("file").click());
  $("file").addEventListener("change", (event) => {
    loadFile(event.target.files[0]);
    event.target.value = "";
  });
  $("fileCamera").addEventListener("change", (event) => {
    loadFile(event.target.files[0]);
    event.target.value = "";
  });

  $("camera").addEventListener("click", async () => {
    // A secure-context camera is the nice path; on the device or on localhost it
    // is available, inside a sandboxed iframe it may be blocked - fall back to the
    // OS camera picker rather than showing a dead button.
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("no camera API");
      }
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
      const track = stream.getVideoTracks()[0];
      const video = document.createElement("video");
      video.srcObject = stream;
      video.playsInline = true;
      await video.play();
      await new Promise((resolve) => setTimeout(resolve, 400));   // let exposure settle
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      track.stop();
      setImage(canvas.toDataURL("image/jpeg", 0.92), "camera capture");
    } catch (error) {
      $("fileCamera").click();   // mobile: opens the native camera
    }
  });

  $("stage").addEventListener("click", () => { if (!state.image) $("file").click(); });
  $("stage").addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      $("file").click();
    }
  });
  ["dragover", "dragenter"].forEach((name) => $("stage").addEventListener(name, (event) => {
    event.preventDefault();
    $("stage").classList.add("hover");
  }));
  ["dragleave", "dragend", "drop"].forEach((name) => $("stage").addEventListener(name, (event) => {
    event.preventDefault();
    $("stage").classList.remove("hover");
  }));
  $("stage").addEventListener("drop", (event) => {
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) loadFile(file);
  });
  window.addEventListener("paste", (event) => {
    const items = (event.clipboardData && event.clipboardData.items) || [];
    for (const item of items) {
      if (item.type && item.type.startsWith("image/")) {
        loadFile(item.getAsFile());
        event.preventDefault();
        return;
      }
    }
  });

  $("random").addEventListener("click", () => {
    if (!state.samples.length) return;
    const pool = state.filter === "all" ? state.samples : state.samples.filter((s) => s.label === state.filter);
    const pick = pool[Math.floor(Math.random() * pool.length)];
    setImage(pick.url, "test image (" + pick.label + ")");
  });

  $("rerun").addEventListener("click", () => run());

  let thresholdTimer = null;
  $("thr").addEventListener("input", (event) => {
    state.threshold = Number(event.target.value) / 100;
    $("thrVal").textContent = state.threshold.toFixed(2);
    $("rerun").hidden = !state.image;
    if (!state.image) return;
    clearTimeout(thresholdTimer);
    thresholdTimer = setTimeout(run, 220);   // live re-run, debounced
  });

  $("gallery").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-url]");
    if (!button) return;
    setImage(button.dataset.url, "test image (" + button.dataset.label + ")");
  });

  $("filters").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-filter]");
    if (!button) return;
    state.filter = button.dataset.filter;
    document.querySelectorAll("#filters .filter").forEach((chip) => {
      chip.setAttribute("aria-pressed", chip.dataset.filter === state.filter ? "true" : "false");
    });
    renderGallery();
  });

  /* ------------------------------------------------------------------- boot */
  async function boot() {
    try {
      const info = await (await fetch("api/info")).json();
      renderModelCard(info);
      $("statusDot").classList.remove("warn");
    } catch (error) {
      $("statusDot").classList.add("warn");
      $("statusText").textContent = "model unavailable";
      showError("The server is running but the model could not be loaded: " + error.message);
    }

    try {
      const data = await (await fetch("api/samples")).json();
      state.samples = data.samples || [];
      renderFilters();
      renderGallery();
    } catch (error) {
      $("galleryNote").hidden = false;
      $("galleryNote").textContent = "Test images could not be listed: " + error.message;
    }

    refreshLog();
    document.body.dataset.ready = "true";
  }

  boot();
})();
</script>
</body>
</html>
"""
