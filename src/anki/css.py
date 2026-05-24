from anki.fonts_helper import FONTS_DIR, build_font_face_css


FONT_FACE_CSS = build_font_face_css(FONTS_DIR)

SHARED_CSS = FONT_FACE_CSS + """
.card {
  font-size: 24px; text-align: center; background: #1e1e2e;
  color: #cdd6f4; padding: 24px 20px; line-height: 1.7;
}
.jp-text { font-size: 30px; margin-bottom: 10px; }
.jp-furigana { font-size: 26px; margin-bottom: 10px; line-height: 2.2; }
ruby rt { font-size: 0.45em; color: #89b4fa; }
.timecode { font-size: 11px; color: #585b70; margin-top: 6px; }
hr { border: none; border-top: 1px solid #313244; margin: 16px 0; }
.frame-wrap img { max-width: 100%; border-radius: 10px; margin-bottom: 12px; }
.translation { font-size: 18px; color: #a6e3a1; font-style: italic; margin-bottom: 10px; }
.gloss-label, .furigana-label, .notes-label, .source-label {
  font-size: 11px; color: #585b70; text-transform: uppercase;
  letter-spacing: 0.08em; margin-top: 14px;
}
.word-gloss {
  font-size: 14px; color: #cdd6f4; margin-top: 4px;
  line-height: 2.0; text-align: left; display: inline-block;
}
.gloss-word { font-size: 15px; font-weight: bold; color: #cdd6f4; padding-right: 6px; }
.gloss-line { display: block; margin-left: 8px; }
.pos-tag { font-size: 11px; opacity: 0.6; }
.pos-verb  { color: #89dceb; }
.pos-noun  { color: #cdd6f4; }
.pos-adj   { color: #a6e3a1; }
.pos-adv   { color: #f9e2af; }
.pos-expr  { color: #cba6f7; }
.pos-other { color: #bac2de; }
.jlpt-badge {
  display: inline-block; font-size: 10px; font-weight: bold;
  padding: 1px 5px; border-radius: 4px; margin-left: 5px;
  vertical-align: middle; color: #1e1e2e;
}
.jlpt-N5 { background: #a6e3a1; }
.jlpt-N4 { background: #89dceb; }
.jlpt-N3 { background: #f9e2af; }
.jlpt-N2 { background: #fab387; }
.jlpt-N1 { background: #f38ba8; }
.pa-container { display: inline-block; margin-left: 8px; vertical-align: middle; }
.pa-wrap { display: inline-flex; align-items: flex-end; font-size: 12px; gap: 0; }
.pa-high { color: #89b4fa; border-top: 2px solid #89b4fa; padding: 0 1px; }
.pa-low  { color: #89b4fa; border-top: 2px solid transparent; padding: 0 1px; }
.pa-drop { color: #f38ba8; font-size: 10px; margin: 0 1px; align-self: center; }
.pa-label { font-size: 10px; color: #585b70; margin-left: 4px; align-self: center; }
.source-name { font-size: 15px; color: #bac2de; min-height: 1.4em; margin-top: 4px; padding-bottom: 4px; }
.notes {
  font-size: 15px; color: #bac2de; min-height: 1.4em;
  border-bottom: 1px dashed #45475a; margin-top: 4px; padding-bottom: 4px;
}
.vocab-word { font-size: 38px; font-weight: bold; margin-bottom: 8px; }
.vocab-reading { font-size: 20px; color: #89b4fa; margin-bottom: 6px; }
.vocab-meaning { font-size: 18px; color: #a6e3a1; font-style: italic; margin-bottom: 10px; }
.vocab-pos { font-size: 12px; color: #585b70; margin-bottom: 4px; }
.vocab-label { font-size: 11px; color: #585b70; text-transform: uppercase;
  letter-spacing: 0.08em; margin-top: 14px; }
"""
