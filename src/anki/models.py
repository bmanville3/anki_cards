import genanki

from anki.css import SHARED_CSS


PARENT_DECK_ID = 9_876_543_210
MODEL_ID       = 1_234_567_897
VOCAB_MODEL_ID = 1_234_567_898


SENTENCE_MODEL = genanki.Model(
    MODEL_ID,
    "Japanese Video Sentence Cards",
    fields=[
        {"name": "Text"},
        {"name": "Audio"},
        {"name": "Image"},
        {"name": "NaturalTranslation"},
        {"name": "LiteralTranslation"},
        {"name": "TTSAudio"},
        {"name": "Furigana"},
        {"name": "WordGloss"},
        {"name": "TimeCode"},
        {"name": "FontName"},
        {"name": "Source"},
        {"name": "Notes"},
    ],
    templates=[{
        "name": "Sentence Card",
        "qfmt": (
            "<div class=\"jp-text\" style=\"font-family: '{{FontName}}', sans-serif;\">{{Text}}</div>\n"
            "{{Audio}}\n"
            "<div class=\"timecode\">{{TimeCode}}</div>\n"
        ),
        "afmt": (
            "{{FrontSide}}\n<hr>\n"
            "{{#Furigana}}\n"
            "<div class=\"jp-furigana\" style=\"font-family: '{{FontName}}', sans-serif;\">{{Furigana}}</div>\n"
            "{{/Furigana}}\n"
            "{{#NaturalTranslation}}\n"
            "<div class=\"translation\">{{NaturalTranslation}}</div>\n"
            "{{/NaturalTranslation}}\n"
            "{{TTSAudio}}\n"
            "<div class=\"frame-wrap\">{{Image}}</div>\n"
            "{{#WordGloss}}\n"
            "<div class=\"gloss-label\">Word by word</div>\n"
            "<div class=\"word-gloss\">{{WordGloss}}</div>\n"
            "{{/WordGloss}}\n"
            "{{#LiteralTranslation}}\n"
            "<div class=\"gloss-label\">Literal</div>\n"
            "<div class=\"translation\" style=\"color:#cba6f7;\">{{LiteralTranslation}}</div>\n"
            "{{/LiteralTranslation}}\n"
            "<div class=\"notes-label\">Notes</div>\n"
            "<div class=\"notes\">{{Notes}}</div>\n"
            "<div class=\"source-label\">Source</div>\n"
            "<div class=\"source-name\">{{Source}}</div>\n"
        ),
    }],
    css=SHARED_CSS,
)

VOCAB_MODEL = genanki.Model(
    VOCAB_MODEL_ID,
    "Japanese Video Vocab Cards",
    fields=[
        {"name": "Word"},          # JP word (shown on front)
        {"name": "WordAudio"},     # JP TTS audio tag
        {"name": "Reading"},       # hiragana reading
        {"name": "Meaning"},       # plain-text meanings for TTS + display
        {"name": "MeaningAudio"},  # EN TTS audio tag
        {"name": "WordGloss"},     # full HTML gloss (same as sentence back)
        {"name": "JLPT"},
        {"name": "FontName"},
        {"name": "Source"},
    ],
    templates=[{
        "name": "Vocab Card",
        "qfmt": (
            "<div class=\"vocab-word\" style=\"font-family: '{{FontName}}', sans-serif;\">{{Word}}</div>\n"
            "{{WordAudio}}\n"
        ),
        "afmt": (
            "{{FrontSide}}\n<hr>\n"
            "{{#Reading}}"
            "<div class=\"vocab-reading\">{{Reading}}</div>\n"
            "{{/Reading}}"
            "<div class=\"vocab-meaning\">{{Meaning}}</div>\n"
            "{{MeaningAudio}}\n"
            "{{#WordGloss}}\n"
            "<div class=\"vocab-label\">Full breakdown</div>\n"
            "<div class=\"word-gloss\">{{WordGloss}}</div>\n"
            "{{/WordGloss}}\n"
            "{{#JLPT}}<div class=\"jlpt-badge jlpt-{{JLPT}}\">{{JLPT}}</div>{{/JLPT}}\n"
            "<div class=\"source-label\">Source</div>\n"
            "<div class=\"source-name\">{{Source}}</div>\n"
        ),
    }],
    css=SHARED_CSS,
)
