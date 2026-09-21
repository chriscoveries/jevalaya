//! Dependency-free language/script detection used to route between Laya
//! checkpoints. Exact port of `laya_mlx/lang.py`.
//!
//! Routing only needs one decision: *is this English Latin text, or is it
//! something the English checkpoint cannot read?* Script detection is exact;
//! the Latin-script language guess is a stopword/diacritic heuristic and is
//! explicitly best-effort.
//!
//! Parity notes vs CPython:
//! * `str.isalpha()` covers general categories L* only; `char::is_alphabetic`
//!   also counts Nl + Other_Alphabetic marks, so we classify via
//!   `unicode-general-category` to match exactly.
//! * `re.findall(r"[^\W\d_]+")` yields runs of `isalnum` minus `\d` minus
//!   `_` = categories L* + Nl + No.
//! * `max(counts.items(), key=count)` keeps the FIRST maximal item in dict
//!   insertion order: non-Latin scripts in first-encounter order, then
//!   `latin` last — so `latin` loses every tie.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use serde_json::Value;
use unicode_general_category::{get_general_category, GeneralCategory};

/// Unicode blocks that the English (ModernBERT-large, 50k English BPE)
/// checkpoint cannot read. Same ranges and order as `lang.py`.
const SCRIPT_RANGES: &[(&str, &[(u32, u32)])] = &[
    ("greek", &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    (
        "cyrillic",
        &[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    ),
    ("hebrew", &[(0x0590, 0x05FF)]),
    (
        "arabic",
        &[
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ],
    ),
    ("devanagari", &[(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("bengali", &[(0x0980, 0x09FF)]),
    ("gurmukhi", &[(0x0A00, 0x0A7F)]),
    ("gujarati", &[(0x0A80, 0x0AFF)]),
    ("oriya", &[(0x0B00, 0x0B7F)]),
    ("tamil", &[(0x0B80, 0x0BFF)]),
    ("telugu", &[(0x0C00, 0x0C7F)]),
    ("kannada", &[(0x0C80, 0x0CFF)]),
    ("malayalam", &[(0x0D00, 0x0D7F)]),
    ("sinhala", &[(0x0D80, 0x0DFF)]),
    ("thai", &[(0x0E00, 0x0E7F)]),
    ("lao", &[(0x0E80, 0x0EFF)]),
    ("tibetan", &[(0x0F00, 0x0FFF)]),
    ("myanmar", &[(0x1000, 0x109F)]),
    ("georgian", &[(0x10A0, 0x10FF)]),
    ("ethiopic", &[(0x1200, 0x137F)]),
    ("khmer", &[(0x1780, 0x17FF)]),
    (
        "hangul",
        &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    ),
    (
        "kana",
        &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    ),
    (
        "han",
        &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)],
    ),
];

/// Function words. Latin-script languages overlap heavily, so each hit is
/// weighted and a margin is required before calling something non-English.
const STOP: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
            "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
            "will", "can", "would", "there", "their", "what", "which", "please", "we", "i",
        ],
    ),
    (
        "fr",
        &[
            "le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
            "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce",
        ],
    ),
    (
        "de",
        &[
            "der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
            "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber",
        ],
    ),
    (
        "es",
        &[
            "el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
            "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus",
        ],
    ),
    (
        "pt",
        &[
            "os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
            "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela",
        ],
    ),
    (
        "it",
        &[
            "il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
            "questo", "questa", "anche", "come", "più", "sono", "nella", "alla",
        ],
    ),
    (
        "nl",
        &[
            "het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
            "door", "maar", "ook", "worden", "deze", "naar", "wordt",
        ],
    ),
];

fn stop_sets() -> &'static HashMap<&'static str, HashSet<&'static str>> {
    static SETS: OnceLock<HashMap<&'static str, HashSet<&'static str>>> = OnceLock::new();
    SETS.get_or_init(|| {
        STOP.iter()
            .map(|(lg, words)| (*lg, words.iter().copied().collect()))
            .collect()
    })
}

fn non_en_diacritics() -> &'static HashSet<char> {
    static SET: OnceLock<HashSet<char>> = OnceLock::new();
    SET.get_or_init(|| "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğıåäö".chars().collect())
}

/// Python `str.isalpha()`: general categories L* only.
fn is_py_alpha(ch: char) -> bool {
    matches!(
        get_general_category(ch),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

/// Python `[^\W\d_]` word char: `isalnum` minus `\d` minus `_`
/// = categories L* + Nl + No.
fn is_py_word_char(ch: char) -> bool {
    matches!(
        get_general_category(ch),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber
    )
}

fn script_of(cp: u32) -> Option<&'static str> {
    for (name, ranges) in SCRIPT_RANGES {
        if ranges.iter().any(|(lo, hi)| *lo <= cp && cp <= *hi) {
            return Some(name);
        }
    }
    None
}

/// Collect the string leaves of a state (str/object/list), so detection
/// sees real content. Mirrors `_iter_text`: dict values and list items
/// recurse, everything else is ignored, depth capped at 6.
fn iter_text<'a>(state: &'a Value, depth: usize, out: &mut Vec<&'a str>) {
    if depth > 6 {
        return;
    }
    match state {
        Value::String(s) => out.push(s.as_str()),
        Value::Object(m) => {
            for v in m.values() {
                iter_text(v, depth + 1, out);
            }
        }
        Value::Array(items) => {
            for v in items {
                iter_text(v, depth + 1, out);
            }
        }
        _ => {}
    }
}

/// Flatten a state into the text used for detection (keys are ignored:
/// they are usually English). `[:max_chars]` is a char (code point) slice,
/// matching Python's slice semantics.
pub fn state_text(state: &Value, max_chars: usize) -> String {
    let mut leaves = Vec::new();
    iter_text(state, 0, &mut leaves);
    let joined = leaves.join(" ");
    joined.chars().take(max_chars).collect()
}

/// Dominant script of `text`: `latin`, `han`, `devanagari`, ... or
/// `unknown` if there are no letters. Mirrors `detect_script` including
/// dict-order tie-breaking (first-encountered non-Latin wins ties;
/// `latin` is appended last and loses them).
pub fn detect_script(text: &str) -> String {
    let mut counts: HashMap<&'static str, usize> = HashMap::new();
    let mut order: Vec<&'static str> = Vec::new();
    let mut latin = 0usize;
    for ch in text.chars() {
        if !is_py_alpha(ch) {
            continue;
        }
        let cp = ch as u32;
        if cp < 0x0250 || (0x1E00..=0x1EFF).contains(&cp) {
            latin += 1;
            continue;
        }
        if let Some(name) = script_of(cp) {
            if !counts.contains_key(name) {
                order.push(name);
            }
            *counts.entry(name).or_insert(0) += 1;
        }
    }
    // Python: counts = {scripts in encounter order..., "latin": latin};
    // total == 0 -> "unknown"; else first maximal in dict order.
    let mut best: Option<(&'static str, usize)> = None;
    for name in &order {
        let c = counts[name];
        if best.is_none() || c > best.unwrap().1 {
            best = Some((name, c));
        }
    }
    match best {
        // latin is the LAST key: wins only on a strict majority.
        Some((name, c)) if latin <= c => name.to_string(),
        Some((_, c)) if latin > c => "latin".to_string(),
        Some((name, _)) => name.to_string(),
        None => {
            if latin > 0 {
                "latin".to_string()
            } else {
                "unknown".to_string()
            }
        }
    }
}

/// Fraction of alphabetic characters belonging to each detected script.
/// Mirrors `script_profile` (zero entries are dropped from the map).
pub fn script_profile(text: &str) -> HashMap<&'static str, f64> {
    let mut counts: HashMap<&'static str, usize> = HashMap::new();
    let mut latin = 0usize;
    for ch in text.chars() {
        if !is_py_alpha(ch) {
            continue;
        }
        let cp = ch as u32;
        if cp < 0x0250 || (0x1E00..=0x1EFF).contains(&cp) {
            latin += 1;
            continue;
        }
        if let Some(name) = script_of(cp) {
            *counts.entry(name).or_insert(0) += 1;
        }
    }
    let total: usize = latin + counts.values().sum::<usize>();
    if total == 0 {
        return HashMap::new();
    }
    let mut out: HashMap<&'static str, f64> = counts
        .iter()
        .map(|(k, v)| (*k, *v as f64 / total as f64))
        .collect();
    if latin > 0 {
        out.insert("latin", latin as f64 / total as f64);
    }
    out
}

fn word_tokens(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if is_py_word_char(ch) {
            cur.extend(ch.to_lowercase());
        } else if !cur.is_empty() {
            words.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words
}

/// Best-effort language code for Latin-script text, or None when
/// undecided. Mirrors `guess_latin_language`: a non-English language
/// needs a clear margin over English function words; short inputs
/// usually return None on purpose.
pub fn guess_latin_language(text: &str) -> Option<String> {
    let words = word_tokens(text);
    if words.len() < 4 {
        return None;
    }
    let sets = stop_sets();
    let mut scores: HashMap<&str, usize> = HashMap::new();
    for (lg, set) in sets {
        scores.insert(
            lg,
            words.iter().filter(|w| set.contains(w.as_str())).count(),
        );
    }
    let lowered = text.to_lowercase();
    let diacs = non_en_diacritics();
    let diac = lowered.chars().filter(|c| diacs.contains(c)).count();
    let diac_rate = diac as f64 / lowered.chars().count().max(1) as f64;
    let en = scores.get("en").copied().unwrap_or(0);
    // Python: max over _STOP insertion order (en,fr,de,es,pt,it,nl) minus
    // "en"; first maximal wins ties.
    let mut best: Option<(&str, usize)> = None;
    for (lg, _) in STOP.iter().filter(|(lg, _)| *lg != "en") {
        let s = scores.get(lg).copied().unwrap_or(0);
        if best.is_none() || s > best.unwrap().1 {
            best = Some((lg, s));
        }
    }
    let (best_lg, best_s) = best.unwrap_or(("", 0));
    if best_s == 0 && diac_rate < 0.02 {
        return if en > 0 { Some("en".to_string()) } else { None };
    }
    // a non-English language needs a clear margin over English function words
    if !best_lg.is_empty() && best_s >= std::cmp::max(2, en + 2) {
        return Some(best_lg.to_string());
    }
    if diac_rate >= 0.04 && !best_lg.is_empty() && best_s >= en {
        return Some(best_lg.to_string());
    }
    if en > 0 {
        Some("en".to_string())
    } else {
        None
    }
}

/// Full detection result for a state — mirrors `analyse`.
#[derive(Debug, Clone)]
pub struct Detection {
    /// `latin`, a script name, or `unknown` when the state has no letters.
    pub script: String,
    /// Fraction of alphabetic chars per detected script.
    pub script_profile: HashMap<&'static str, f64>,
    /// Best-effort Latin-script language guess, may be None.
    pub language: Option<String>,
    pub is_english: bool,
    /// `round(1 - latin_fraction, 4)` — 0.0 for unknown script.
    pub non_latin_fraction: f64,
}

pub fn analyse(state: &Value) -> Detection {
    let text = state_text(state, 4000);
    let prof = script_profile(&text);
    let script = detect_script(&text);
    let non_latin = if prof.is_empty() {
        0.0
    } else {
        // round(1 - latin, 4) like Python
        let v = 1.0 - prof.get("latin").copied().unwrap_or(0.0);
        (v * 10000.0).round() / 10000.0
    };
    if script == "unknown" {
        return Detection {
            script,
            script_profile: prof,
            language: None,
            is_english: true,
            non_latin_fraction: 0.0,
        };
    }
    if script != "latin" {
        return Detection {
            script,
            script_profile: prof,
            language: None,
            is_english: false,
            non_latin_fraction: non_latin,
        };
    }
    let lang = guess_latin_language(&text);
    Detection {
        script: "latin".to_string(),
        script_profile: prof,
        is_english: matches!(lang.as_deref(), None | Some("en")),
        language: lang,
        non_latin_fraction: non_latin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn english_latin_is_english() {
        let det = analyse(&json!("I was charged twice and want a refund on the order"));
        assert_eq!(det.script, "latin");
        assert!(det.is_english);
    }

    #[test]
    fn non_english_latin_routes_non_english() {
        let det = analyse(&json!(
            "Mein Konto wurde zweimal belastet und ich möchte eine Rückerstattung für die Bestellung"
        ));
        assert_eq!(det.script, "latin");
        assert_eq!(det.language.as_deref(), Some("de"));
        assert!(!det.is_english);
    }

    #[test]
    fn non_latin_scripts_detected() {
        assert_eq!(analyse(&json!("我想退款")).script, "han");
        assert_eq!(analyse(&json!("Я хочу вернуть деньги")).script, "cyrillic");
        assert_eq!(analyse(&json!("مرحبا بالعالم")).script, "arabic");
        assert!(!analyse(&json!("我想退款")).is_english);
    }

    #[test]
    fn no_letters_is_unknown() {
        let det = analyse(&json!("1234 !!! 56"));
        assert_eq!(det.script, "unknown");
        assert!(det.is_english);
        assert_eq!(det.non_latin_fraction, 0.0);
    }

    #[test]
    fn nested_state_leaves_only() {
        let det = analyse(&json!({
            "message": "Mein Konto wurde zweimal belastet",
            "meta": {"note": "und ich möchte eine Rückerstattung"},
            "n": 42
        }));
        assert_eq!(det.script, "latin");
        assert!(!det.is_english);
    }

    #[test]
    fn short_latin_is_undecided() {
        // <4 words -> None language -> treated as english-compatible
        let det = analyse(&json!("Bonjour"));
        assert_eq!(det.script, "latin");
        assert!(det.is_english);
    }

    #[test]
    fn mixed_state_dominant_script_wins() {
        let det = analyse(&json!("hello 中文字符中文字符"));
        assert_eq!(det.script, "han");
    }
}
