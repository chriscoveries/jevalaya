//! Python-`json.dumps`-exact JSON rendering for prompt construction.
//!
//! The canonical Laya renderer serialises structured criteria, states, and
//! non-string instructions with CPython `json.dumps(x, ensure_ascii=False)`.
//! The two load-bearing formatting facts:
//!
//! * default separators are `(', ', ': ')` — `{"reason": "no"}`, NOT the
//!   `{"reason":"no"}` that `serde_json::to_string` emits. Different bytes
//!   tokenize to different ids, which silently shifts the ANE token gate.
//! * `ensure_ascii=False` — non-ASCII is emitted raw (serde_json does this
//!   by default too).
//!
//! [`py_dumps`] replicates that spacing exactly for `serde_json::Value`.
//! (`default=str` fallback and dict ordering are handled by the caller:
//! values arrive already parsed, and `serde_json` is built with
//! `preserve_order` so object key order matches Python dict insertion order.)

use serde_json::Value;

/// Serialize `v` the way CPython `json.dumps(v, ensure_ascii=False)` does.
pub fn py_dumps(v: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, v);
    out
}

fn write_value(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        // serde_json string escaping (quotes, backslash, controls) matches
        // CPython ensure_ascii=False for all inputs representable in Rust str.
        Value::String(s) => out.push_str(&serde_json::to_string(s).expect("string escapes")),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, val)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::to_string(k).expect("key escapes"));
                out.push_str(": ");
                write_value(out, val);
            }
            out.push('}');
        }
    }
}

/// Serialize a prompt `state`: strings pass through raw, anything structured
/// becomes `py_dumps` (mirrors `serialize_state`).
pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        other => py_dumps(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn object_uses_python_separators() {
        assert_eq!(py_dumps(&json!({"reason": "no"})), r#"{"reason": "no"}"#);
        assert_eq!(
            py_dumps(&json!({"text": "你好", "flag": false})),
            r#"{"text": "你好", "flag": false}"#
        );
        assert_eq!(
            py_dumps(&json!({"task": "verify"})),
            r#"{"task": "verify"}"#
        );
    }

    #[test]
    fn nested_and_mixed_values() {
        assert_eq!(
            py_dumps(&json!(["low", {"tier": 2}, 5])),
            r#"["low", {"tier": 2}, 5]"#
        );
        assert_eq!(py_dumps(&json!({"a": 1})), r#"{"a": 1}"#);
        assert_eq!(py_dumps(&json!([1, true, null])), "[1, true, null]");
    }

    #[test]
    fn state_passthrough() {
        assert_eq!(serialize_state(&json!("raw text")), "raw text");
        assert_eq!(serialize_state(&json!({"n": 3})), r#"{"n": 3}"#);
    }
}
