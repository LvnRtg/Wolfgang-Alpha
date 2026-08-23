//! Store raw JavaScript code to avoid visual bugs due to auto-formatting.

pub const MOVE_CURSOR_TO_RIGHT_END: &str = r#"
setTimeout(() => {
    const input = document.getElementById("Display 1 Input");
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
}, 0);
"#;
pub const MOVE_CURSOR_TO_LEFT_END: &str = r#"
setTimeout(() => {
    const input = document.getElementById("Display 1 Input");
    input.focus();
    input.setSelectionRange(0, 0);
}, 0);
"#;
pub const SELECT_UNTIL_RIGHT_END: &str = r#"
setTimeout(() => {
    const input = document.getElementById("Display 1 Input");
    input.focus();
    input.setSelectionRange(input.selectionStart, input.value.length);
}, 0);
"#;
pub const SELECT_UNTIL_LEFT_END: &str = r#"
setTimeout(() => {
    const input = document.getElementById("Display 1 Input");
    input.focus();
    input.setSelectionRange(0, input.selectionEnd);
}, 0);
"#;
pub const FOCUS_MAIN_INPUT: &str = r#"
document.getElementById("Display 1 Input").focus();
"#;

/// Inserts a symbol sent through the eval channel at the input's current
/// selection, then restores focus and positions the caret after it.
pub const INSERT_SYMBOL_AT_CURSOR: &str = r#"
const symbol = await dioxus.recv();
const input = document.getElementById("Display 1 Input");

if (input) {
    const start = input.selectionStart ?? input.value.length;
    const end = input.selectionEnd ?? start;
    const nextValue = input.value.slice(0, start) + symbol + input.value.slice(end);
    const nextCursor = start + symbol.length;
    const valueSetter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype,
        "value"
    ).set;

    valueSetter.call(input, nextValue);
    input.dispatchEvent(new Event("input", { bubbles: true }));

    await new Promise(requestAnimationFrame);
    input.focus({ preventScroll: true });
    input.setSelectionRange(nextCursor, nextCursor);
}
"#;
