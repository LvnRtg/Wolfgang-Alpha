//! Store raw JavaScript code to avoid visual bugs due to auto-formatting.

pub const MOVE_CURSOR_TO_RIGHT_END: &str = r#"
setTimeout(() => {
    const input = document.getElementById("console-input");
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
}, 0);
"#;
pub const MOVE_CURSOR_TO_LEFT_END: &str = r#"
setTimeout(() => {
    const input = document.getElementById("console-input");
    input.focus();
    input.setSelectionRange(0, 0);
}, 0);
"#;
pub const SELECT_UNTIL_RIGHT_END: &str = r#"
setTimeout(() => {
    const input = document.getElementById("console-input");
    input.focus();
    input.setSelectionRange(input.selectionStart, input.value.length);
}, 0);
"#;
pub const SELECT_UNTIL_LEFT_END: &str = r#"
setTimeout(() => {
    const input = document.getElementById("console-input");
    input.focus();
    input.setSelectionRange(0, input.selectionEnd);
}, 0);
"#;
pub const FOCUS_MAIN_INPUT: &str = r#"
document.getElementById("console-input").focus();
"#;
/// Focuses the input unless the user has text selected (i.e. is copying output).
pub const FOCUS_INPUT_UNLESS_SELECTING: &str = r#"
if (window.getSelection().toString() === "") {
    document.getElementById("console-input").focus();
}
"#;
