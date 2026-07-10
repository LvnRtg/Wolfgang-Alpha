use dioxus::prelude::*;
use std::cell::RefCell;
use web_sys::window;
use wolfgang_alpha::{math, defaults, repl};

mod js_snippets;

const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/style/main.css");
const TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");
const FONT_REGULAR: Asset = asset!("/style/fonts/JetBrainsMono-Regular.woff2");
const FONT_MEDIUM: Asset = asset!("/style/fonts/JetBrainsMono-Medium.woff2");
const FONT_BOLD: Asset = asset!("/style/fonts/JetBrainsMono-Bold.woff2");
const FONT_EXTRABOLD_ITALIC: Asset = asset!("/style/fonts/JetBrainsMono-ExtraBold-Italic.woff2");

/// Clickable starter expressions shown while the console is empty.
const EXAMPLES: [&str; 4] = [
    "sqrt(2) * sin(pi/4)",
    "d/dx (x^3 + 2x)",
    "[1, 2 \\ 3, 4] * [5 \\ 6]",
    "sum_{i=1}^10 i^2",
];

/// Calls `js_snippets::{name}` the next time the DOM updates
macro_rules! call_js_on_dom_update {
    ($name:ident) => {
        spawn(async move {
            let _ = dioxus::document::eval(js_snippets::$name).await;
        });
    };
}


fn main() {
    // Below line would log a bunch of extra things to the console (e.g. calls to listeners).
    // tracing_wasm::set_as_global_default();

    dioxus::launch(App);
}

fn get_element_by_id(id: &str) -> Option<web_sys::Element> {
    window()?.document()?.get_element_by_id(id)
}

fn scroll_to_bottom(id: &str) {
    if let Some(element) = get_element_by_id(id) {
        let height = element.scroll_height();
        element.set_scroll_top(height);
    }
}

/// Given the user input as parameter, returns the output lines of the evaluation.
fn validate_input(input: &str) -> Vec<String> {
    thread_local! {
        static ENV: RefCell<math::Env> = RefCell::new(math::Env {
            constants: defaults::default_constants(),
            functions: defaults::default_functions()
        });
    }
    ENV.with(|c| {
        let mut env = c.borrow_mut();
        repl::eval_line(input, &mut env)
    })
}

/// The `@font-face` rules live here (not in main.css) so the woff2 files in
/// `style/fonts/` are bundled and fingerprinted by the asset pipeline.
fn font_faces() -> String {
    [
        (FONT_REGULAR, 400, "normal"),
        (FONT_MEDIUM, 500, "normal"),
        (FONT_BOLD, 700, "normal"),
        (FONT_EXTRABOLD_ITALIC, 800, "italic"),
    ]
    .map(|(src, weight, style)| {
        format!(
            "@font-face {{ font-family: 'JetBrains Mono'; src: url('{src}') format('woff2'); \
             font-weight: {weight}; font-style: {style}; font-display: swap; }}"
        )
    })
    .join("\n")
}

/// One evaluated input together with the lines it produced.
#[derive(Clone, PartialEq)]
struct HistoryEntry {
    input: String,
    output: Vec<String>,
}

/// An `In[n]:=` / `Out[n]=` cell, Mathematica style. Output lines starting
/// with `[ERROR]` are rendered as an `Err[n]:` block instead.
#[component]
fn Cell(number: usize, entry: HistoryEntry) -> Element {
    let is_error = entry.output.first().is_some_and(|line| line.starts_with("[ERROR]"));
    let lines: Vec<String> = entry
        .output
        .iter()
        .map(|line| line.strip_prefix("[ERROR] ").unwrap_or(line).to_string())
        .collect();
    rsx! {
        div { class: "cell",
            div { class: "row",
                span { class: "label in", "In[{number}]:=" }
                span { class: "cmd", "{entry.input}" }
            }
            if !lines.is_empty() {
                div { class: "row",
                    if is_error {
                        span { class: "label err", "Err[{number}]:" }
                    } else {
                        span { class: "label out", "Out[{number}]=" }
                    }
                    div { class: if is_error { "err-msg" } else { "res" },
                        for (index , line) in lines.into_iter().enumerate() {
                            div { key: "{index}", "{line}" }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn App() -> Element {
    let mut input_value = use_signal(String::new);
    let mut entries = use_signal(Vec::<HistoryEntry>::new);
    // Inputs for arrow-up recall: real commands only (no filler), no consecutive duplicates.
    let mut command_history = use_signal(Vec::<String>::new);
    // Set to 0 every time an input is validated.
    // Pressing the up arrow increases it by 1 (until it hits `command_history.len()`),
    // pressing the down arrow decreases it by 1 (until it hits 0, which clears the input).
    // The recalled command is `command_history[command_history.len() - rollback_index]`.
    let mut rollback_index: Signal<usize> = use_signal(|| 0);
    let mut scroll_to_bottom_signal = use_signal(|| 0); // This allows to perform actions after the DOM is updated
    use_effect(move || {
        if scroll_to_bottom_signal() > 0 {
            scroll_to_bottom("console-history");
        }
    });

    let mut submit = move |input: String| {
        if input.trim().is_empty() {
            return;
        }
        rollback_index.set(0);
        let output = validate_input(&input);
        entries.write().push(HistoryEntry { input: input.clone(), output });
        let is_repeat = command_history.read().last() == Some(&input);
        if !is_repeat {
            command_history.write().push(input);
        }
        input_value.set(String::new());
        scroll_to_bottom_signal += 1;
    };

    let font_css = font_faces();
    let next_number = entries.read().len() + 1;

    rsx! {
        document::Title { "Wolfgang Alpha" }
        document::Link { rel: "icon", href: FAVICON }
        document::Stylesheet { href: MAIN_CSS }
        document::Stylesheet { href: TAILWIND_CSS }
        document::Style { "{font_css}" }
        div { class: "app",
            header { class: "topbar",
                span { class: "brand",
                    "Wolfgang"
                    span { class: "alpha", "α" }
                }
                span { class: "tagline", "symbolic · numeric calculator" }
                span { class: "hints",
                    kbd { "↑" }
                    kbd { "↓" }
                    " history\u{2002}·\u{2002}"
                    kbd { "⌃←" }
                    kbd { "⌃→" }
                    " jump"
                }
            }
            main {
                id: "console-history",
                class: "history",
                // Clicking the console focuses the input, like a real terminal —
                // unless the user is selecting text to copy.
                onclick: move |_| {
                    call_js_on_dom_update!(FOCUS_INPUT_UNLESS_SELECTING);
                },
                if entries.read().is_empty() {
                    div { class: "welcome",
                        p { "Type an expression and press Enter. Try one of these to start:" }
                        div { class: "examples",
                            for example in EXAMPLES {
                                button {
                                    class: "example",
                                    onclick: move |_| {
                                        input_value.set(example.to_string());
                                        call_js_on_dom_update!(MOVE_CURSOR_TO_RIGHT_END);
                                    },
                                    "{example}"
                                }
                            }
                        }
                        p { class: "keys",
                            "↑ ↓ browse history · Ctrl+←/→ jump to line ends · Esc clears the line"
                        }
                    }
                }
                for (index , entry) in entries().into_iter().enumerate() {
                    Cell { key: "{index}", number: index + 1, entry }
                }
            }
            footer { class: "inputbar",
                span { class: "label in", "In[{next_number}]:=" }
                input {
                    r#type: "text",
                    class: "console-input",
                    id: "console-input",
                    autocomplete: "off",
                    spellcheck: "false",
                    autocapitalize: "off",
                    aria_label: "Expression input",
                    value: "{input_value}",
                    oninput: move |event| input_value.set(event.value()), // Update 'input_value' every time the content of the input field is modified
                    onmounted: |_| {
                        call_js_on_dom_update!(FOCUS_MAIN_INPUT);
                    },
                    onkeydown: move |event| {
                        let modifiers = event.modifiers();
                        let ctrl = modifiers.contains(Modifiers::CONTROL);
                        let shift = modifiers.contains(Modifiers::SHIFT);
                        match event.data.key() {
                            Key::Enter => {
                                submit(input_value());
                            }
                            Key::Escape => {
                                input_value.set(String::new());
                                rollback_index.set(0);
                            }
                            Key::ArrowUp => {
                                let rbi = rollback_index();
                                let history = command_history();
                                if rbi < history.len() {
                                    rollback_index.set(rbi + 1);
                                    input_value.set(history[history.len() - 1 - rbi].clone());
                                    call_js_on_dom_update!(MOVE_CURSOR_TO_RIGHT_END);
                                }
                            }
                            Key::ArrowDown => {
                                let rbi = rollback_index();
                                let history = command_history();
                                if rbi > 0 {
                                    rollback_index.set(rbi - 1);
                                    input_value
                                        .set(
                                            if rbi > 1 {
                                                history[history.len() - (rbi - 1)].clone()
                                            } else {
                                                String::new()
                                            },
                                        );
                                    call_js_on_dom_update!(MOVE_CURSOR_TO_RIGHT_END);
                                }
                            }
                            Key::ArrowLeft if ctrl => {
                                if shift {
                                    call_js_on_dom_update!(SELECT_UNTIL_LEFT_END);
                                } else {
                                    call_js_on_dom_update!(MOVE_CURSOR_TO_LEFT_END);
                                }
                            }
                            Key::ArrowRight if ctrl => {
                                if shift {
                                    call_js_on_dom_update!(SELECT_UNTIL_RIGHT_END);
                                } else {
                                    call_js_on_dom_update!(MOVE_CURSOR_TO_RIGHT_END);
                                }
                            }
                            _ => {}
                        }
                    },
                }
            }
        }
    }
}
