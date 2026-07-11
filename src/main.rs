use dioxus::prelude::*;
use std::cell::RefCell;
use web_sys::window;
use wolfgang_alpha::{defaults, math, repl};

mod js_snippets;

const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/assets/main.css");
const MONO_REGULAR: Asset = asset!("/style/fonts/JetBrainsMono-Regular.woff2");
const MONO_BOLD: Asset = asset!("/style/fonts/JetBrainsMono-Bold.woff2");

const EXAMPLES: [Example; 4] = [
    Example {
        label: "Quick arithmetic",
        expression: "sqrt(144) + 2^5",
    },
    Example {
        label: "Differentiate",
        expression: "d/dx (x^3 + 2x + 1)",
    },
    Example {
        label: "Matrix determinant",
        expression: "det([1, 2; 3, 4])",
    },
    Example {
        label: "Finite sum",
        expression: "sum_{i=1}^10 i^2",
    },
];

#[derive(Clone, Copy)]
struct Example {
    label: &'static str,
    expression: &'static str,
}

#[derive(Clone, PartialEq)]
struct Calculation {
    query: String,
    output: Vec<String>,
    is_error: bool,
}

/// Calls `js_snippets::{name}` the next time the DOM updates.
macro_rules! call_js_on_dom_update {
    ($name:ident) => {
        spawn(async move {
            let _ = dioxus::document::eval(js_snippets::$name).await;
        });
    };
}

fn main() {
    dioxus::launch(App);
}

fn get_element_by_id(id: &str) -> Option<web_sys::Element> {
    window()?.document()?.get_element_by_id(id)
}

fn scroll_to_bottom(id: &str) {
    if let Some(element) = get_element_by_id(id) {
        element.set_scroll_top(element.scroll_height());
    }
}

/// Given the user input as parameter, returns the new lines to be added to the console.
fn validate_input(input: &str) -> Vec<String> {
    thread_local! {
        static ENV: RefCell<math::Env> = RefCell::new(math::Env {
            constants: defaults::default_constants(),
            functions: defaults::default_functions()
        });
    }
    ENV.with(|cell| {
        let mut env = cell.borrow_mut();
        repl::eval_line(input, &mut env)
    })
}

fn submit_calculation(
    query: String,
    mut calculations: Signal<Vec<Calculation>>,
    mut command_history: Signal<Vec<String>>,
    mut rollback_index: Signal<usize>,
    mut input_value: Signal<String>,
    mut scroll_signal: Signal<u32>,
) {
    let query = query.trim().to_string();
    if query.is_empty() {
        return;
    }

    let output = validate_input(&query);
    let is_error = output.iter().any(|line| line.starts_with("[ERROR]"));

    calculations.write().push(Calculation {
        query: query.clone(),
        output,
        is_error,
    });
    command_history.write().push(query);
    rollback_index.set(0);
    input_value.set(String::new());
    scroll_signal.set(scroll_signal() + 1);
}

#[component]
fn App() -> Element {
    let mut input_value = use_signal(String::new);
    let mut calculations = use_signal(Vec::<Calculation>::new);
    let mut command_history = use_signal(Vec::<String>::new);
    let mut rollback_index = use_signal(|| 0usize);
    let scroll_signal = use_signal(|| 0u32);

    use_effect(move || {
        if scroll_signal() > 0 {
            scroll_to_bottom("history-scroll");
        }
    });

    let calculation_count = calculations().len();
    let calculation_label = if calculation_count == 1 {
        "calculation"
    } else {
        "calculations"
    };
    let font_faces = format!(
        r#"
        @font-face {{
            font-family: "JetBrains Mono";
            src: url("{MONO_REGULAR}") format("woff2");
            font-weight: 400;
            font-style: normal;
            font-display: swap;
        }}
        @font-face {{
            font-family: "JetBrains Mono";
            src: url("{MONO_BOLD}") format("woff2");
            font-weight: 700;
            font-style: normal;
            font-display: swap;
        }}
        "#
    );

    rsx! {
        document::Title { "Wolfgang Alpha — Symbolic Calculator" }
        document::Link { rel: "icon", href: FAVICON }
        document::Style { {font_faces} }
        document::Stylesheet { href: MAIN_CSS }

        main { class: "app-shell",
            header { class: "topbar",
                div { class: "brand",
                    div { class: "brand-mark", aria_hidden: "true",
                        span { "W" }
                        span { "α" }
                    }
                    div { class: "brand-copy",
                        span { class: "brand-name", "Wolfgang Alpha" }
                        span { class: "brand-tagline", "Symbolic computation, locally" }
                    }
                }
                div { class: "engine-status",
                    span { class: "status-dot", aria_hidden: "true" }
                    span { "Engine ready" }
                }
            }

            div { class: "workspace",
                aside { class: "reference-panel",
                    div { class: "reference-glow", aria_hidden: "true" }
                    div { class: "intro-copy",
                        p { class: "eyebrow", "Math, without the busywork" }
                        h1 { "Think in", br {}, em { "expressions." } }
                        p { class: "intro-text",
                            "Evaluate, simplify, differentiate, and explore — all from one focused workspace."
                        }
                    }

                    section { class: "examples", aria_label: "Example expressions",
                        div { class: "section-heading",
                            span { "Try an example" }
                            span { class: "section-rule" }
                        }
                        div { class: "example-list",
                            for example in EXAMPLES {
                                button {
                                    key: "{example.expression}",
                                    class: "example-button",
                                    r#type: "button",
                                    onclick: move |_| {
                                        input_value.set(example.expression.to_string());
                                        rollback_index.set(0);
                                        call_js_on_dom_update!(MOVE_CURSOR_TO_RIGHT_END);
                                    },
                                    span { class: "example-label", "{example.label}" }
                                    code { "{example.expression}" }
                                    span { class: "example-arrow", aria_hidden: "true", "↗" }
                                }
                            }
                        }
                    }

                    div { class: "reference-footer",
                        div { class: "reference-stat",
                            strong { "π" }
                            span { "Built-in constants" }
                        }
                        div { class: "reference-stat",
                            strong { "f(x)" }
                            span { "Custom definitions" }
                        }
                        div { class: "reference-stat",
                            strong { "[ ]" }
                            span { "Matrix operations" }
                        }
                    }
                }

                section { class: "calculator-panel", aria_label: "Calculator workspace",
                    header { class: "panel-header",
                        div {
                            p { class: "panel-kicker", "Current session" }
                            h2 { "Calculation workspace" }
                        }
                        div { class: "panel-actions",
                            span { class: "calculation-count",
                                "{calculation_count} {calculation_label}"
                            }
                            button {
                                class: "clear-button",
                                r#type: "button",
                                disabled: calculation_count == 0,
                                aria_label: "Clear calculation history",
                                onclick: move |_| {
                                    calculations.set(Vec::new());
                                    command_history.set(Vec::new());
                                    rollback_index.set(0);
                                    input_value.set(String::new());
                                    call_js_on_dom_update!(FOCUS_MAIN_INPUT);
                                },
                                span { aria_hidden: "true", "×" }
                                "Clear"
                            }
                        }
                    }

                    div {
                        id: "history-scroll",
                        class: "history",
                        aria_live: "polite",
                        if calculations().is_empty() {
                            div { class: "empty-state",
                                div { class: "empty-orbit", aria_hidden: "true",
                                    span { class: "orbit-symbol orbit-symbol-one", "∑" }
                                    span { class: "orbit-symbol orbit-symbol-two", "π" }
                                    span { class: "orbit-symbol orbit-symbol-three", "√" }
                                    div { class: "orbit-core", "=" }
                                }
                                h3 { "Ready when you are." }
                                p {
                                    "Enter an expression below or choose an example to begin your session."
                                }
                            }
                        } else {
                            div { class: "calculation-list",
                                for (index, calculation) in calculations().into_iter().enumerate() {
                                    {
                                        let query = calculation.query.clone();
                                        let output = calculation.output.join("\n");
                                        rsx! {
                                            article {
                                                key: "{index}-{calculation.query}",
                                                class: if calculation.is_error { "calculation error" } else { "calculation" },
                                                div { class: "calculation-index", "{index + 1:02}" }
                                                div { class: "calculation-content",
                                                    div { class: "query-row",
                                                        span { class: "query-prompt", aria_hidden: "true", "ƒ" }
                                                        code { "{calculation.query}" }
                                                        button {
                                                            class: "reuse-button",
                                                            r#type: "button",
                                                            aria_label: "Reuse expression {calculation.query}",
                                                            title: "Edit and run again",
                                                            onclick: move |_| {
                                                                input_value.set(query.clone());
                                                                rollback_index.set(0);
                                                                call_js_on_dom_update!(MOVE_CURSOR_TO_RIGHT_END);
                                                            },
                                                            "Reuse"
                                                        }
                                                    }
                                                    div { class: "result-row",
                                                        div { class: "result-meta",
                                                            span { class: "result-dot", aria_hidden: "true" }
                                                            span { if calculation.is_error { "Error" } else { "Result" } }
                                                        }
                                                        pre { "{output}" }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    footer { class: "composer",
                        div { class: "composer-heading",
                            label { r#for: "Display 1 Input", "Enter an expression" }
                            span { "Press " kbd { "Enter" } " to calculate" }
                        }
                        form {
                            class: "input-form",
                            onsubmit: move |event| {
                                event.prevent_default();
                                submit_calculation(
                                    input_value(),
                                    calculations,
                                    command_history,
                                    rollback_index,
                                    input_value,
                                    scroll_signal,
                                );
                            },
                            span { class: "input-prompt", aria_hidden: "true", ">" }
                            input {
                                r#type: "text",
                                id: "Display 1 Input",
                                value: "{input_value}",
                                placeholder: "e.g. d/dx (x^3 + 2x)",
                                autocomplete: "off",
                                autocapitalize: "off",
                                spellcheck: "false",
                                aria_label: "Mathematical expression",
                                oninput: move |event| {
                                    input_value.set(event.value());
                                    rollback_index.set(0);
                                },
                                onmounted: |_| {
                                    call_js_on_dom_update!(FOCUS_MAIN_INPUT);
                                },
                                onkeydown: move |event| {
                                    let modifiers = event.modifiers();
                                    let ctrl = modifiers.contains(Modifiers::CONTROL);
                                    let shift = modifiers.contains(Modifiers::SHIFT);
                                    match event.data.key() {
                                        Key::ArrowUp => {
                                            event.prevent_default();
                                            let commands = command_history();
                                            if !commands.is_empty() {
                                                let next = (rollback_index() + 1).min(commands.len());
                                                rollback_index.set(next);
                                                input_value.set(commands[commands.len() - next].clone());
                                                call_js_on_dom_update!(MOVE_CURSOR_TO_RIGHT_END);
                                            }
                                        }
                                        Key::ArrowDown => {
                                            event.prevent_default();
                                            let commands = command_history();
                                            let current = rollback_index();
                                            if current > 0 {
                                                let next = current - 1;
                                                rollback_index.set(next);
                                                if next == 0 {
                                                    input_value.set(String::new());
                                                } else {
                                                    input_value.set(commands[commands.len() - next].clone());
                                                }
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
                            button {
                                class: "run-button",
                                r#type: "submit",
                                disabled: input_value().trim().is_empty(),
                                span { "Run" }
                                span { class: "run-arrow", aria_hidden: "true", "→" }
                            }
                        }
                        div { class: "composer-footer",
                            span { kbd { "↑" } kbd { "↓" } " command history" }
                            span { class: "footer-divider", aria_hidden: "true" }
                            span { kbd { "Ctrl" } "+" kbd { "←" } kbd { "→" } " jump to edge" }
                        }
                    }
                }
            }
        }
    }
}
