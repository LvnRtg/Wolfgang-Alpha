use dioxus::prelude::*;
use std::vec::Vec;
use std::cell::RefCell;
use web_sys::window;

mod math;
mod lang;
mod defaults;
mod js_snippets;

const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/style/main.css");
const TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");


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

#[component]
fn Display(content: Vec<String>) -> Element {
    rsx! { "Content: {content[0]}" }
}

fn get_element_by_id(id: &str) -> Option<web_sys::Element> {
    window()?.document()?.get_element_by_id(id)
}

// fn scroll_to_top(id: &str) {
//     let document = window().unwrap().document().unwrap();
//     if let Some(element) = document.get_element_by_id(id) {
//         element.set_scroll_top(0);
//     }
// }
fn scroll_to_bottom(id: &str) {
    if let Some(element) = get_element_by_id(id) {
        let height = element.scroll_height();
        element.set_scroll_top(height);
    }
}

/// Given the user input as parameter, returns the new lines to be added to the console.
fn validate_input(input: &str) -> Vec<String> {
    // This creates a mutable variable that can only be accessed and modified from inside this function (making it safe)
    // while retaining its value between function calls.
    thread_local! {
        static ENV: RefCell<math::Env> = RefCell::new(math::Env {
            constants: defaults::default_constants(),
            functions: defaults::default_functions()
        });
    }


    let tokens = match lang::tokenize(input) {
        Ok(x) => x,
        Err(e) => {return vec![format!("[ERROR] {e}")];}
    };
    let mut parser = lang::Parser::from(tokens);
    let mut output = Vec::<String>::new();
    ENV.with(|c: &RefCell<math::Env>| {
        let mut env = c.borrow_mut();
        match parser.parse(&mut env) {
            Ok(expressions) => {
                // tracing::info!("{}", expressions.iter().map(|x| format!("{}", x)).collect::<Vec<_>>().join("; "));
                for expr in expressions {
                    let eval = lang::eval(&expr, &lang::evaluator::VarStack::Empty, &mut env);
                    match eval {
                        Ok(obj) => {
                            output = obj.to_multline();
                        }
                        Err(e) => {
                            output.push(format!("[ERROR] {}", e));
                        }
                    }
                }
            },
            Err(e) => {output.push(format!("[ERROR] {}", e));}
        };
    });
    output
}

#[component]
fn App() -> Element {
    let mut input_value = use_signal(String::new);
    let mut console_lines = use_signal(Vec::<String>::new);
    let mut previous_commands = use_signal(Vec::<String>::new);
    // Set to 0 every time an input is validated.
    // Pressing the up arrow increases it by 1 (until it hits `previous_commands.len()`),
    // pressing the down arrow decreases it by 1 (until it hits 1).
    // The corresponding command is `previous_commands[previous_commands.len() - rollback_index]`.
    let mut rollback_index: Signal<usize> = use_signal(|| 1);
    let mut scroll_to_bottom_signal = use_signal(|| 0); // This allows to perform actions after the DOM is updated
    use_effect(move || {
        if scroll_to_bottom_signal() > 0 {
            scroll_to_bottom("Display 1");
        }
    });

    rsx! {
        document::Title { "Wolfgang Alpha" }
        document::Link { rel: "icon", href: FAVICON }
        document::Stylesheet { href: MAIN_CSS }
        document::Stylesheet { href: TAILWIND_CSS }
        body {
            div {
                id: "Display 1",
                class: "display",
                class: "fullwidth",
                class: "fullheight",
                div { class: "display_top_part",
                    div { class: "previous_lines",
                        for (index , line) in console_lines().into_iter().enumerate() {
                            div { key: "{index}", "{line}" }
                        }
                    }
                    div { class: "previous_commands",
                        for (index , command) in previous_commands().into_iter().enumerate() {
                            div { key: "{index}", "{command}" }
                        }
                    }
                }
                div { class: "display_bottom_part",
                    ">"
                    input {
                        r#type: "search",
                        class: "inline_input",
                        id: "Display 1 Input",
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
                                    rollback_index.set(0); // Reset rollback index
                                    let input = input_value();
                                    let mut cl = console_lines();
                                    let mut pc = previous_commands();
                                    let mut new_lines = validate_input(&input);
                                    // If we newly add N lines to the LHS, we want to use the first line of the RHS to show the command
                                    // and fill the remaining N-1 lines with some filler (currently "│"; the box drawing char, not the pipe char).
                                    pc.push(input);
                                    let n = new_lines.len();
                                    if n > 1 {
                                        pc.extend(std::iter::repeat_n("│".to_string(), n - 1));
                                    }
                                    cl.append(&mut new_lines);
                                    console_lines.set(cl);
                                    previous_commands.set(pc);
                                    input_value.set(String::new());
                                    scroll_to_bottom_signal += 1;
                                }
                                Key::ArrowUp => {
                                    let rbi = rollback_index();
                                    let pc = previous_commands();
                                    if rbi < pc.len() {
                                        rollback_index.set(rbi + 1);
                                    }
                                    if rbi < pc.len() && !pc.is_empty() {
                                        input_value.set(pc[pc.len() - (rbi + 1)].clone());
                                        call_js_on_dom_update!(MOVE_CURSOR_TO_RIGHT_END);
                                    }
                                }
                                Key::ArrowDown => {
                                    let rbi = rollback_index();
                                    let pc = previous_commands();
                                    if rbi > 0 {
                                        rollback_index.set(rbi - 1);
                                    }
                                    input_value
                                        .set(
                                            if rbi > 1 && !pc.is_empty() {
                                                pc[pc.len() - (rbi - 1)].clone()
                                            } else {
                                                String::new()
                                            },
                                        );
                                    call_js_on_dom_update!(MOVE_CURSOR_TO_RIGHT_END);
                                }
                                Key::ArrowLeft if ctrl => {
                                    if shift {
                                        call_js_on_dom_update!(SELECT_UNTIL_LEFT_END);
                                    }
                                    else {
                                        call_js_on_dom_update!(MOVE_CURSOR_TO_LEFT_END);
                                    }
                                }
                                Key::ArrowRight if ctrl => {
                                    if shift {
                                        call_js_on_dom_update!(SELECT_UNTIL_RIGHT_END);
                                    }
                                    else {
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
}
