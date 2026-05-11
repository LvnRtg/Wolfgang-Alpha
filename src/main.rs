use dioxus::prelude::*;
use dioxus_logger::tracing;
//use dioxus_logger::tracing;
use std::vec::Vec;
use std::collections::HashMap;
use std::cell::RefCell;
use web_sys::window;

mod math;
mod parser;
pub use crate::parser::*;
mod defaults;

const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/style/main.css");
const TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");

fn main() {
    // Below line would log a bunch of extra things to the console (e.g. calls to listeners).
    // tracing_wasm::set_as_global_default();

    dioxus::launch(App);
}

#[component]
fn Display(content: Vec<String>) -> Element {
    rsx! { "Content: {content[0]}" }
}

// fn scroll_to_top(id: &str) {
//     let document = window().unwrap().document().unwrap();
//     if let Some(element) = document.get_element_by_id(id) {
//         element.set_scroll_top(0);
//     }
// }
fn scroll_to_bottom(id: &str) {
    let document = window().unwrap().document().unwrap();
    if let Some(element) = document.get_element_by_id(id) {
        let height = element.scroll_height();
        element.set_scroll_top(height);
    }
    // Below code would create a smooth scroll effect, but in our case, we do not want this (it looks bad).
    // let code = format!("const el = document.getElementById(\"{}\"); if (el) {{ el.scrollTo({{ top: el.scrollHeight, behavior: \"smooth\" }}); }}", id);
    // document::eval(&code);
}

/// Given the user input as parameter, returns the new lines to be added to the console.
fn validate_input(input: String) -> Vec<String> {
    // This creates a mutable variable that can only be accessed and modified from inside this function (making it safe)
    // while retaining its value between function calls.
    thread_local! {
        static CONSTANTS: RefCell<HashMap<String, math::Object>> = RefCell::new(defaults::default_constants());
        static FUNCTIONS: RefCell<HashMap<String, math::FunctionRepr>> = RefCell::new(defaults::default_functions());
    }

    let mut parser = Parser::new(&input);
    //tracing::info!("{:?}", parser.tokens);
    let mut output = Vec::<String>::new();
    CONSTANTS.with(|c: &RefCell<HashMap<String, math::Object>>| {
        let mut constants = c.borrow_mut();
        FUNCTIONS.with(|f| {
            let mut functions = f.borrow_mut();
            let ast = parser.parse(&mut constants, &mut functions);
            let eval = eval(&ast, &HashMap::<&String, &math::Object>::new(), &mut constants, &mut functions);
            match eval {
                Ok(obj) => {
                    output = obj.to_multline();
                }
                Err(e) => {
                    output.push(format!("[ERROR] {}", e));
                }
            }
            tracing::info!("{:?}", constants);
            tracing::info!("{:?}", functions);
        });
    });
    
    output
}

#[component]
fn App() -> Element {
    let mut input_value = use_signal(String::new);
    let mut console_lines = use_signal(Vec::<String>::new);
    let joined_lines = console_lines().join("\n");
    let mut previous_commands = use_signal(Vec::<String>::new);
    let joined_previous_commands = previous_commands().join("\n");
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
                class: "halfwidth",
                class: "fullheight",
                div { class: "display_top_part",
                    div { class: "previous_lines", "{joined_lines}" }
                    div { class: "previous_commands", "{joined_previous_commands}" }
                }
                div { class: "display_bottom_part",
                    ">"
                    input {
                        r#type: "search",
                        class: "inline_input",
                        value: "{input_value}",
                        oninput: move |event| input_value.set(event.value()), // Update 'input_value' every time the content of the input field is modified
                        onkeydown: move |event| {
                            if event.data.key() == Key::Enter {
                                let input = input_value();
                                let mut cl = console_lines();
                                let mut pc = previous_commands();
                                let mut new_lines = validate_input(input.clone());
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
                        },
                    }
                }
            }
        }
    }
}
