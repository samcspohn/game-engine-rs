use components::Component;

mod components;

struct CompTest {

}

impl components::Component for CompTest {
    fn update(&mut self) {
        println!("update from test")
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self as &dyn std::any::Any
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self as &mut dyn std::any::Any
    }

}

fn main() {
    let mut ct = CompTest {};
    ct.update();
}