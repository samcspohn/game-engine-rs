use super::EditorWindow;


pub struct ConsoleWindow {

}
impl ConsoleWindow {
    pub fn new() -> ConsoleWindow {
        ConsoleWindow {}
    }
}
impl EditorWindow for ConsoleWindow {
    fn draw(
        &mut self,
        ui: &mut egui::Ui,
        editor_args: &mut super::EditorArgs,
        inspectable: &mut Option<std::sync::Arc<parking_lot::Mutex<dyn crate::engine::prelude::Inspectable_>>>,
        rec: egui::Rect,
        id: egui::Id,
    ) {
        
    }

    fn get_name(&self) -> &str {
        "Console"
    }
}