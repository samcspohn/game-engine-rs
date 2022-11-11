pub trait Inspectable {
    fn inspect(&mut self, ui: &mut egui::Ui);
}