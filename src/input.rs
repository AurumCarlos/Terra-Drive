//! Etat du clavier et de la souris pour une image.

use std::collections::HashSet;
use winit::keyboard::KeyCode;

/// Les touches sont physiques : `KeyW` est la touche Z sur un clavier AZERTY.
#[derive(Default)]
pub struct Input {
    held: HashSet<KeyCode>,
    pressed: HashSet<KeyCode>,
    /// Deplacement de la souris depuis l'image precedente.
    pub mouse_delta: (f64, f64),
    pub scroll: f32,
    /// Vrai quand la souris est capturee par le jeu.
    pub captured: bool,
}

impl Input {
    /// Vrai tant que la touche est enfoncee.
    pub fn held(&self, key: KeyCode) -> bool {
        self.held.contains(&key)
    }

    /// Vrai uniquement a l'image ou la touche est enfoncee.
    pub fn just_pressed(&self, key: KeyCode) -> bool {
        self.pressed.contains(&key)
    }

    pub(crate) fn set_key(&mut self, key: KeyCode, down: bool, repeat: bool) {
        if down {
            if !repeat {
                self.pressed.insert(key);
            }
            self.held.insert(key);
        } else {
            self.held.remove(&key);
        }
    }

    pub(crate) fn release_all(&mut self) {
        self.held.clear();
        self.pressed.clear();
    }

    pub(crate) fn end_frame(&mut self) {
        self.pressed.clear();
        self.mouse_delta = (0.0, 0.0);
        self.scroll = 0.0;
    }
}
