//! Erreurs du moteur : tout ce qui peut echouer hors du calcul (fenetre, carte graphique, fichiers).

use std::path::{Path, PathBuf};

/// Resultat d'une operation du moteur.
pub type Result<T> = std::result::Result<T, Error>;

/// Ce qui peut empecher le moteur de demarrer, de dessiner ou d'ecrire un fichier.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// La boucle d'evenements du systeme n'a pas pu etre creee, ou s'est arretee sur une erreur.
    #[error("boucle d'evenements : {0}")]
    EventLoop(#[from] winit::error::EventLoopError),
    /// Le systeme a refuse d'ouvrir la fenetre.
    #[error("creation de la fenetre impossible : {0}")]
    Window(#[from] winit::error::OsError),
    /// La fenetre n'a pas pu recevoir de surface de rendu.
    #[error("surface de rendu : {0}")]
    Surface(#[from] wgpu::CreateSurfaceError),
    /// Aucun adaptateur (Vulkan, Metal, DX12) ne sait dessiner dans la fenetre.
    #[error("aucune carte graphique compatible")]
    NoAdapter,
    /// L'adaptateur ne propose aucun format d'image pour la surface.
    #[error("la carte graphique ne propose aucun format d'image pour la fenetre")]
    NoSurfaceFormat,
    /// Le peripherique graphique n'a pas pu etre ouvert.
    #[error("peripherique graphique : {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    /// Erreur de rendu sans reprise possible (memoire video epuisee, par exemple).
    #[error("erreur de rendu : {0}")]
    Render(#[from] wgpu::SurfaceError),
    /// L'image rendue n'a pas pu etre relue depuis la carte graphique.
    #[error("lecture de l'image rendue : {0}")]
    Readback(#[from] wgpu::BufferAsyncError),
    /// Lecture ou ecriture d'un fichier.
    #[error("{} : {source}", path.display())]
    Io {
        /// Le fichier en cause.
        path: PathBuf,
        /// L'erreur du systeme.
        source: std::io::Error,
    },
    /// L'image n'a pas pu etre encodee en PNG.
    #[error("encodage PNG : {0}")]
    Png(#[from] png::EncodingError),
    /// L'image a ecrire n'a pas la taille annoncee.
    #[error("image de {width} x {height} : {expected} octets attendus, {actual} recus")]
    ImageSize {
        /// Largeur annoncee, en pixels.
        width: u32,
        /// Hauteur annoncee, en pixels.
        height: u32,
        /// Taille que les donnees devraient avoir.
        expected: usize,
        /// Taille des donnees recues.
        actual: usize,
    },
}

impl Error {
    /// Erreur d'entree-sortie sur `path`.
    pub fn io(path: &Path, source: std::io::Error) -> Self {
        Self::Io { path: path.to_path_buf(), source }
    }
}

/// Ecrit une image 8 bits par canal dans un fichier PNG. `data` contient les lignes de haut en bas,
/// sans bourrage : `width * height * canaux` octets.
pub fn write_png(path: &Path, width: u32, height: u32, color: png::ColorType, data: &[u8]) -> Result<()> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(color.samples()))
        .unwrap_or(usize::MAX);
    if data.len() != expected {
        return Err(Error::ImageSize { width, height, expected, actual: data.len() });
    }
    let file = std::fs::File::create(path).map_err(|e| Error::io(path, e))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(color);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(data)?;
    writer.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_png_rejects_a_buffer_of_the_wrong_size() {
        let path = std::env::temp_dir().join("terra_write_png_wrong_size.png");
        let err = write_png(&path, 4, 4, png::ColorType::Rgba, &[0; 10]).unwrap_err();
        assert!(matches!(err, Error::ImageSize { expected: 64, actual: 10, .. }), "{err}");
        assert!(!path.exists(), "rien ne doit etre ecrit");
    }

    #[test]
    fn write_png_reports_the_path_that_failed() {
        let path = Path::new("/dossier/qui/n/existe/pas/image.png");
        let err = write_png(path, 1, 1, png::ColorType::Rgb, &[0; 3]).unwrap_err();
        assert!(err.to_string().contains("image.png"), "{err}");
    }
}
