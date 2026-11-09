Oui, le jeu devrait tourner sur Arch Linux, mais je n'ai pas pu le tester : je n'ai que Windows ici. J'ai relu le code, et rien n'y bloque Linux.

- **Le seul code propre à Windows** est la priorité basse des threads de génération, dans `src/world/generator.rs:339`. Il est marqué `#[cfg(windows)]`, donc il est simplement ignoré sous Linux. Le jeu tournera, mais sans cette priorité, ce qui peut donner un peu plus d'à-coups au démarrage.
- **Aucun chemin Windows en dur** dans le code (`C:\`, `APPDATA`, `.exe`).
- **Les bibliothèques fonctionnent sous Linux :** wgpu passe par Vulkan, winit gère X11 et Wayland, cpal passe par ALSA.
- **`cargo.ps1` sert seulement sous Windows.** Sous Linux, on utilise `cargo` directement.

Le projet est maintenant dans `C:\Users\charl\Desktop\travail\github\terradrive`, pas directement dans `github`. Lance donc les commandes git d'avant depuis `terradrive`.

**Sur Arch :**

1. Installe Rust, les outils de compilation, ALSA (pour le son), Vulkan et les bibliothèques de fenêtre. `vulkan-radeon` correspond à ta RX 9070 XT ; avec une carte Nvidia ou Intel, il faut un autre pilote.
```bash
sudo pacman -S --needed rustup base-devel pkgconf alsa-lib vulkan-icd-loader vulkan-radeon libxkbcommon wayland libx11 libxcursor libxrandr libxi
```

2. Choisis la version stable de Rust :
```bash
rustup default stable
```

3. Clone le dépôt une fois qu'il est sur GitHub :
```bash
git clone git@github.com:AurumCarlos/Terra-Drive.git
```

4. Lance le jeu depuis le dossier cloné :
```bash
cd Terra-Drive && cargo run --release
```

**Si quelque chose bloque :**
- **Pas de son :** sous PipeWire, installe aussi `pipewire-alsa` pour que l'ALSA de cpal soit redirigé vers PipeWire.
- **La fenêtre ne s'ouvre pas sous Wayland :** essaie de forcer X11.
```bash
WINIT_UNIX_BACKEND=x11 cargo run --release
```
- **Pour vérifier que Vulkan voit ta carte** (installe d'abord `vulkan-tools`) :
```bash
vulkaninfo --summary
```

Si tu as une erreur au lancement sous Arch, envoie-la-moi et je la corrige.
