# Terra

Moteur 3D en Rust qui genere des planetes entieres de la taille de la Terre, avec une demo : marcher,
voler, monter jusque dans l'espace, redescendre sans chargement, et creer une nouvelle planete depuis un menu.

## Lancer

```powershell
.\cargo.ps1 run --release
```

`cargo.ps1` ajoute Rust et MinGW au PATH et compile dans `%LOCALAPPDATA%\terra-target`.
Il attend Rust (toolchain `stable-x86_64-pc-windows-gnu`) et WinLibs MinGW, installe via winget.

## Commandes

Les touches sont physiques : ce sont bien ZQSD sur un clavier AZERTY.

| Touche | Action |
| --- | --- |
| Clic gauche | Capturer la souris |
| Echap | Ouvrir / fermer le menu |
| Z Q S D | Se deplacer |
| F | Basculer marche / vol |
| Molette, Page haut / bas | Regler la vitesse de vol, de 2 m/s a 30 000 km/s |
| Maj | Courir (vitesse x8 en vol) |
| Espace | Sauter, ou monter en vol |
| Ctrl | Descendre en vol |
| T / G | Avancer / reculer l'heure |
| F12 | Capture d'ecran PNG |
| M | Couper / remettre le son |
| E | Prendre une voiture (la sienne si elle est a moins de 12 m, sinon une neuve garee dans la rue ou sur l'autoroute la plus proche) ; en descendre a l'arret |

Au volant :

| Touche | Action |
| --- | --- |
| Z / S | Accelerer / freiner, puis reculer |
| Q / D | Braquer |
| Espace | Frein a main (fait deraper) |
| C | Vue de poursuite ou vue du conducteur |
| L | Phares |
| R | Remettre la voiture sur ses roues (a l'arret) |
| Souris | Regarder autour |

Un compteur en bas a droite donne la vitesse, le regime et le rapport.

Pour aller dans l'espace : F, quelques crans de molette, regarder vers le ciel et avancer.
Le bouton "Aller dans l'espace" du menu y va directement.

## Menu

- **Nouvelle planete aleatoire**, ou **Generer avec cette graine** : la graine peut etre un nombre ou un texte.
- Reglages de la prochaine planete : part d'ocean, hauteur du relief, temperature, **taille** (de 0.25 a 2 fois la Terre),
  **nombre de villes** (de 0 a 400) et **rivieres** (a decocher pour une planete sans rivieres).
- **Distance de rendu**, de 5 a 1000 km (80 par defaut), appliquee tout de suite et conservee d'une planete a l'autre.
  Au-dela, le relief perd vite ses details. Elle regle aussi la portee de la vegetation : jusqu'a 6 km pour les arbres,
  320 m pour l'herbe. Au demarrage : `--render-distance 200`.
- **Aller dans l'espace** et **Retourner a la surface**.
- **Aller a la ville la plus proche** : arrivee en vol, a 2 km au sud du centre. L'affichage tete haute donne le nom, le plan et la direction de la ville la plus proche.
- **Carte** de la planete : relief, rivieres et lacs, frontieres de plaques en option. Un clic y envoie le joueur en survol.

La simulation d'une nouvelle planete prend quelques secondes en arriere-plan : le jeu continue sur l'ancienne en attendant.

Depuis l'espace, la camera reste en place et la nouvelle planete apparait dessous.
Pres du sol, elle est deplacee sur un point de depart de la nouvelle planete.

## Options

`terra-demo --help` liste tout. Les valeurs avec une virgule se mettent entre guillemets sous PowerShell. Exemples :

```powershell
# La planete 3, vue de l'espace a midi
.\cargo.ps1 run --release -- --seed 3 --altitude 13000000 --pitch -89 --time 0.5

# Au-dessus d'un point precis, au lever du soleil
.\cargo.ps1 run --release -- --pos "48.85,2.35" --altitude 2000 --time 0.28

# Capture automatique une fois le monde charge
.\cargo.ps1 run --release -- --seed 7 --screenshot vue.png

# Une petite planete, 60 villes, sans rivieres
.\cargo.ps1 run --release -- --planet-size 0.5 --cities 60 --no-rivers

# Au volant, sur l'autoroute qui sort de Pallac (planete 3) ; --cockpit pour la vue du conducteur
.\cargo.ps1 run --release -- --seed 3 --drive --pos "-48.1126,48.5451" --yaw 93

# Sur la grille de depart du circuit le plus proche du point d'arrivee (Sochi Autodrom pres de Venolport, planete 3)
.\cargo.ps1 run --release -- --seed 3 --circuit

# Au pied de la route de montagne la plus proche : ici celle de Toranbourg, 13 epingles (planete 3)
.\cargo.ps1 run --release -- --seed 3 --pos "-53.9113,81.1318" --montagne

# Descente de l'orbite jusqu'au sol, affiche les temps d'image
.\cargo.ps1 run --release -- --no-vsync --altitude 13000000 --pitch -89 --autopilot 440000
```

## Ecrire un jeu

Implementer `terra::Game`, puis appeler `terra::run` :

```rust
use terra::{Camera, Context, EngineConfig, Game, PlanetParams, PlayerController, PLANET_RADIUS};

struct MonJeu { joueur: PlayerController }

impl Game for MonJeu {
    fn start(&mut self, ctx: &mut Context) {
        let dir = ctx.world.find_spawn();
        let rayon = PLANET_RADIUS + ctx.world.height(dir) as f64 + 2.0;
        *ctx.camera = Camera::new(dir * rayon, 0.0, 0.0);
    }
    fn update(&mut self, ctx: &mut Context, dt: f32) {
        self.joueur.update(ctx.input, ctx.camera, ctx.streamer, dt);
    }
}

fn main() {
    let planet = PlanetParams { seed: 42, ..Default::default() };
    let jeu = MonJeu { joueur: PlayerController::default() };
    terra::run(EngineConfig { planet, ..Default::default() }, jeu).unwrap();
}
```

`Context` donne acces aux entrees, a la camera, au ciel, a la fonction de planete
(hauteur, climat, biome pour une direction), aux objets charges et a la regeneration (`ctx.regenerate`).

## Architecture

| Fichier | Role |
| --- | --- |
| `world/noise.rs` | Bruit de gradient 3D a derivees analytiques, en f64 |
| `world/cube.rs` | Grilles globales sur le cube-sphere, champs interpoles sans raccord |
| `world/tectonics.rs` | Simulation des plaques : derive, collisions, subduction, dorsales |
| `world/landforms.rs` | Filtre d'erosion, provinces de paysage, volcans, vents et humidite |
| `world/hydrology.rs` | Erosion, drainage, lacs et reseau de rivieres |
| `world/generator.rs` | Fonction (parametres, direction) vers relief, eau, climat, biome, couleur |
| `world/chunk.rs` | Quadtree sur les 6 faces d'un cube-sphere, maillages (morphing, jupes), surface de l'eau |
| `world/city.rs` | Villes : emplacement sur la planete, plan des rues (damier, ilots, etoile, vieille ville), ilots, parcelles, batiments, collisions |
| `world/highway.rs` | Autoroutes entre les villes : graphe de voisinage, trace A* qui evite la mer et les autres autoroutes, ponts, viaducs et passages superieurs, profil en long, remodelage du sol |
| `world/mountain.rs` | Routes de montagne : sommet vise, trace en lacets par A* a pente limitee (32 caps, epingles), lissage, profil en long, remodelage du sol, index spatial |
| `world/circuit.rs`, `world/circuit_data.rs` | Circuits : 40 traces reels de F1, place pres de chaque ville sur un terrain plat et sec, route d'acces, profil en long, degagements, index spatial |
| `world/city_mesh.rs` | Maillages des batiments, des rues, des ponts et du mobilier urbain, par niveau de detail |
| `world/scatter.rs` | Placement deterministe de la vegetation et des rochers |
| `world/streaming.rs` | Selection du LOD, horizon, threads de generation, cache, collisions |
| `render/mod.rs` | Rendu wgpu : ombres, scene HDR, ocean, nuages, sortie, interface |
| `render/meshes.rs` | Arbres, herbes et rochers generes par code |
| `render/noise_tex.rs` | Texture de bruit 3D periodique, pour les nuages, le terrain et l'eau |
| `render/shaders/` | Shaders WGSL, dont `city.wgsl` : facades, fenetres, toits, chaussee, eclairage de nuit, et la voiture dessinee au pixel (joints, poignees, optiques, feux, calandre, plaques, pneus, compteurs) |
| `vehicle/model.rs` | Voiture generee par code (plus de 20 000 sommets) : carrosserie lissee sur 161 sections, passages de roues, pneus, jantes a 10 branches, disques et etriers, retroviseurs, essuie-glaces, echappements, habitacle (sieges, planche de bord, compteurs, ecran, volant) |
| `vehicle/dynamics.rs` | Dynamique du vehicule : caisse rigide en 3D, suspensions (ressorts, amortisseurs, butees, barres antiroulis), pneus Pacejka avec relaxation, moteur, boite automatique 7 rapports, differentiel, ABS, antipatinage, aerodynamique, contacts de la caisse avec le sol |
| `vehicle/mod.rs` | Conduite : commandes, plan du sol sous chaque roue (relief, tabliers d'autoroute et ponts de ville, eau, revetement), chocs contre batiments et arbres, bosses de carrosserie, remise sur roues, cameras |
| `audio.rs` | Son synthetise en direct (cpal) : moteur, pneus, roulement, vent, chocs ; vent, feuillages, oiseaux, grillons, vagues, riviere, ville |
| `world/fast_hash.rs` | Table de hachage rapide (FxHash) pour les identifiants de noeuds et de cellules |
| `ui.rs` | Menu et affichage tete haute, avec egui |
| `app.rs`, `camera.rs`, `input.rs`, `player.rs`, `sky.rs` | Boucle de jeu, API et controleur |
| `bin/audit.rs`, `bin/probe.rs`, `tour.ps1` | Controle de la beaute : statistiques, sonde d'un lieu (`probe GRAINE LAT LON`), planches de captures (compiler avant `tour.ps1`) |

Choix principaux :

- **Une simulation, puis une fonction pure.** Rayon de 6 371 km. La creation calcule une fois les grilles globales ; ensuite, tout se deduit de ces grilles et du bruit, et une meme direction donne toujours le meme resultat.
- **Tectonique.** De 10 a 19 plaques rigides tournent autour de leur pole d'Euler pendant 48 pas. La croute continentale se heurte et s'epaissit, l'ocean plonge sous les continents, les dorsales creent de la croute neuve. Les chaines de montagnes viennent de ce soulevement.
- **Erosion.** Un modele de puissance fluviale creuse les bassins versants a 39 puis 20 km de resolution.
- **Lacs.** Le remplissage des depressions sur une grille de 13 km donne leur niveau et leur exutoire. Des cuvettes s'ajoutent dans le froid et au pied des chaines. Chaque lac deborde dans une riviere.
- **Rivieres.** Le cumul des pluies designe les rivieres, qui naissent plus tot en altitude et descendent jusqu'a la mer ou un lac. Le debit fixe leur largeur. Leur trace serpente autour de la grille et un cone de pente taille leur vallee.
- **Sans couture.** Le bruit est evalue en 3D sur la sphere et les grilles sont raccordees le long des aretes du cube.
- **Erosion sans simulation.** Le filtre de ravines de Rune Skovbo Johansen, adapte a la sphere : des rayures alignees sur la pente, en plusieurs octaves qui se ramifient, dessinent cretes et vallons en chaque point, sans voisins ni passes.
- **Provinces de paysage.** Chaque region recoit une signature selon son climat : collines, plateaux et canyons a strates, terres ravinees, dunes, tours calcaires, landes a lacs, monolithes, vallees glaciaires et fjords. S'y ajoutent environ 150 volcans, sur les subductions et en chapelets d'iles.
- **Climat par les vents.** L'humidite part des oceans, suit les vents dominants et tombe sur les reliefs : les deserts se forment derriere les chaines.
- **Relief a toutes les echelles.** Le spectre du relief suit a peu pres celui de la Terre (amplitude proportionnelle a la longueur d'onde) : grandes ondulations de 9 km, collines de 2,4 km, mottes de 420 m, detail de 95 m. Les ondulations et les collines utilisent un bruit amorti par sa propre pente (Inigo Quilez) : fonds de vallee doux, cretes nettes, comme apres l'erosion. Un bruit regional de 70 km fait alterner de rares vraies plaines et des pays bosseles.
- **Montagnes alpines.** Sur les grandes cretes de 26 km s'ajoutent des aretes et des pics secondaires de 6,5 km, plus marques en altitude (multifractale hybride de Musgrave).
- **Forets lisibles.** De grandes clairieres, des massifs de feuillus ou de coniferes, des taches claires et sombres dans la canopee, et des cretes degarnies qui dessinent le relief sous les arbres. Les maillages lointains et les arbres suivent les memes regles.
- **Plaines vivantes.** Une houle de fond sur toutes les terres, des buttes isolees comme reperes, de grands pans d'herbe d'or et d'herbe verte, et des monolithes a parois de roche nue.
- **Forets jusqu'a l'horizon.** Au-dela de la portee des arbres, le sol prend la couleur de la canopee, selon les memes regles que le placement des arbres.
- **LOD CDLOD.** 18 niveaux, de 1,5 m a plus de 200 km entre deux sommets. Les noeuds caches par l'horizon ne sont ni generes ni dessines.
- **Coordonnees f64.** La camera est en f64 et le GPU travaille relativement a elle, donc pas de tremblement a la surface.
- **Vol.** Le regard suit la courbure pres du sol, puis garde une trajectoire droite au-dela de 300 km d'altitude.
- **Generation en arriere-plan.** Le parent reste affiche tant que ses enfants visibles ne sont pas prets, donc pas de trous.
- **Rendu.** Atmosphere a diffusion simple calculee par pixel, valable du sol a l'orbite, soleil selon la longitude, ombres en 3 cascades pres du sol, HDR, halo lumineux, rayons du soleil, FXAA. Le tone mapping melange ACES et une courbe sur la luminance, pour que les couchers de soleil restent orange.
- **Nuages en volume.** Marche de rayons a demi-resolution dans un bruit 3D calcule au demarrage, avec voiles d'altitude. Bases plates, liseres d'argent face au soleil, ombres portees sur le relief et l'eau.
- **Eau.** Vagues sur quatre echelles, reflets de l'image (montagnes et nuages) puis du ciel calcule, chemin de lumiere du soleil, ecume le long des rives.
- **Terrain.** Couleur et relief ajoutes au pixel, du caillou au massif : le lointain garde sa texture, et les forets leurs cimes.
- **Vegetation.** Herbes courbes, hautes herbes a epis qui s'embrasent a contre-jour, sapins dentueles, feuillages ajoures au pixel.
- **Villes.** Environ 140 villes sur la planete 3, posees sur des terres seches et peu pentues, souvent au bord de l'eau. Quatre plans inspires de villes reelles : damier a avenues larges (Manhattan, plan de 1811), ilots a pans coupes (l'Eixample de Cerda a Barcelone), etoile d'avenues rayonnantes avec un quartier d'affaires au bout de l'axe (Paris et la Defense), vieille ville aux rues organiques (croissance par segments, d'apres Parish et Muller 2001). Les ilots sont decoupes en parcelles, a la maniere de Vanegas et al. 2012. Le bati s'effiloche vers l'exterieur : lobes, echancrures, hameaux et maisons le long des routes de sortie.
- **Autoroutes.** Les villes voisines sont reliees par un graphe de voisinage relatif (Toussaint 1980), complete pour rejoindre les morceaux isoles. Le trace contourne relief, lacs et villes, passe la mer peu profonde sur de longs ponts, et s'ecarte des autres autoroutes ; les croisements deviennent des passages superieurs. Profil en long lisse, deblais et remblais, viaducs sur piles, glissieres en beton, portiques, 2 ou 3 voies par sens selon la taille des villes.
- **Carrefours.** Une rue qui debouche en biais s'arrete le long du bord de la chaussee croisee, sans triangle de terre entre les deux ; un test verifie qu'il y a de l'asphalte de part et d'autre de chaque bout de rue.
- **Rues reliees.** Le plan de chaque ville est verifie : toutes ses rues forment un seul reseau (les contacts sont calcules geometriquement), et chaque bout d'autoroute arrive sur une rue. Une rue isolee est rattachee par le plus court chemin parmi les rues ecartees, ou retiree avec son ilot. `planet_stats` controle tous les plans d'une planete.
- **Physique de la voiture.** Dans l'esprit des simulateurs comme BeamNG, rien n'est anime : tout sort du calcul. La caisse est un corps rigide en 3D (1 550 kg, inertie en tangage, lacet et roulis) porte par quatre suspensions independantes : ressort d'environ 1,5 Hz, amortisseur plus ferme en detente qu'en compression, butee de fin de course, barres antiroulis. La charge de chaque pneu est la force de sa suspension, donc la plongee au freinage, le cabrage, le roulis en virage, les roues qui decollent, les sauts et les retombees viennent d'eux-memes. Les pneus suivent la formule magique de Pacejka sur le glissement combine, avec une longueur de relaxation : le pneu se deforme avant de glisser, ce qui tient aussi la voiture en pente a l'arret. Le sol est un plan sonde sous chaque roue (relief, rues, ponts, autoroutes, circuits, eau), avec son adherence et sa resistance au roulement. Si la caisse touche le sol (tonneau, gros saut), sa tole frotte et rebondit ; R remet la voiture sur ses roues. Les chocs contre les batiments et les arbres sont des impulsions au point d'impact (un choc sur un coin fait pivoter la voiture) et laissent des bosses dans la carrosserie. Moteur de 470 Nm, boite automatique a 7 rapports, differentiel a glissement limite, ABS, antipatinage, frein a main, trainee et appui. Integration a 240 Hz. Mesures des tests : 0 a 100 km/h en 4,9 s, freinage de 100 a 0 en 35 m au plus, environ 1,2 g en virage (pneus tres adherents), une oscillation puis le repos apres une chute.
- **Circuits de F1.** Pres de chaque ville qui a la place, un circuit au trace reel de Formule 1 (Monza, Spa, Suzuka, Silverstone, Interlagos, Bakou...), a l'echelle exacte, pose sur un terrain sec et peu pentu, loin des autoroutes. Une route d'acces part d'une avenue de la ville. Piste a lignes blanches et trace de gomme, vibreurs rouges et blancs dans les virages, bacs a gravier a l'exterieur des courbes rapides, degagements en herbe, glissieres, ligne d'arrivee en damier, grille de depart, tribune couverte et muret des stands. Pentes ecretees a 12 %. Traces tires de [bacinger/f1-circuits](https://github.com/bacinger/f1-circuits) (licence MIT).
- **Routes de montagne.** Autour de chaque ville entouree de reliefs partent jusqu'a quatre routes (pres de 70 sur la planete 3), chacune vers un sommet dans une direction differente, au moins 300 m plus haut (route du mont, du col, des cretes, du pic), et finit en raquette autour d'un ilot d'herbe. Le trace est cherche par A* sur une grille de 40 m dont chaque etat est une case et un cap parmi 32, dans l'esprit de Galin et al. 2010 : la route ne tourne que d'un cran par pas et ne monte jamais plus de 10 % ; la ou le versant est trop raide, le seul moyen de monter est l'epingle, un demi-tour decale d'une ou deux cases, suivi d'au moins 120 m de ligne droite. Les lacets s'empilent donc d'eux-memes sur les pentes fortes, et la route reste droite ailleurs. Une route qui se recroise est recherchee en evitant l'endroit. Profil en long ecrete a 12 %, deblais et remblais, deux voies etroites a axe en tirets, glissieres d'acier ou parapets de pierre du cote du vide.
- **Voiture.** Berline modelisee par code et dessinee au pixel, posee a chaque image : roues qui tournent et braquent, volant, suspension, feux stop, de recul, clignotants et phares.
- **Forets claires.** Couvert de 50 a 80 % au coeur des massifs, bois clairs sur les lisieres, trouees de quelques dizaines de metres, grandes clairieres sur un tiers du pays. L'age du peuplement fait varier la taille des arbres ; les arbres isoles sont plus grands.
- **Villes differentes.** Chaque ville a son caractere : basse et etalee ou haute et serree, et une palette dominante (enduits blancs, ocres, brique, pastels ; tuiles rouges ou ardoise). Maisons en L, pavillons et villas, petits immeubles parmi les maisons ; tours sur socle, en gradins ou jumelles ; une ligne de toits irreguliere.
- **Batiments.** Maisons avec garage, piscine et haie ; maisons de ville a volets ; immeubles haussmanniens a balcons et toits mansardes ; barres et residences ; bureaux ; quatre familles de gratte-ciel (art deco, verre, octogone, fuseau) ; entrepots, citernes, cheminees, eglises, places et fontaines. Fenetres, encadrements, vitrines et enseignes sont dessines au pixel ; la nuit, une partie des fenetres s'allument.
- **Rues.** Chaussee a voies et marquages, trottoirs et bordures, passages pietons, lampadaires, arbres d'alignement, voitures garees, ponts a parapets et piles au-dessus de l'eau. De loin, seule la silhouette des tours reste.
- **Arrivee.** Le joueur apparait sur une rive degagee, face au soleil couchant : il reste fige a 5 degres de l'horizon jusqu'a ce qu'on change l'heure avec T et G, ou qu'on lance `--day-length 1800`.

## Son

Aucun fichier audio : tout est calcule en direct par `audio.rs`, sur la carte son par defaut (`--mute` ou M pour couper).

- **Voiture.** Le moteur suit le regime et la charge de la dynamique : harmoniques de l'allumage d'un V8 (un cycle sur deux tours, l'ordre 8 pour l'allumage, l'ordre 4 pour le grondement), bruit d'echappement a chaque explosion, filtre qui s'ouvre avec la charge, souffle d'admission, limiteur, petarades au leve de pied. S'y ajoutent le claquement des passages de rapport, le crissement des pneus quand ils depassent leur pic d'adherence (sur l'asphalte seulement), le roulement et les graviers hors de la route, le vent qui monte avec la vitesse, et les chocs. En vue conducteur, l'habitacle etouffe les aigus.
- **Ambiance.** Calculee autour de la camera quatre fois par seconde : vent en rafales (plus fort en altitude), feuillages et oiseaux pres des arbres le jour, grillons les nuits chaudes, vagues sur un rivage, eau qui coule pres d'une riviere ou d'un lac, rumeur et klaxons lointains en ville, souffle de la vitesse en vol. Les sons du sol s'effacent quand on prend de la hauteur.
- **Cout.** Moins de 4 % d'un coeur avec toutes les sources a la fois (mesure par un test). Le thread audio n'attend jamais le jeu.
- **Ecouter sans lancer le jeu.** Une demonstration de 48 s (depart arrete avec la vraie dynamique, derapage, choc, bois, nuit, rivage, ville) :

```powershell
$env:TERRA_WAV = "$PWD\son_demo.wav"; .\cargo.ps1 test --release --lib write_a_listening_demo
```

## Performances

`TERRA_PROFILE=1` affiche toutes les 5 s le temps de chaque phase de l'image (jeu, streaming, envoi des chunks, interface, rendu), le detail du streaming (chunks choisis, charges, generes), le temps GPU de chaque passe (ombres, scene, nuages, reflets, eau, halo, sortie) et, au demarrage, chaque etape de la creation de la planete (eau, villes, autoroutes, plan des villes). `--no-vsync --autopilot VITESSE` fait un vol de mesure de 30 s et signale les images de plus de 12 ms.

`gen_bench [GRAINE]` mesure la generation seule : creation de la planete, plan des villes, chunks au depart et en ville par niveau de detail, arbres et herbes, detail par etape d'un chunk, et sondes de terrain sur 1 coeur puis sur tous.

```powershell
.\cargo.ps1 run --release --bin gen_bench -- 3
```

Mesures (planete 3, Ryzen 5 9600X, RX 9070 XT, sans vsync) :

| Mesure | Debut | Avant cette passe | Apres |
| --- | --- | --- | --- |
| Survol de ville a 40 m/s, image mediane | 4,49 ms | 2,59 ms | 2,18 ms |
| Survol de ville, 99e centile | | 3,45 ms | 2,73 ms |
| Vol rapide a 250 m/s, image mediane | 5,41 ms | 2,65 ms | 1,95 ms |
| Vol rapide, 99e centile | | 5,3 ms | 3,0 ms |
| Chunk moyen (1 coeur) | | 32,4 ms | 19,4 ms |
| Gain du calcul sur 12 threads | | x2,4 | x7,6 |
| Chunks generes en 5 s de jeu | | 920 | 1526 |
| Creation de la planete | | 8,0 s | 3,6 s |
| Plan des villes | | 2,2 s | 0,36 s |
| Ombres GPU (1080p, 60 m) | | 0,93 ms | 0,5 ms |

- Le rayon de la planete etait lu dans une variable de thread a chaque echantillon de bruit ; sous Windows (MinGW) cette lecture coutait 14 ns et empechait le calcul de profiter des coeurs. Un compteur atomique l'evite quand aucune planete n'est en creation.
- Le bruit calcule ses 8 coins une seule fois et a une variante sans derivees ; les champs des provinces partagent la projection sur le cube ; les couleurs d'un sommet partagent un seul calcul du site.
- La pente des ravines est interpolee sur une grille fixe de 175 m (ecart moyen 0,8 deg avec la pente exacte) et mise en cache entre les chunks.
- Index des rivieres, plan des villes, routes noyees et autoroutes sont calcules en parallele (`par_map`) ; les threads de generation ont une priorite basse pour ne pas voler d'images au rendu.
- Le quadtree n'est re-selectionne que si la camera a bouge ou si de nouveaux chunks sont arrives.
- Les objets ne sont dessines que dans les cascades d'ombre qu'ils touchent ; la cascade lointaine utilise la geometrie du niveau de detail parent ; l'eau ne calcule le ciel que si le reflet ecran manque.
- Les instances d'arbres et d'herbes (plus de 100 000 par image) se preparent sur plusieurs threads et partent directement dans le tampon du GPU ; les tables de chunks utilisent FxHash.

## Tests

```powershell
.\cargo.ps1 test --lib
```

## Limites actuelles

- Les voitures garees et les lampadaires sont de simples volumes ; seule la voiture conduite est detaillee.
- Les phares ne projettent pas de lumiere sur la route. Les vitres sont opaques de l'exterieur.
- La voiture n'est pas deformable comme dans BeamNG (pas de poutres ni de noeuds) : la caisse reste rigide, seules des bosses s'affichent, et les degats ne changent pas la conduite. Pas d'usure ni d'echauffement des pneus. La direction au clavier est progressive mais moins fine qu'un volant. Il n'y a pas de circulation.
- Routes de montagne : en cul-de-sac (pas de route de col qui redescende de l'autre cote) ; pas de tunnels ni de ponts, donc une route s'arrete avant une falaise ou une riviere, et une route qui en croiserait une autre est abandonnee. Les reliefs de la planete restent moderes : de 150 a 1 000 m de denivele. Le calcul ajoute environ 1,7 s a la creation de la planete.
- Circuits : pas d'adversaires ni de chronometre. La route d'acces est une ligne droite. Une ville sans terrain plat libre autour d'elle n'a pas de circuit, et un meme trace peut revenir. Le detail des facades s'efface avec la distance.
- Les rampes des ponts peuvent faire une marche aux carrefours, et les batiments s'enfoncent un peu sur les pentes.
- Les autoroutes partent du bord des villes, dans le prolongement d'une avenue : pas d'echangeurs ni de bretelles.
  Deux autoroutes qui se croisent passent l'une au-dessus de l'autre, sans se raccorder.
- Les iles et continents separes par une mer profonde (plus de 70 m) restent sans liaison : il peut y avoir plusieurs reseaux.
- Avec des autoroutes, une planete se genere plus lentement : environ 10 s pour 140 villes, 18 s pour 400.
- Les panneaux d'autoroute n'ont pas de texte. Certains hameaux de faubourg n'ont pas de rue d'acces.
- Pas de circulation ni d'habitants. Vue de loin la nuit, la ville parait un peu pale.
- Une rue qui s'avance sur l'eau est retiree, mais quelques passerelles restent dans les lagunes.
- Pas de grottes ni de surplombs : le terrain est un champ de hauteur. Les falaises des canyons et des tours sont des pentes tres raides.
- Les ravines restent une illusion locale : elles ne rejoignent pas toujours les rivieres simulees.
- Pas de cascades ni de deltas : un fort denivele donne des rapides, et les rivieres arrivent a la mer par un seul lit.
- Tous les lacs debordent : pas de lacs sales sans exutoire dans les deserts.
- Le reseau de rivieres s'arrete aux bassins d'environ 300 a 800 km2 : plus petit, il n'y a pas de ruisseau, seulement des vallons secs.
- Pas d'audio, de PNJ, de sauvegarde ni de chargement de modeles 3D.
- Un jeu ne peut pas encore ajouter ses propres objets a la scene.
- La physique se limite au sol, aux troncs et aux rochers. Il n'y a pas de gravite orbitale : le vol est libre.
- Les nuages n'ont pas de filtrage temporel : ils gardent un leger grain de pres.
- Le reflet ne montre que ce qui est a l'ecran ; le reste vient du ciel calcule, sans nuages.
- Les arbres proches restent des volumes simples habilles au pixel, sans vraies feuilles.
