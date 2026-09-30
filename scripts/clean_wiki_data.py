import json
import re


# =============================================================================
# DÉTECTION DES ANOMALIES
# =============================================================================

def est_desambiguisation(texte):
    """
    Retourne True si l'extrait est une page de désambiguïsation Wikipedia.
    Ces pages listent plusieurs significations au lieu de décrire la plante.
    Exemple : "Kale peut désigner : Botanique, Patronymes, Toponymes..."
    """
    debut = texte.strip().lower()[:150]
    return any(signal in debut for signal in [
        'peut désigner',
        'peut faire référence',
        'est un terme',
        'est un nom qui peut',
        'désigne plusieurs',
    ])


def est_hors_sujet(texte):
    """
    Retourne True si l'extrait ne parle pas d'une plante.
    Vérifie l'absence de tout mot-clé botanique dans les 300 premiers caractères.
    Exemple : "Bilibili" renvoie vers un site de vidéos chinois.
    """
    mots_botaniques = [
        'plante', 'espèce', 'famille', 'fruit', 'fleur', 'feuille',
        'racine', 'cultivé', 'botanique', 'arbre', 'arbuste', 'graine',
        'rhizome', 'légume', 'céréale', 'herbacée', 'annuelle', 'vivace',
        'tropical', 'comestible', 'alimentaire', 'médicinal',
    ]
    apercu = texte.lower()[:300]
    return not any(mot in apercu for mot in mots_botaniques)


# =============================================================================
# NETTOYAGE DU TEXTE
# =============================================================================

def nettoyer_extrait(texte):
    """
    Nettoie un extrait valide en 4 étapes :
      1. Supprime les marqueurs de sections  == Titre ==
      2. Supprime les références             [1], [2], [note 1]
      3. Remplace les sauts de ligne         par des espaces
      4. Supprime les espaces multiples
    """
    texte = re.sub(r'\n?==+[^=\n]+=+\n?', ' ', texte)   # == Titre ==
    texte = re.sub(r'\[\d+\]|\[note\s*\d+\]', '', texte) # [1], [note 1]
    texte = texte.replace('\n', ' ')                      # sauts de ligne
    texte = re.sub(r'  +', ' ', texte)                   # espaces multiples
    return texte.strip()


# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

def main():
    input_file  = "wiki_data.json"
    output_file = "wiki_data_clean.json"

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f"\n{'='*55}")
    print(f"  Nettoyage de {input_file} — {len(raw_data)} entrées")
    print(f"{'='*55}\n")

    valides  = []
    rejetes  = []

    for i, page_list in enumerate(raw_data):
        page    = page_list[0]
        titre   = page.get('title', f'ENTRÉE_{i}')
        missing = page.get('missing', False)
        extract = page.get('extract', '') or ''

        print(f"[{i:02}] {titre}")

        # Article manquant sur Wikipedia
        if missing:
            rejetes.append((i, titre, 'Article manquant'))
            print(f"        rejeté — article manquant")
            continue

        # Extrait vide
        if extract.strip() == '':
            rejetes.append((i, titre, 'Extrait vide'))
            print(f"         rejeté — extrait vide")
            continue

        # Page de désambiguïsation
        if est_desambiguisation(extract):
            rejetes.append((i, titre, 'Désambiguïsation'))
            print(f"         rejeté — page de désambiguïsation")
            continue

        # Contenu hors-sujet
        if est_hors_sujet(extract):
            rejetes.append((i, titre, 'Hors-sujet'))
            print(f"         rejeté — contenu hors-sujet")
            continue

        # Entrée valide : nettoyage et sauvegarde
        valides.append({
            'titre'   : titre,
            'extrait' : nettoyer_extrait(extract),
        })
        print(f"       ✓  ok")

    # Sauvegarde
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(valides, f, ensure_ascii=False, indent=4)

    # Rapport
    print(f"\n{'='*55}")
    print(f"  Résultat")
    print(f"{'='*55}")
    print(f"  Entrées traitées  : {len(raw_data)}")
    print(f"  Entrées valides   : {len(valides)}")
    print(f"  Entrées rejetées  : {len(rejetes)}")

    if rejetes:
        print(f"\n  Entrées rejetées :")
        for index, titre, raison in rejetes:
            print(f"    [{index:02}] {titre} — {raison}")

    print(f"\n  Fichier produit : {output_file}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
