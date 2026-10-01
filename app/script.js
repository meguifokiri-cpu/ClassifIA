// ══════════════════════════════════════════════════════════════
// script.js — Logique du frontend PlantID
// Gère : la zone de dépôt d'image, l'appel à l'API /predict,
// et l'affichage de la fiche botanique retournée.
// ══════════════════════════════════════════════════════════════

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const submitBtn = document.getElementById('submitBtn');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const resultName = document.getElementById('resultName');
const resultLatin = document.getElementById('resultLatin');
const confidencePct = document.getElementById('confidencePct');
const confidenceFill = document.getElementById('confidenceFill');
const warningBlock = document.getElementById('warningBlock');
const descriptionBlock = document.getElementById('descriptionBlock');
const infoGrid = document.getElementById('infoGrid');
const apiUrlInput = document.getElementById('apiUrl');

let selectedFile = null;

// ── Interaction : clic ou glisser-déposer sur la zone de dépôt ──
dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.style.display = 'block';
  };
  reader.readAsDataURL(file);
  submitBtn.disabled = false;
  resultEl.classList.remove('visible');
  statusEl.className = 'status';
}

// ── Appel à l'API /predict ────────────────────────────────────
submitBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  statusEl.className = 'status loading';
  statusEl.textContent = 'Analyse en cours...';
  resultEl.classList.remove('visible');
  submitBtn.disabled = true;

  const formData = new FormData();
  formData.append('image', selectedFile);

    try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      // Erreur renvoyée par l'API : 400 (fichier invalide) ou 404 (espèce absente de la base)
      statusEl.className = 'status error';
      statusEl.textContent = data.error || `Erreur serveur (${response.status})`;
      resultEl.classList.remove('visible');
      return;
    }

    displayResult(data);
    statusEl.className = 'status';
  } catch (err) {
    statusEl.className = 'status error';
    statusEl.textContent = `Erreur : ${err.message}. Vérifie que l'API tourne et que l'URL est correcte.`;
  } finally {
    submitBtn.disabled = false;
  }
});

// ── Affichage de la fiche botanique ───────────────────────────
function ligneInfo(label, valeur) {
  if (!valeur) return '';
  return `
    <div class="info-item">
      <div class="info-label">${label}</div>
      <div class="info-value">${valeur}</div>
    </div>
  `;
}

function displayResult(data) {
  // Mapping exact des champs retournés par predict.py (routes.py + predict_bp)
  resultName.textContent = data.nom_commun || data.nom_dossier || 'Résultat reçu';
  resultLatin.textContent = data.nom_latin || '';

  const pct = data.confiance ?? 0; // déjà en % (ex: 91.34), pas besoin de x100
  confidencePct.textContent = `${pct}%`;
  confidenceFill.style.width = `${pct}%`;

  warningBlock.innerHTML = data.avertissement_securite
    ? `<div class="warning-block">⚠️ ${data.avertissement_securite}</div>`
    : '';

  descriptionBlock.innerHTML = data.description_wikipedia
    ? `<div class="description-block">${data.description_wikipedia}</div>`
    : '';

  infoGrid.innerHTML = [
    ligneInfo('Famille', data.famille),
    ligneInfo('Utilité principale', data.utilite_principale),
    ligneInfo('Hauteur', data.hauteur),
    ligneInfo('Besoin en eau', data.besoin_eau),
    ligneInfo('Exposition', data.exposition),
    ligneInfo('Saison de floraison', data.saison_floraison),
    ligneInfo('Origine', data.origine),
  ].join('');

  resultEl.classList.add('visible');
}