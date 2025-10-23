from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pymatgen.core import Structure, Lattice
import os
import traceback

app = Flask(__name__)
CORS(app)

ELEMENTS = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca']
EL_TO_IDX = {e:i for i,e in enumerate(ELEMENTS)}
MAX_ATOMS = 64
N_ELEM = len(ELEMENTS)
LATENT_DIM = 128

class SpaceGroupEmbedding(nn.Module):
    def __init__(self, max_sg=230, emb_dim=64):
        super().__init__()
        self.embed = nn.Embedding(max_sg+1, emb_dim)

    def forward(self, x):
        return self.embed(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sg_emb = SpaceGroupEmbedding()
        in_dim = 9 + MAX_ATOMS*3 + MAX_ATOMS*N_ELEM
        self.fc1 = nn.Linear(in_dim + 64, 512)
        self.fc_mu = nn.Linear(512, LATENT_DIM)
        self.fc_logvar = nn.Linear(512, LATENT_DIM)

    def forward(self, lattice, frac, species_oh, sg_idx):
        x = torch.cat([lattice.view(lattice.size(0), -1), frac.view(frac.size(0), -1), species_oh.view(species_oh.size(0), -1)], dim=-1)
        sg_e = self.sg_emb(sg_idx)
        x = torch.cat([x, sg_e], dim=-1)
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sg_emb = SpaceGroupEmbedding()
        out_dim = 9 + MAX_ATOMS*3 + MAX_ATOMS*N_ELEM
        self.fc1 = nn.Linear(LATENT_DIM + 64, 512)
        self.fc_out = nn.Linear(512, out_dim)

    def forward(self, z, sg_idx):
        sg_e = self.sg_emb(sg_idx)
        x = torch.cat([z, sg_e], dim=-1)
        h = F.relu(self.fc1(x))
        out = self.fc_out(h)
        lattice = out[:, :9].view(-1,3,3)
        frac = out[:,9:9+MAX_ATOMS*3].view(-1,MAX_ATOMS,3)
        species_logits = out[:,9+MAX_ATOMS*3:].view(-1,MAX_ATOMS,N_ELEM)
        return lattice, frac, species_logits

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, lattice, frac, species_oh, sg_idx):
        mu, logvar = self.enc(lattice, frac, species_oh, sg_idx)
        z = self.reparameterize(mu, logvar)
        lattice_rec, frac_rec, species_logits = self.dec(z, sg_idx)
        return lattice_rec, frac_rec, species_logits, mu, logvar

model = CVAE()
model_path = os.path.join(os.path.dirname(__file__), 'cvae_latest.pt')

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"Model loaded from {model_path}")
else:
    print(f"Warning: Model file not found at {model_path}")

def parse_composition(comp_str):
    comp_dict = {}
    i = 0
    while i < len(comp_str):
        if comp_str[i].isupper():
            el = comp_str[i]
            i += 1
            if i < len(comp_str) and comp_str[i].islower():
                el += comp_str[i]
                i += 1

            num_str = ''
            while i < len(comp_str) and (comp_str[i].isdigit() or comp_str[i] == '.'):
                num_str += comp_str[i]
                i += 1

            count = float(num_str) if num_str else 1.0
            comp_dict[el] = count
    return comp_dict

def infer_structure(spacegroup_idx: int, comp_dict: dict, num_atoms: int = 8, temperature: float = 1.0):
    try:
        z = torch.randn((1, LATENT_DIM)) * temperature
        sg_idx = torch.tensor([spacegroup_idx], dtype=torch.long)

        with torch.no_grad():
            lat_pred, frac_pred, species_logits = model.dec(z, sg_idx)

        lat = lat_pred.detach().numpy()[0]
        frac = frac_pred.detach().numpy()[0]

        elements_from_comp = list(comp_dict.keys())
        num_elements = len(elements_from_comp)
        species_symbols = [elements_from_comp[i % num_elements] for i in range(num_atoms)]

        lattice = Lattice(lat)
        sites = []

        for i in range(num_atoms):
            species_symbol = species_symbols[i]
            if species_symbol in ELEMENTS:
                sites.append({
                    'species': [{"element": species_symbol, "occu": 1}],
                    'abc': frac[i].tolist()
                })

        if not sites:
            return None

        structure = Structure.from_dict({
            'lattice': lattice.as_dict(),
            'sites': sites,
            'charge': 0
        })

        return structure
    except Exception as e:
        print(f"Error in infer_structure: {str(e)}")
        traceback.print_exc()
        return None

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()

        spacegroup = int(data.get('spaceGroup', 225))
        composition_str = data.get('composition', 'FeO')
        num_atoms = int(data.get('numAtoms', 8))
        temperature = float(data.get('temperature', 1.0))

        comp_dict = parse_composition(composition_str)

        if not comp_dict:
            return jsonify({'error': 'Invalid composition format'}), 400

        structure = infer_structure(spacegroup, comp_dict, num_atoms, temperature)

        if structure is None:
            return jsonify({'error': 'Failed to generate structure'}), 500

        cif_string = structure.to(fmt='cif')
        formula = structure.composition.reduced_formula

        return jsonify({
            'cif': cif_string,
            'formula': formula,
            'spacegroup': spacegroup,
            'num_atoms': len(structure.sites)
        })

    except Exception as e:
        print(f"Error in /api/generate: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': os.path.exists(model_path)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
