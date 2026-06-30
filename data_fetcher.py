import json
import urllib.request

def fetch_dataset_from_network(url, allowed_labels):
    """
    Mengunduh data latih riil dari url target (misal: JSON/CSV dari sosial media)
    dan mengembalikan array data latih hasil ekstraksi yang terjamin valid.
    """
    if not url:
        raise ValueError("URL target tidak boleh kosong.")
        
    try:
        # Request HTTP dengan User-Agent untuk menghindari pencegatan bot
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
        
        parsed = json.loads(html)
        if not isinstance(parsed, list):
            raise ValueError("Struktur data harus berupa JSON Array berkas opini.")
            
        import random
        user_bases = [
            "petani", "anggota", "warga", "pengurus", "nasabah", "koperasi", "umkm", "desa",
            "tani", "sawah", "modal", "dana", "investor", "simpanan", "pinjaman", "shu",
            "budi", "ani", "siti", "joko", "eko", "rudi", "iwan", "wawan", "sri", "dewi",
            "agus", "bambang", "hendra", "supri", "mamat", "udin", "asep", "dadang", "cecep"
        ]

        valid_items = []
        for item in parsed:
            txt = item.get("text", "").strip()
            lbl = item.get("label", "").strip()
            username = item.get("username", "").strip()
            if not username:
                state = random.getstate()
                random.seed(txt)
                username = f"@{random.choice(user_bases)}_{random.randint(10, 999)}"
                random.setstate(state)
            elif not username.startswith("@"):
                username = "@" + username
            
            # Cari label emosi yang cocok (case-insensitive) dengan EMOTIONS_METADATA keys
            label_cap = next((c for c in allowed_labels if c.lower() == lbl.lower()), None)
            
            if label_cap and txt:
                valid_items.append({
                    "username": username,
                    "text": txt,
                    "label": label_cap
                })
                
        if not valid_items:
            raise ValueError("Tidak ada opini ulasan berlabel emosi valid yang berhasil diekstrak.")
            
        return valid_items
        
    except Exception as e:
        raise RuntimeError(f"Gagal menarik data dari jaringan internet: {str(e)}")
