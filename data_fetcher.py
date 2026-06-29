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
            
        valid_items = []
        for item in parsed:
            txt = item.get("text", "").strip()
            lbl = item.get("label", "").strip()
            
            # Cari label emosi yang cocok (case-insensitive) dengan EMOTIONS_METADATA keys
            label_cap = next((c for c in allowed_labels if c.lower() == lbl.lower()), None)
            
            if label_cap and txt:
                valid_items.append({
                    "text": txt,
                    "label": label_cap
                })
                
        if not valid_items:
            raise ValueError("Tidak ada opini ulasan berlabel emosi valid yang berhasil diekstrak.")
            
        return valid_items
        
    except Exception as e:
        raise RuntimeError(f"Gagal menarik data dari jaringan internet: {str(e)}")
