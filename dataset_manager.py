import os
import json
import random

# Path lokasi penyimpanan berkas data latih JSON
DATASET_PATH = 'dataset_sosmed.json'

# ==========================================================================
# TEMPLAT DATA LATIH DEFAULT (7 Kategori Emosi - 715 Ulasan Unik Per Kategori)
# ==========================================================================
TEMPLATES = {
    "Senang": {
        "pembuka": ["Saya sangat gembira", "Sangat senang sekali", "Bersyukur sekali", "Senang rasanya", "Kami sangat puas", "Luar biasa senang", "Masyarakat gembira", "Sangat terbantu", "Saya sungguh bersyukur", "Sangat mengapresiasi"],
        "subjek": ["petani desa", "anggota koperasi", "kami para warga", "seluruh pengurus", "pelaku UMKM", "warga desa", "nasabah koperasi", "anggota tani"],
        "tindakan": ["menerima kemudahan", "mendapatkan pelayanan terbaik", "merasakan keuntungan", "menikmati bagi hasil SHU", "memperoleh bantuan modal", "menikmati subsidi pupuk", "memperoleh dispensasi"],
        "kebijakan": ["dari program baru koperasi", "atas kebijakan bunga rendah", "karena pembagian SHU adil", "melalui dana bergulir desa", "terkait transparansi keuangan", "dalam penyediaan bibit gratis", "atas kemudahan kredit"],
        "keterangan": ["yang berjalan sangat lancar", "dengan proses cepat dan mudah", "secara terbuka serta kekeluargaan", "yang sangat mensejahterakan kami", "tanpa hambatan birokrasi", "dengan petugas yang ramah"],
        "penutup": ["terima kasih koperasi!", "semoga koperasi terus jaya!", "ini sangat luar biasa.", "sangat menyokong ekonomi desa.", "koperasi desa memang terbaik.", "mantap sekali."]
    },
    "Percaya": {
        "pembuka": ["Saya sangat yakin", "Kami menaruh kepercayaan penuh", "Sangat percaya", "Merasa aman", "Masyarakat percaya", "Yakin sekali", "Kami sangat mengandalkan", "Percaya sepenuhnya", "Sangat menaruh harapan", "Optimis sekali"],
        "subjek": ["seluruh anggota", "warga masyarakat", "para petani kecil", "kami selaku nasabah", "masyarakat desa", "para pemegang saham", "kelompok tani", "peternak lokal"],
        "tindakan": ["menitipkan modal usaha", "mempercayakan uang tabungan", "mendukung penuh keputusan", "menyetujui pengelolaan dana", "yakin dengan kepemimpinan", "mengikuti petunjuk pengurus", "mendukung program kerja"],
        "kebijakan": ["kepada pengurus koperasi desa", "pada sistem bagi hasil baru", "terhadap transparansi pembukuan", "mengenai audit keuangan tahunan", "dalam program simpan pinjam", "terkait kepengurusan periode ini", "pada penyaluran dana sosial"],
        "keterangan": ["yang dikelola secara profesional", "karena terbukti jujur dan amanah", "secara terbuka dan akuntabel", "yang selalu mengutamakan kepentingan anggota", "dengan sistem keamanan yang terjamin"],
        "penutup": ["koperasi ini sangat terpercaya.", "kami tidak ragu lagi.", "semoga amanah ini terjaga.", "pengurus bekerja dengan jujur.", "kepercayaan kami terbayar lunas."]
    },
    "Terkejut": {
        "pembuka": ["Sungguh terkejut", "Sangat tidak menyangka", "Kaget dan heran", "Luar biasa terkejut", "Saya heran sekaligus kagum", "Hampir tidak percaya", "Kaget luar biasa", "Wah mengagumkan sekali", "Tiba-tiba terkejut", "Kagum sekali"],
        "subjek": ["saya pribadi", "kami semua", "seluruh anggota", "para warga desa", "petani di sini", "nasabah koperasi", "masyarakat luas"],
        "tindakan": ["melihat hasil SHU melonjak", "menerima pelayanan secepat ini", "mendengar bunga pinjaman diturunkan", "melihat inovasi digital koperasi", "mengetahui koperasi mendapat penghargaan", "menerima kejutan hadiah undian", "melihat pesatnya kemajuan usaha"],
        "kebijakan": ["dari kebijakan pengurus baru", "pada rapat anggota tahunan kemarin", "terkait efisiensi operasional koperasi", "dalam sistem pencairan dana instan", "mengenai pembebasan denda simpan pinjam", "dari subsidi tak terduga"],
        "keterangan": ["yang ternyata sangat menguntungkan", "jauh melampaui ekspektasi kami", "secara tiba-tiba tanpa berbelit-belit", "yang membuat kami kagum sekali", "dengan kemajuan yang sangat pesat"],
        "penutup": ["ini kejutan yang sangat positif!", "sungguh luar biasa perkembangannya!", "koperasi desa membuat lompatan besar.", "kami kagum dengan kinerja ini.", "luar biasa, pertahankan!"]
    },
    "Netral": {
        "pembuka": ["Koperasi desa mengumumkan", "Informasi resmi menyatakan", "Sesuai dengan ketentuan", "Berdasarkan laporan tata tertib", "Secara umum koperasi", "Pengurus menyampaikan", "Adapun jadwal kegiatan", "Sesuai rilis berkas", "Mengacu pada prosedur"],
        "subjek": ["kegiatan simpan pinjam", "rapat anggota tahunan (RAT)", "kantor pelayanan koperasi", "pendaftaran anggota baru", "pembagian pupuk bersubsidi", "penyetoran iuran wajib", "pengelolaan berkas pinjaman"],
        "tindakan": ["dilaksanakan setiap bulan", "dimulai pada pukul delapan pagi", "memerlukan dokumen fotokopi KTP", "bertempat di balai pertemuan desa", "dikelola oleh bagian administrasi", "menerapkan bunga simpanan berkala", "diawasi oleh dinas kabupaten"],
        "kebijakan": ["sesuai anggaran dasar koperasi", "berdasarkan keputusan rapat pengurus", "merupakan bagian program kerja tahunan", "terkait peraturan simpanan pokok", "mengikuti instruksi dinas koperasi", "berdasarkan ad art terbaru"],
        "keterangan": ["yang berlaku bagi semua anggota", "secara administratif dan prosedural", "sesuai dengan urutan nomor antrean", "dengan ketentuan yang tertulis di papan pengumuman", "tanpa adanya pemungutan biaya tambahan"],
        "penutup": ["demikian informasi tersebut disampaikan.", "harap anggota maklum adanya.", "sesuai dengan agenda kerja.", "kegiatan berjalan seperti biasa.", "berkas harus dilengkapi."]
    },
    "Sedih": {
        "pembuka": ["Sangat sedih melihat", "Saya kecewa sekali", "Sangat prihatin dengan", "Merasa kecewa dan lesu", "Aduh kasihan sekali", "Kecewa mendalam", "Sedih rasanya", "Sayang sekali", "Masyarakat mengeluh kecewa", "Merasa tersisihkan"],
        "subjek": ["kondisi koperasi kita", "nasib petani kecil", "pelayanan pengurus", "keberadaan unit usaha", "tabungan masa depan kami", "kegiatan rapat tahunan", "nasib modal warga"],
        "tindakan": ["yang kian hari makin sepi", "mengalami penurunan pendapatan", "tidak berkembang sama sekali", "terpuruk akibat salah urus", "kehilangan dana cadangan", "mengabaikan keluhan anggota", "menunjukkan kinerja buruk"],
        "kebijakan": ["atas dampak regulasi baru", "akibat pembagian hasil yang kecil", "karena minimnya transparansi dana", "terkait langkanya pasokan pupuk", "dalam penanganan kredit macet", "akibat kerugian tahun lalu"],
        "keterangan": ["yang membuat warga merugi", "dengan nasib yang tidak menentu", "sehingga anggota merasa terabaikan", "yang membawa kemunduran bagi desa", "tanpa ada solusi nyata dari pengurus"],
        "penutup": ["semoga ada perbaikan ke depan.", "kami hanya bisa pasrah.", "sungguh memprihatinkan nasib kami.", "sangat disayangkan pelayanan memburuk.", "sedih melihat koperasi hampir bangkrut."]
    },
    "Takut": {
        "pembuka": ["Saya sangat cemas", "Khawatir sekali", "Takut jika nanti", "Ngeri membayangkan", "Masyarakat dilingkupi rasa takut", "Sangat khawatir", "Cemas dan gelisah", "Khawatir uang hilang", "Takut tidak kebagian", "Was-was rasanya"],
        "subjek": ["tabungan simpanan kami", "nasib sertifikat tanah agunan", "kelangsungan hidup koperasi", "stok pupuk bersubsidi bulan depan", "kesehatan keuangan pengurus", "potensi likuidasi koperasi", "keamanan sertifikat anggota"],
        "tindakan": ["akan hilang tanpa jejak", "dibawa kabur oleh oknum", "mengalami kebangkrutan total", "tidak bisa ditarik kembali", "disalahgunakan untuk kepentingan pribadi", "macet dan tidak terbayar", "mengalami penyitaan aset"],
        "kebijakan": ["karena desas-desus kas kosong", "akibat pengelolaan dana yang buruk", "imbas dari isu investasi bodong", "terkait audit eksternal yang ditutupi", "dalam skema peminjaman berisiko tinggi", "akibat manajemen yang tidak stabil"],
        "keterangan": ["yang membahayakan aset kami", "tanpa adanya jaminan keamanan", "sehingga menimbulkan kepanikan warga", "yang membuat kami tidak bisa tidur", "dengan ketidakpastian hukum yang ada"],
        "penutup": ["apakah uang kami aman?", "semoga kekhawatiran ini tidak terjadi.", "kami takut rugi besar.", "tolong jelaskan keamanannya.", "kami cemas jika koperasi kolaps."]
    },
    "Marah": {
        "pembuka": ["Saya benar-benar geram", "Sangat marah", "Ini tidak adil!", "Sangat kesal dan emosi", "Brengsek sekali pengurus", "Kami protes keras", "Benar-benar keterlaluan!", "Korupsi merajalela!", "Muak sekali melihat", "Ini perlakuan biadab"],
        "subjek": ["pengurus koperasi desa", "tindakan pilih kasih petugas", "penyelewengan dana anggota", "pembagian pupuk bersubsidi", "monopoli pinjaman sepihak", "penundaan RAT sepihak", "petugas lapangan koperasi"],
        "tindakan": ["yang menimbun bantuan warga", "melakukan korupsi terang-terangan", "membohongi seluruh anggota", "memeras rakyat kecil dengan bunga tinggi", "menolak memberikan penjelasan keuangan", "menyalahgunakan wewenang jabatan", "memalsukan tanda tangan anggota"],
        "kebijakan": ["demi memperkaya diri sendiri", "yang memihak kepada keluarga mereka saja", "secara curang dan manipulatif", "yang mencekik perekonomian petani", "tanpa memedulikan penderitaan kami", "secara ilegal dan sembunyi-sembunyi"],
        "keterangan": ["yang merugikan rakyat kecil", "tanpa ada rasa bersalah", "secara terang-terangan melanggar aturan", "yang menghancurkan perekonomian desa", "demi keuntungan pribadi mereka"],
        "penutup": ["pecat pengurus sekarang juga!", "kami akan bawa ke jalur hukum!", "ini pemerasan terbuka!", "pengurus tidak tahu malu!", "kembalikan uang kami segera!"]
    }
}

def load_dataset():
    """
    Memuat dataset dari file JSON lokal. 
    Jika tidak ada file, sistem men-generate 5.005 data latih seimbang secara programmatis.
    """
    user_bases = [
        "petani", "anggota", "warga", "pengurus", "nasabah", "koperasi", "umkm", "desa",
        "tani", "sawah", "modal", "dana", "investor", "simpanan", "pinjaman", "shu",
        "budi", "ani", "siti", "joko", "eko", "rudi", "iwan", "wawan", "sri", "dewi",
        "agus", "bambang", "hendra", "supri", "mamat", "udin", "asep", "dadang", "cecep"
    ]

    if os.path.exists(DATASET_PATH):
        try:
            with open(DATASET_PATH, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                
            # Fallback jika ada data yang belum memiliki username
            modified = False
            for idx, item in enumerate(dataset):
                if "username" not in item:
                    state = random.getstate()
                    random.seed(item.get("text", str(idx)))
                    item["username"] = f"@{random.choice(user_bases)}_{random.randint(10, 999)}"
                    random.setstate(state)
                    modified = True
            
            if modified:
                save_dataset(dataset)
            return dataset
        except Exception as e:
            print(f"Gagal membaca file dataset, memuat default: {e}")
            
    # Hasilkan data latih deterministik default
    dataset = []
    categories = list(TEMPLATES.keys())
    for cat in categories:
        temp = TEMPLATES[cat]
        unique_texts = set()
        random.seed(cat)
        
        while len(unique_texts) < 715:
            p = random.choice(temp["pembuka"])
            s = random.choice(temp["subjek"])
            t = random.choice(temp["tindakan"])
            kb = random.choice(temp["kebijakan"])
            kt = random.choice(temp["keterangan"])
            pn = random.choice(temp["penutup"])
            
            sentence = f"{p} {s} {t} {kb} {kt} {pn}"
            unique_texts.add(sentence)
            
        for txt in unique_texts:
            state = random.getstate()
            random.seed(txt)
            username = f"@{random.choice(user_bases)}_{random.randint(10, 999)}"
            random.setstate(state)
            
            dataset.append({
                "username": username,
                "text": txt,
                "label": cat
            })
            
    save_dataset(dataset)
    return dataset

def save_dataset(dataset):
    """ Menyimpan dataset latih aktif ke dalam berkas JSON """
    with open(DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


