import sys
import os
import json
import asyncio
import re
import random

# Memastikan modul dari direktori kerja saat ini dapat diimpor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from twifork import Client
except ImportError:
    try:
        from twikit import Client
    except ImportError:
        print("="*70)
        print("Pustaka 'twifork' atau 'twikit' belum terinstal di sistem Anda.")
        print("Silakan jalankan perintah berikut untuk menginstalnya:")
        print("  pip install twifork")
        print("="*70)
        sys.exit(1)

from dataset_manager import load_dataset
from naive_bayes import NaiveBayesClassifier

COOKIE_FILE = "twitter_cookies.json"
AUTH_FILE = "twitter_auth.json"

async def scrape_twitter_real(query, max_tweets):
    """
    Fungsi reusable untuk melakukan scraping Twitter/X secara real-time.
    Menggunakan cookies yang ada di COOKIE_FILE.
    Mengklasifikasikan emosi menggunakan model Naive Bayes yang aktif.
    """
    client = Client('en-US')
    
    # 1. Coba memuat sesi cookie yang tersimpan
    logged_in = False
    if os.path.exists(COOKIE_FILE):
        try:
            with open(COOKIE_FILE, 'r', encoding='utf-8') as f:
                cookie_data = json.load(f)
            
            # Jika berkas berupa list/array dari ekstensi browser (Cookie-Editor/EditThisCookie)
            if isinstance(cookie_data, list):
                cookie_dict = {}
                for c in cookie_data:
                    name = c.get("name")
                    value = c.get("value")
                    if name and value:
                        cookie_dict[name] = value
                cookie_data = cookie_dict
            
            # Simpan ulang dalam format kamus/dict bersih yang didukung twikit
            with open(COOKIE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cookie_data, f, ensure_ascii=False, indent=2)
                
            client.load_cookies(COOKIE_FILE)
            logged_in = True
        except Exception as e:
            raise Exception(f"Gagal memuat cookie: {e}. Pastikan file twitter_cookies.json valid.")
            
    if not logged_in:
        raise Exception("Belum login. Silakan ekspor cookie akun Twitter/X Anda ke file 'twitter_cookies.json' terlebih dahulu.")

    # 2. Melakukan pencarian tweet
    all_tweets = []
    try:
        tweets = await client.search_tweet(query, product='Latest', count=min(20, max_tweets))
        if tweets:
            all_tweets.extend(tweets)
            
            # Paginasi untuk mengambil halaman-halaman berikutnya
            while len(all_tweets) < max_tweets:
                # Jeda waktu acak antara 5 s.d 10 detik agar tidak dicurigai sebagai bot/spam
                await asyncio.sleep(random.randint(5, 10))
                
                tweets = await tweets.next()
                if not tweets:
                    break
                all_tweets.extend(tweets)
    except Exception as e:
        raise Exception(f"Gagal mencari tweet di Twitter/X: {e}")

    if not all_tweets:
        return []

    # 3. Melatih model Naive Bayes lokal untuk auto-labeling emosi
    dataset = load_dataset()
    classifier = NaiveBayesClassifier()
    classifier.train(dataset)

    # 4. Ekstraksi data & Prediksi Emosi
    scraped_dataset = []
    for idx, t in enumerate(all_tweets):
        text = t.text.strip()
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            continue
            
        username = t.user.screen_name
        if username and not username.startswith("@"):
            username = "@" + username
        elif not username:
            username = f"@user_{idx}"
            
        # Prediksi label emosi secara otomatis menggunakan Naive Bayes Classifier aktif
        prediction = classifier.predict(text)["prediction"]
        
        scraped_dataset.append({
            "username": username,
            "text": text,
            "label": prediction
        })
        
    return scraped_dataset

async def run_scraper():
    print("="*70)
    print("         REAL TWITTER/X SCRAPER (AUTHENTICATED) - SENTIMENTUM")
    print("="*70)
    
    query = "koperasi"
    max_tweets = 10
    
    if len(sys.argv) > 1:
        query = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            max_tweets = int(sys.argv[2])
        except ValueError:
            max_tweets = 10
            
    print(f"[*] Kata kunci pencarian: '{query}'")
    print(f"[*] Target jumlah komentar: {max_tweets}")
    
    # Deteksi otomatis jika cookies salah ditempel di twitter_auth.json
    if os.path.exists(AUTH_FILE):
        try:
            with open(AUTH_FILE, 'r', encoding='utf-8') as f:
                auth_data = json.load(f)
            if isinstance(auth_data, list):
                print("[*] Mendeteksi cookies di 'twitter_auth.json'. Memindahkan secara otomatis ke 'twitter_cookies.json'...")
                with open(COOKIE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(auth_data, f, ensure_ascii=False, indent=2)
                template = {
                    "username": "USERNAME_TWITTER_ANDA",
                    "email": "EMAIL_TWITTER_ANDA",
                    "password": "PASSWORD_TWITTER_ANDA"
                }
                with open(AUTH_FILE, 'w', encoding='utf-8') as f:
                    json.dump(template, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    try:
        results = await scrape_twitter_real(query, max_tweets)
    except Exception as e:
        print(f"[!] {e}")
        return

    if not results:
        print("[!] Tidak ada tweet baru yang ditemukan.")
        return

    print(f"[*] Berhasil mengekstrak {len(results)} tweet riil. Memulai klasifikasi...")
    for idx, item in enumerate(results):
        try:
            print(f"  [{idx+1}] {item['username']}: {item['text'][:60]}... -> Emosi: {item['label']}")
        except UnicodeEncodeError:
            clean_text = item['text'][:60].encode('ascii', errors='replace').decode('ascii')
            print(f"  [{idx+1}] {item['username']}: {clean_text}... -> Emosi: {item['label']}")

    # Menggabungkan dengan data lama (Incremental Save)
    output_file = "scraped_tweets.json"
    existing_data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
        except Exception:
            existing_data = []

    existing_texts = {item["text"].strip().lower() for item in existing_data}
    
    new_count = 0
    for item in results:
        if item["text"].strip().lower() not in existing_texts:
            existing_data.append(item)
            new_count += 1
            
    if new_count > 0 or not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
        print("="*70)
        print(f"[V] SELESAI! Menambahkan {new_count} ulasan baru unik.")
        print(f"    Total data tersimpan di '{output_file}' saat ini: {len(existing_data)} ulasan.")
        print("="*70)
    else:
        print("="*70)
        print("[!] Selesai. Semua tweet yang ditarik kali ini sudah ada di dalam database.")
        print(f"    Total data tersimpan di '{output_file}' saat ini: {len(existing_data)} ulasan.")
        print("="*70)

if __name__ == "__main__":
    asyncio.run(run_scraper())
