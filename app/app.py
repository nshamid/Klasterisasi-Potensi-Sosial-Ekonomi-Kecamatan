import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.decomposition import PCA

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Sosial-Ekonomi Palembang 2025", layout="wide", page_icon="🏙️")

# --- CSS CUSTOM: DESIGN SYSTEM ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

    :root {
        --primary: #fe6e00;
        --primary-dark: #e05f00;
        --primary-soft: #fff1e6;
        --bg: #fcfaf7;
        --surface: #f3f4f6;
        --text: #423d38;
        --text-soft: #7a746c;
        --border: #e7e2da;
        --tinggi: #2f9e44;
        --menengah: #3b6ea5;
        --rendah: #d64545;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: var(--bg);
    }

    h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown, span {
        color: var(--text) !important;
    }

    h1, h2, h3 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
    }

    /* Judul utama (konten) pakai aksen garis tebal */
    section[data-testid="stMain"] h1 {
        border-left: 6px solid var(--primary);
        padding-left: 18px;
        margin-bottom: 0.3em !important;
        margin-left: 4px;
    }

    h3, h4 {
        color: var(--primary-dark) !important;
        font-weight: 600 !important;
    }

    hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, var(--primary) 0%, var(--border) 100%);
        border-radius: 4px;
        margin: 1.2em 0 !important;
    }

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: var(--surface);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    /* Judul "Navigasi" pakai aksen lebih tipis & renggang, beda dari judul utama */
    [data-testid="stSidebarUserContent"] h1 {
        border-left: 4px solid var(--primary);
        padding-left: 12px;
        margin-left: 2px;
        margin-top: 0.6em !important;
        margin-bottom: 0.6em !important;
        font-size: 1.5rem !important;
        color: var(--primary-dark) !important;
    }

    /* Radio button navigasi: indikator garis kiri, bukan kotak penuh */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background-color: #ffffff;
        border: 1px solid var(--border);
        border-left: 4px solid transparent;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
        width: 100%;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        border-color: var(--primary);
        background-color: var(--primary-soft);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background-color: var(--primary-soft);
        border-left: 4px solid var(--primary);
        border-top-color: var(--border);
        border-right-color: var(--border);
        border-bottom-color: var(--border);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
        color: var(--primary-dark) !important;
        font-weight: 600;
    }

    /* --- METRIC CARDS --- */
    [data-testid="stMetric"] {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-left: 5px solid var(--primary);
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 2px 6px rgba(66, 61, 56, 0.06);
    }
    [data-testid="stMetricValue"] {
        color: var(--primary-dark) !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        white-space: normal !important;
        overflow-wrap: break-word;
        line-height: 1.3 !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-soft) !important;
        font-weight: 500 !important;
    }

    /* --- TABEL / DATAFRAME --- */
    .stDataFrame {
        border: 1px solid var(--border) !important;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(66, 61, 56, 0.05);
    }

    /* --- ALERT BOXES (info / success) --- */
    div[data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid var(--border);
    }
    div[data-baseweb="notification"] {
        background-color: var(--primary-soft) !important;
        border-left: 5px solid var(--primary) !important;
    }
    div[data-testid="stAlertContentSuccess"] {
        color: var(--text) !important;
    }

    /* --- SELECTBOX --- */
    [data-baseweb="select"] > div {
        border-color: var(--border) !important;
        border-radius: 8px !important;
    }
    [data-baseweb="select"] > div:hover {
        border-color: var(--primary) !important;
    }

    /* --- GAMBAR (logo, banner) --- */
    [data-testid="stImage"] img {
        border-radius: 12px;
    }

    /* --- LINK --- */
    a {
        color: var(--primary-dark) !important;
        font-weight: 600;
        text-decoration: none;
    }
    a:hover {
        color: var(--primary) !important;
        text-decoration: underline;
    }

    [data-testid="column"] > div > div[data-testid="stVerticalBlock"] {
        gap: 0.4rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Daftar kolom fitur asli (Sesuai model saat training)
fitur_ekonomi = [
    'Jumlah Penduduk', 'Kepadatan Penduduk', 'Sarana Pendidikan', 
    'Sarana Kesehatan', 'Transportasi', 'Sarana Perdagangan dan Jasa', 
    'Keberadaan Pasar dan Pertokoan', 'Bank dan Koperasi', 'IKM dan Sentra'
]

# Palet warna kategori (konsisten dengan design system)
COLOR_MAP = {
    'Potensi Tinggi': '#2f9e44',
    'Potensi Menengah': '#3b6ea5',
    'Potensi Rendah': '#d64545'
}

# --- 2. LOAD DATA & MODEL ---
@st.cache_resource
def load_essentials():
    df = pd.read_csv("Dataset/Dataset Potensi Ekonomi Kecamatan di Kota Palembang 2025.csv")
    model = joblib.load('Model/model_kmeans_potensiekonomi.pkl')
    scaler = joblib.load('Model/scaler_potensiekonomi.pkl')
    return df, model, scaler

try:
    df_raw, kmeans, scaler = load_essentials()
except Exception as e:
    st.error(f"Gagal memuat file. Error: {e}")
    st.stop()

# --- 3. PROSES KLASTERING ---
X = df_raw[fitur_ekonomi].copy()

X['Jumlah Penduduk'] = (X['Jumlah Penduduk'] * 1000).astype(int)
X['Kepadatan Penduduk'] = (X['Kepadatan Penduduk'] * 1000).astype(int)

X_scaled = scaler.transform(X)

df_raw['Cluster'] = kmeans.predict(X_scaled)

ranking = df_raw.groupby('Cluster')[fitur_ekonomi].mean(numeric_only=True).sum(axis=1).sort_values().index
mapping = {ranking[0]: 'Potensi Rendah', ranking[1]: 'Potensi Tinggi', ranking[2]: 'Potensi Menengah'}
df_raw['Kategori'] = df_raw['Cluster'].map(mapping)

# --- 4. SIDEBAR ---
st.sidebar.image("Images/logo_bps.png", width=80)
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["🏠 Beranda & Dataset", "📊 Analisis Klasterisasi", "👥 Profil Kelompok"])

st.sidebar.divider()
st.sidebar.caption("Project Magang BPS Kota Palembang")

# --- 5. HALAMAN 1: BERANDA & DATASET ---
if menu == "🏠 Beranda & Dataset":
    st.title("🏙️ Potensi Sosial-Ekonomi Kecamatan di Kota Palembang 2025")
    st.markdown("---")
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("📌 Informasi Project")
        st.write("""
        Project ini bertujuan untuk memetakan kekuatan sosial-ekonomi wilayah di Kota Palembang menggunakan 
        pendekatan Machine Learning (**K-Means Clustering**). Analisis ini mengelompokkan kecamatan 
        ke dalam tingkatan potensi sosial-ekonomi untuk membantu perencanaan pembangunan daerah.
        """)
        st.info("""
        **Sumber Data Resmi:**
        - BPS Kota Palembang: *Kota Palembang Dalam Angka 2025*
        - BPS Kota Palembang: *Statistik Potensi Desa (Podes) Kota Palembang 2025*
        """)
    
    with col_b:
        st.subheader("📈 Statistik Data")
        st.metric("Total Wilayah", f"{len(df_raw)} Kecamatan")
        st.metric("Jumlah Indikator", f"{len(fitur_ekonomi)} Kolom")
        st.metric("Jumlah Klaster (K)", "3 Kategori")

    st.divider()

    st.subheader("📋 Fitur (Atribut) Dataset")
    st.markdown("""
    | Fitur | Keterangan |
    | :--- | :--- |
    | **Kecamatan** | Nama kecamatan di Kota Palembang yang menjadi unit analisis |
    | **Jumlah Penduduk** | Jumlah penduduk yang berdomisili di masing-masing kecamatan |
    | **Kepadatan Penduduk** | Kepadatan penduduk per km² di setiap kecamatan |
    | **Sarana Pendidikan** | Jumlah seluruh fasilitas pendidikan (TK, SD, SMP, SMA/SMK, Perguruan Tinggi) |
    | **Sarana Kesehatan** | Jumlah seluruh fasilitas kesehatan (RS, Puskesmas, Klinik, Apotek, dll) |
    | **Sarana Perdagangan & Jasa** | Jumlah fasilitas perdagangan modern (Minimarket, Restoran, Hotel, dll) |
    | **Pasar dan Pertokoan** | Jumlah pasar tradisional dan kelompok pertokoan yang tersedia |
    | **Transportasi** | Jumlah desa/kelurahan yang memiliki akses angkutan umum dan online |
    | **Bank dan Koperasi** | Jumlah lembaga keuangan berupa bank dan koperasi |
    | **IMK dan Sentra** | Jumlah industri mikro dan kecil serta sentra industri |
    """)

    st.divider()
    
    st.write("### 📄 Dataset Utama")
    st.dataframe(df_raw[['Kecamatan'] + fitur_ekonomi], use_container_width=True)

    st.write("---")
    st.write("### 🔗 Akses Project & Dokumentasi")
    
    col_link1, col_link2 = st.columns(2)
    with col_link1:
        st.markdown("[📁 Repository GitHub Project](https://github.com/nshamid/Klasterisasi-Potensi-Ekonomi-Kecamatan/tree/main)")
    
    with col_link2:
        st.markdown("[📓 Notebook Google Colab](https://colab.research.google.com/drive/1UgG0mtfn3TkSO6zTTJIJ7U2AK2CwWvdU?usp=sharing)")

# --- 6. HALAMAN 2: ANALISIS KLASTERISASI ---
elif menu == "📊 Analisis Klasterisasi":
    st.title("📊 Hasil Analisis Klasterisasi")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("#### 📍 Peta Sebaran Klaster (PCA 2D)")
        pca = PCA(n_components=2)
        pca_res = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(pca_res, columns=['PC1', 'PC2'])
        df_pca['Kecamatan'] = df_raw['Kecamatan']
        df_pca['Kategori'] = df_raw['Kategori']

        fig_pca = px.scatter(
            df_pca, x='PC1', y='PC2', color='Kategori',
            hover_name='Kecamatan', text='Kecamatan',
            color_discrete_map=COLOR_MAP,
            template="plotly_white"
        )
        fig_pca.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
        fig_pca.update_layout(
            plot_bgcolor='#fcfaf7', paper_bgcolor='#fcfaf7',
            font=dict(color='#423d38'),
            legend=dict(bgcolor='#f3f4f6', bordercolor='#e7e2da', borderwidth=1)
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with col2:
        st.write("#### 🥧 Proporsi Kategori")
        count_data = df_raw['Kategori'].value_counts().reset_index()
        fig_pie = px.pie(
            count_data, names='Kategori', values='count',
            color='Kategori',
            color_discrete_map=COLOR_MAP,
            hole=0.4
        )
        fig_pie.update_layout(
            plot_bgcolor='#fcfaf7', paper_bgcolor='#fcfaf7',
            font=dict(color='#423d38'),
            legend=dict(bgcolor='#f3f4f6', bordercolor='#e7e2da', borderwidth=1)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    st.write("### 📋 Pembagian Wilayah per Kategori")
    
    cat_cols = st.columns(3)
    kategori_list = ['Potensi Tinggi', 'Potensi Menengah', 'Potensi Rendah']

    for i, kat in enumerate(kategori_list):
        with cat_cols[i]:
            warna = COLOR_MAP[kat]
            st.markdown(
                f"""
                <div style="background-color:{warna}18; border-left:5px solid {warna};
                            border-radius:10px; padding:10px 14px; margin-bottom:10px;">
                    <span style="color:{warna}; font-weight:700; font-family:'Poppins',sans-serif;">{kat}</span>
                </div>
                """, unsafe_allow_html=True
            )
            list_kecamatan = df_raw[df_raw['Kategori'] == kat]['Kecamatan'].values
            if len(list_kecamatan) > 0:
                for kec in list_kecamatan:
                    st.write(f"- {kec}")
            else:
                st.write("*Tidak ada data*")

    st.divider()

    st.write("### 📈 Karakteristik Indikator Per Kategori")
    feature = st.selectbox("Pilih Indikator untuk Melihat Perbandingan:", fitur_ekonomi)
    
    df_avg = df_raw.groupby('Kategori')[fitur_ekonomi].mean(numeric_only=True).reset_index()
    fig_bar = px.bar(
        df_avg, x='Kategori', y=feature, color='Kategori',
        text_auto='.2f',
        title=f"Rata-rata {feature} per Kategori",
        color_discrete_map=COLOR_MAP
    )
    fig_bar.update_layout(
        plot_bgcolor='#fcfaf7', paper_bgcolor='#fcfaf7',
        font=dict(color='#423d38'),
        title_font=dict(color='#423d38', family='Poppins')
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# --- 7. HALAMAN 3: PROFIL KELOMPOK (CREDIT) ---
elif menu == "👥 Profil Kelompok":
    st.title("👥 Profil Kelompok Kerja Praktik")
    st.markdown("---")

    st.image("Images/banner_kelompok.jpg", 
             caption="Dokumentasi Bersama Bapak Edi Subeno, S.E., M.Si. Kepala BPS Kota Palembang", 
             use_container_width=True)
    
    st.divider()

    col_l1, col_l2, col_l3 = st.columns([1, 1, 1])
    with col_l2:
        st.image("Images/logo_unsri.png", width=200)
    
    st.markdown("<h2 style='text-align: center;'>Teknik Informatika Bilingual 2023</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Fakultas Ilmu Komputer, Universitas Sriwijaya</h3>", unsafe_allow_html=True)
    
    st.divider()

    col_info, col_anggota = st.columns(2)

    with col_info:
        st.write("### 📖 Informasi Akademik")
        st.markdown("""
        - **Mata Kuliah:** Kerja Praktik (FTI4001)
        - **Dosen Pengampu:** Yunita, S.Si., M.Cs.
        - **Dosen Pembimbing Lapangan:** Aharmisa Rahmatullah, S.ST
        """)
        
    with col_anggota:
        st.write("### 👩‍🎓 Anggota Kelompok")
        st.markdown("""
        1. **Nabilah Shamid** (09021382328147)
        2. **Saravina Zharfa Kelana P** (09021382328149)
        3. **Raka Athallah Ananta** (09021382328163)
        """)

    st.divider()
    st.success("Terima Kasih kepada BPS Kota Palembang atas bimbingan dan kesempatan yang diberikan.")
