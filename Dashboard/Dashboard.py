import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "main_data.csv"))

df['yr_label'] = df['yr'].map({0: '2011', 1: '2012'})

def classify_day(row):
    score = 0
    if 0.3 <= row['atemp'] <= 0.75:
        score += 1
    if row['hum'] < 0.8:
        score += 1
    if row['windspeed'] < 0.4:
        score += 1
    if score == 3:
        return 'Ideal'
    elif score == 2:
        return 'Cukup'
    else:
        return 'Tidak Ideal'

df['bike_condition'] = df.apply(classify_day, axis=1)

df['rental_segment'] = pd.cut(
    df['cnt'],
    bins=[0, 2000, 5500, 7500, 10000],
    labels=['Sepi', 'Normal', 'Ramai', 'Sangat Ramai']
)

st.title("Dashboard Penyewaan Sepeda")
st.markdown("Analisis data penyewaan sepeda periode **2011–2012**")
st.divider()

st.sidebar.header("Filter Data")

tahun_options = sorted(df['yr_label'].dropna().unique().tolist())
musim_options = [s for s in ['Semi', 'Panas', 'Gugur', 'Dingin']
                 if s in df['season'].unique()]

tahun_filter = st.sidebar.multiselect(
    "Pilih Tahun",
    options=tahun_options,
    default=tahun_options
)
musim_filter = st.sidebar.multiselect(
    "Pilih Musim",
    options=musim_options,
    default=musim_options
)

df_filtered = df[
    df['yr_label'].isin(tahun_filter) &
    df['season'].isin(musim_filter)
]

if df_filtered.empty:
    st.warning("Tidak ada data untuk filter yang dipilih. Silakan ubah filter.")
    st.stop()

st.subheader("Penyewaan Sepeda per Bulan (2011-2012)")
st.caption("Untuk merencanakan strategi promosi pada bulan dengan penyewaan rendah")

order_bulan = ['Januari','Februari','Maret','April','Mei','Juni',
               'Juli','Agustus','September','Oktober','November','Desember']

monthly_data = (
    df_filtered.groupby('mnth', observed=True)['cnt']
    .sum()
    .reindex(order_bulan)
)

fig1, ax1 = plt.subplots(figsize=(10, 5))
colors = ['#e74c3c' if v == monthly_data.max() or v == monthly_data.min()
          else '#3498db' for v in monthly_data.values]
ax1.barh(order_bulan, monthly_data.values, color=colors)
ax1.set_title('Total Penyewaan Sepeda per Bulan')
ax1.set_xlabel('Total Penyewaan')
ax1.set_ylabel('Bulan')
ax1.invert_yaxis()
plt.tight_layout()
st.pyplot(fig1)
plt.close()

monthly_clean = monthly_data.dropna()
if not monthly_clean.empty:
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Tertinggi:** {monthly_clean.idxmax()} ({int(monthly_clean.max()):,} penyewaan)")
    with col2:
        st.error(f"**Terendah:** {monthly_clean.idxmin()} ({int(monthly_clean.min()):,} penyewaan)")

st.divider()

st.subheader("Pengaruh Suhu terhadap Penyewaan (2011-2012)")
st.caption("Untuk menentukan strategi promosi dan operasional berdasarkan kondisi cuaca")

fig2, ax2 = plt.subplots(figsize=(9, 5))
sns.regplot(
    x=df_filtered['atemp'] * 50,
    y=df_filtered['cnt'],
    ax=ax2,
    color='steelblue',
    scatter_kws={'alpha': 0.4, 's': 30},
    line_kws={'color': 'red', 'linewidth': 2}
)
ax2.set_title('Pengaruh Suhu yang Dirasakan vs Jumlah Penyewaan')
ax2.set_xlabel('Suhu yang Dirasakan (°C)')
ax2.set_ylabel('Jumlah Penyewaan')
plt.tight_layout()
st.pyplot(fig2)
plt.close()

corr = df_filtered['atemp'].corr(df_filtered['cnt'])
kekuatan = 'positif kuat' if corr > 0.6 else ('positif sedang' if corr > 0.3 else 'lemah')
st.info(
    f"Korelasi suhu yang dirasakan (atemp) dengan jumlah penyewaan: "
    f"**{corr:.2f}** ({kekuatan}) — semakin hangat suhu, semakin banyak penyewaan."
)

st.divider()

st.subheader("Analisis Lanjutan: Clustering dengan Binning")
st.markdown("Pengelompokan hari berdasarkan kondisi cuaca dan volume penyewaan untuk rekomendasi operasional bisnis.")

tab1, tab2 = st.tabs(["Clustering 1: Kelayakan Bersepeda", "Clustering 2: Segmentasi Penyewaan"])

with tab1:
    st.markdown(
        "Setiap hari dikelompokkan ke **Ideal / Cukup / Tidak Ideal** berdasarkan "
        "kombinasi suhu (`atemp`), kelembapan (`hum`), dan kecepatan angin (`windspeed`)."
    )

    condition_order = ['Ideal', 'Cukup', 'Tidak Ideal']
    colors_cond     = ['#2ecc71', '#f39c12', '#e74c3c']

    condition_counts = (
        df_filtered['bike_condition']
        .value_counts()
        .reindex(condition_order)
        .fillna(0)
    )
    avg_by_condition = (
        df_filtered.groupby('bike_condition', observed=True)['cnt']
        .mean()
        .reindex(condition_order)
    )

    fig3, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(condition_order, condition_counts.values, color=colors_cond, edgecolor='white')
    axes[0].set_title('Jumlah Hari per Kategori Kelayakan')
    axes[0].set_xlabel('Kategori')
    axes[0].set_ylabel('Jumlah Hari')
    for i, v in enumerate(condition_counts.values):
        axes[0].text(i, v + 2, str(int(v)), ha='center', fontweight='bold')

    axes[1].bar(condition_order, avg_by_condition.values, color=colors_cond, edgecolor='white')
    axes[1].set_title('Rata-rata Penyewaan per Kategori Kelayakan')
    axes[1].set_xlabel('Kategori')
    axes[1].set_ylabel('Rata-rata Penyewaan')
    for i, v in enumerate(avg_by_condition.values):
        if not pd.isna(v):
            axes[1].text(i, v + 30, f'{v:.0f}', ha='center', fontweight='bold')

    plt.suptitle('Clustering Kelayakan Hari Bersepeda', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Hari Ideal", f"{int(condition_counts.get('Ideal', 0))} hari")
    with col2:
        st.metric("Hari Cukup", f"{int(condition_counts.get('Cukup', 0))} hari")
    with col3:
        st.metric("Hari Tidak Ideal", f"{int(condition_counts.get('Tidak Ideal', 0))} hari")

    st.info(
        "💡 **Insight:** Hari dengan kondisi **Ideal** menghasilkan rata-rata penyewaan tertinggi. "
        "Kondisi cuaca (suhu, kelembapan, angin) secara bersama-sama menentukan minat masyarakat untuk bersepeda."
    )

with tab2:
    st.markdown(
        "Setiap hari dikelompokkan ke **Sepi / Normal / Ramai / Sangat Ramai** "
        "berdasarkan total penyewaan harian (`cnt`)."
    )

    order_seg  = ['Sepi', 'Normal', 'Ramai', 'Sangat Ramai']
    seg_colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

    segment_counts = (
        df_filtered['rental_segment']
        .value_counts()
        .reindex(order_seg)
        .fillna(0)
    )

    fig4, axes = plt.subplots(1, 2, figsize=(12, 4))

    non_zero = segment_counts[segment_counts > 0]
    axes[0].pie(
        non_zero.values,
        labels=non_zero.index,
        colors=[seg_colors[order_seg.index(s)] for s in non_zero.index],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    axes[0].set_title('Proporsi Hari per Segmen Penyewaan')

    season_seg = (
        df_filtered
        .groupby(['season', 'rental_segment'], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    season_seg   = season_seg.reindex(columns=order_seg, fill_value=0)
    season_order = [s for s in ['Semi', 'Panas', 'Gugur', 'Dingin'] if s in season_seg.index]
    season_seg   = season_seg.reindex(season_order)
    season_seg.plot(kind='bar', stacked=True, ax=axes[1], color=seg_colors, edgecolor='white')
    axes[1].set_title('Distribusi Segmen Penyewaan per Musim')
    axes[1].set_xlabel('Musim')
    axes[1].set_ylabel('Jumlah Hari')
    axes[1].legend(title='Segmen', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].tick_params(axis='x', rotation=0)

    plt.suptitle('Segmentasi Intensitas Penyewaan Harian', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Sepi", "Sepi"), ("Normal", "Normal"),
        ("Ramai", "Ramai"), ("Sangat Ramai", "Sangat Ramai")
    ]
    for col, (label, seg) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.metric(label, f"{int(segment_counts.get(seg, 0))} hari")

    st.info(
        "**Insight:** Segmen **Sangat Ramai** terkonsentrasi di musim Panas & Gugur — "
        "periode kritis untuk kesiapan armada. Segmen **Sepi** paling banyak di musim Dingin & Semi — "
        "waktu ideal untuk perawatan armada dan kampanye promosi."
    )
