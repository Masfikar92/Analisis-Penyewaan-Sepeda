import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "main_data.csv"))

df['atemp'] = pd.cut(df['atemp'] * 50,
    bins=[0, 10, 20, 30, 40, 50],
    labels=['Sangat Dingin', 'Dingin', 'Hangat', 'Panas', 'Sangat Panas']
)

st.title("Dashboard Penyewaan Sepeda")
st.divider()

st.subheader("penyewaan sepeda perbulan paling tinggi dan paling rendah")

order_bulan = ['Januari','Februari','Maret','April','Mei','Juni',
               'Juli','Agustus','September','Oktober','November','Desember']

monthly = df.groupby('mnth')['cnt'].mean().reindex(order_bulan)

fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(x='cnt', y='mnth', data=df, order=order_bulan, ax=ax1)
ax1.set_title('Jumlah Penyewaan Sepeda per Bulan')
ax1.set_xlabel('Jumlah Penyewaan Sepeda')
ax1.set_ylabel('Bulan')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig1)
plt.close()

st.info("Penyewaan tertinggi:Bulan September dan Penyewaan Terendah Bulan Januari")

st.divider()

st.subheader("Pengaruh suhu yang dirasakan tubuh terhadap penyewaan sepeda")

atemp_order = ['Sangat Dingin', 'Dingin', 'Hangat', 'Panas', 'Sangat Panas']
atemp_grouped = df.groupby('atemp', observed=True)['cnt'].mean().reindex(atemp_order)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.bar(atemp_grouped.index, atemp_grouped.values, color='#3498db')
ax2.set_title('Rata-rata Penyewaan Sepeda per Kategori Suhu yang Dirasakan')
ax2.set_xlabel('Kategori Suhu')
ax2.set_ylabel('Rata-rata Penyewaan')
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.info("Semakin hangat suhu yang dirasakan tubuh, semakin banyak sepeda yang disewa. Penyewaan terbanyak terjadi pada kategori Hangat dan Panas(30-40°C).")