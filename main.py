# Libraries
import pandas as pd
import numpy as np
import streamlit as st
import time
import plotly.figure_factory as ff
import plotly.express as px
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error, davies_bouldin_score


# Import functions
import functions_ as dp

# Main function
def main():
    # global list_p, df_list

    # Set default list of perusahaan
    st.session_state.input_file_paths = dp.get_list_path()
    st.session_state.df_list = dp.get_df_list(st.session_state.input_file_paths)
    st.session_state.list_p = dp.get_list_perusahaan(st.session_state.df_list)

    st.title(':blue[CUSTHEALTH]')
    st.header('_:blue[Aplikasi Cluster Prediksi Asuransi]_')
    st.write("""
             Aplikasi ini digunakan untuk memprediksi cluster perusahaan asuransi,
             kemudian melakukan regresi pendapatan rumah sakit dari perusahaan asuransi.
             """)
    
    # Session handling
    if 'input_file_paths' not in st.session_state:
        st.session_state.input_file_paths = []

    if 'year' not in st.session_state:
        st.session_state.year = ["2018", "2019", "2020", "2021", "2022"]

    if 'df_list' not in st.session_state:
        st.session_state.df_list = []

    if 'list_p' not in st.session_state:
        st.session_state.list_p = []

    if 'table_sum' not in st.session_state:
        st.session_state.table_sum = []

    if 'dtf_all_cat_wo_s' not in st.session_state:
        st.session_state.dtf_all_cat_wo_s = []

    if 'butt1' not in st.session_state:
        st.session_state.butt1 = 0

    if 'mySeries2' not in st.session_state:
        st.session_state.mySeries2 = []

    if 'namesofMySeries2' not in st.session_state:
        st.session_state.namesofMySeries2 = []

    if 'model_kmeans_tslearn' not in st.session_state:
        st.session_state.model_kmeans_tslearn = []

    if 'model_kmeans_sklearn' not in st.session_state:
        st.session_state.model_kmeans_sklearn = []

    if 'cluster_count' not in st.session_state:
        st.session_state.cluster_count = 0
    
    if 'butt2' not in st.session_state:
        st.session_state.butt2 = 0

    if 'butt21' not in st.session_state:
        st.session_state.butt21 = 0

    if 'table_sum_timeseries' not in st.session_state:
        st.session_state.table_sum_timeseries = []
    
    if 'df_n' not in st.session_state:
        st.session_state.df_n = pd.DataFrame()

    if 'opt_cat' not in st.session_state:
        st.session_state.opt_cat = ""

    if 'num_lag' not in st.session_state:
        st.session_state.num_lag = 0

    if 'select3' not in st.session_state:
        st.session_state.select3 = '0'
    
    if 'select4' not in st.session_state:
        st.session_state.select4 = ''

    if 'regressor' not in st.session_state:
        st.session_state.regressor = 0

    if 'transform_train' not in st.session_state:
        st.session_state.transform_train = 0
    
    if 'transform_test' not in st.session_state:
        st.session_state.transform_test = 0

    if 'scaler' not in st.session_state:
        st.session_state.scaler = []

    if 'pred_result' not in st.session_state:
        st.session_state.pred_result = []

    if 'dff' not in st.session_state:
        st.session_state.dff = {}

    if 'y_pred' not in st.session_state:
        st.session_state.y_pred = 0

    
    # Sidebar Initiation
    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox("Home:", ["Abstrak", "Riwayat Pembuat"])
    menu1 = st.sidebar.selectbox("Analisis dan Pemodelan:", ["Default", "Cluster", "Timeseries"])
    menu2 = st.sidebar.selectbox("Hasil Prediksi", ["Default", "Golongan Perusahaan Asuransi", "Pendapatan dari Perusahaan Asuransi"])
    
    # Abstrak
    if menu == "Abstrak" and menu1 == "Default" and menu2 == "Default":
        st.subheader("ABSTRAK")
        st.write("""
                abstrak isi abstrak isi abstrak isi abstrak isi abstrak isi
                 abstrak isi abstrak isi abstrak isi abstrak isi abstrak isi 
                 abstrak isi abstrak isi abstrak isi abstrak isi abstrak isi
                 """)
    
    # Riwayat Pembuat
    if menu == "Riwayat Pembuat" and menu1 == "Default" and menu2 == "Default":
        st.subheader("RIWAYAT PEMBUAT")
        st.write("""
                Riwayat Pembuat Riwayat Pembuat Riwayat Pembuat Riwayat Pembuat
                 Riwayat Pembuat Riwayat Pembuat Riwayat Pembuat Riwayat Pembuat
                 Riwayat Pembuat Riwayat Pembuat Riwayat Pembuat Riwayat Pembuat
                 """)
    
    # Analisis dan Pemodelan Cluster
    if menu1 == "Cluster" and menu2 == "Default":

        st.subheader("Analisis dan Pemodelan _Cluster_ Perusahaan Asuransi")

        # Get existing table_sum
        if len(st.session_state.table_sum) == 0:

            _cat = ["Emergency", 
                        "Health Checkup", 
                        "Inpatient", 
                        "Outpatient", 
                        "Total"]
            
            st.write("Table Sum belum Tersedia. Lakukan Import!")
            butt1 = st.button("Import Table Sum")
            if butt1:
                # Create Dataframe
                st.session_state.table_sum = dp.get_existing_data_ts(st.session_state.df_list, st.session_state.list_p)

                for cat in _cat:
                    st.session_state.dtf_all_cat_wo_s.append(dp.table_sum_big_data(st.session_state.table_sum, cat))
                    st.session_state.mySeries2.append(dp.table_sum_big_data2(st.session_state.table_sum, cat)[1])
                st.write("Table Sum Sudah Diimport dan Diubah Menjadi Dataframe Untuk Semua Kategori. Pilihan Tahapan Selanjutnya")
        
        # Go Analytics and Modelling
        if len(st.session_state.table_sum) > 0:
            select1 = st.selectbox("Pilih Tahapan Selanjutnya:", 
                                   ["Default", 
                                    "Analisis", 
                                    "Pemodelan"])
            
            if select1 == "Analisis" and len(st.session_state.dtf_all_cat_wo_s) > 0:
                opt_pt = st.selectbox("Pilih Perusahaan:", st.session_state.list_p)
                _cat = ["Emergency", 
                        "Health Checkup", 
                        "Inpatient", 
                        "Outpatient", 
                        "Total"]
                
                opt_cat = st.selectbox("Pilih Category Layanan:", _cat)
                
                _df_ = st.session_state.dtf_all_cat_wo_s[_cat.index(opt_cat)][2]
                year = ["2018", "2019", "2020", "2021", "2022"]

                # Create scatter plot
                fig = px.scatter(_df_,
                                 x = _df_.index,
                                 y = opt_pt,
                                 title = f"Kategori: {opt_cat}, {opt_pt}")
                
                # Plot
                st.plotly_chart(fig, use_container_width=True)

            if select1 == "Pemodelan":

                _cat = ["Emergency", 
                        "Health Checkup", 
                        "Inpatient", 
                        "Outpatient", 
                        "Total"]
                
                opt_model = st.selectbox("Pilih Model:", ["Default", "KMeans - sklearn"])

                st.session_state.opt_cat = st.selectbox("Pilih Category Layanan:", 
                                       ["Emergency", 
                                        "Health Checkup", 
                                        "Inpatient", 
                                        "Outpatient", 
                                        "Total"])
                
                st.session_state.namesofMySeries2 = st.session_state.dtf_all_cat_wo_s[_cat.index(st.session_state.opt_cat)][0]

                st.session_state.cluster_count = st.number_input("Masukkan Jumlah Cluster", 
                                                                 min_value=3, 
                                                                 max_value=10, 
                                                                 step=1)
                
                butt2 = st.button("Training Cluster")
                if butt2:
                    st.session_state.butt2 = 1

                if st.session_state.butt2 > 0 and opt_model == "KMeans - sklearn" and st.session_state.cluster_count > 0:
                    dp.training_kmeans_sklearn(
                            st.session_state.cluster_count, 
                            st.session_state.mySeries2[_cat.index(st.session_state.opt_cat)]
                    )
                    st.success(f"Training {opt_model} - {st.session_state.opt_cat} Selesai")
                    st.session_state.butt2 = 0
                
                butt3 = st.button("Plot Cluster Kmeans")                  
                if butt3 and opt_model == "KMeans - sklearn":
                    mySeries_transformed, labels, centroid, score = dp.training_kmeans_sklearn(st.session_state.cluster_count, 
                                                                           st.session_state.mySeries2[_cat.index(st.session_state.opt_cat)])
                    
                    df__ = st.session_state.dtf_all_cat_wo_s[_cat.index(st.session_state.opt_cat)][2]
                    st.session_state.df_n = dp.df_clust(labels, st.session_state.namesofMySeries2, df__)

                    dp.plot_cluster_kmeans_sklearn(centroid, df__,
                                                   st.session_state.namesofMySeries2, 
                                                   st.session_state.cluster_count, 
                                                   st.session_state.mySeries2[_cat.index(st.session_state.opt_cat)], 
                                                   mySeries_transformed, 
                                                   labels)


    # Analisis dan Pemodelan Timeseries
    if menu1 == "Timeseries" and menu2 == "Default":
        st.subheader(f"Analisis dan Pemodelan _Timeseries_ Perusahaan Asuransi Kategori - {st.session_state.opt_cat}")

        if len(st.session_state.table_sum) > 0:
            st.session_state.table_sum_timeseries = dp.table_sum_ts(
                st.session_state.cluster_count, 
                st.session_state.df_n, 
                st.session_state.table_sum
                )
            
            select2 = st.selectbox("Pilih Tahapan:", 
            ["Default",
            "Analisis",
            "Pemodelan Regresi Timeseries - Linear", 
            # "Pemodelan Regresi Timeseries - RNN"
            ])

            if select2 == "Analisis":
                st.session_state.select3 = st.selectbox("Pilih Cluster:", [str(item) for item in range(st.session_state.cluster_count)])
                clustt = st.session_state.table_sum_timeseries[int(st.session_state.select3)].keys()
                if len(clustt) == 0:
                    st.write(f"Cluster {int(st.session_state.select3)} Berjumlah 0, Pilih Cluster Lain.")
                else:
                    st.write(f"Cluster {int(st.session_state.select3)} Berjumlah {len(clustt)} Perusahaan Asuransi.")
                    st.session_state.select4 = st.selectbox("Pilih Perusahaan:", st.session_state.table_sum_timeseries[int(st.session_state.select3)].keys())

                    df_io = st.session_state.table_sum_timeseries[int(st.session_state.select3)][st.session_state.select4]

                    # Create scatter plot
                    fig = px.scatter(df_io,
                                 x = st.session_state.year,
                                 y = st.session_state.opt_cat,
                                 title = f"Pendapatan Dari {st.session_state.select4} - Cluster {int(st.session_state.select3)}")

                    # Plot
                    st.plotly_chart(fig, use_container_width=True)
            if select2 == "Pemodelan Regresi Timeseries - Linear":
                st.session_state.num_lag = st.number_input("Masukkan Jumlah Lag", 
                                                                        min_value = 1,
                                                                        max_value = 4,
                                                                        step = 1)
                st.write(f"Jumlah Lag - {st.session_state.num_lag}")

                if st.session_state.num_lag > 0:
                    st.session_state.dff = dp.new_df_clust(
                        st.session_state.table_sum_timeseries[int(st.session_state.select3)], 
                        int(st.session_state.num_lag), 
                        st.session_state.opt_cat
                        )
                

                    st.write(f"Contoh Pembagian Dataset Perusahaan {st.session_state.select4} - Cluster {st.session_state.select3}")

                    st.dataframe(st.session_state.dff[st.session_state.select4])

                    st.write(f"Plot Training Testing Perusahaan {st.session_state.select4} - Cluster {st.session_state.select3}")

                    dp.plot_ts_traintest(st.session_state.dff[st.session_state.select4], 
                    st.session_state.num_lag, 
                    st.session_state.opt_cat, 
                    )

                    st.write(f"Hasil Training Perusahaan {st.session_state.select4} - Cluster {st.session_state.select3}")
                    st.session_state.y_pred = dp.linear_regression(True, st.session_state.dff[st.session_state.select4], 
                    st.session_state.num_lag, 
                    st.session_state.opt_cat,
                    st.session_state.year)
 
                

        if len(st.session_state.table_sum) == 0:
            st.write("Tabel Belum Ada. Mohon Import Terlebih Dulu di Menu: _Analisis dan Pemodelan --> Cluster_")

    # Prediksi Golongan Perusahaan Asuransi
    if menu2 == "Golongan Perusahaan Asuransi" and menu1 == "Default":
        if st.session_state.opt_cat == "": 
            st.write("Kategori Asuransi Belum Dipilih. Mohon Training KMeans Berdasarkan Kategori yang Diinginkan.")

        if len(st.session_state.opt_cat) > 0:
            st.subheader(f"Prediksi Golongan Perusahaan Asuransi - {st.session_state.opt_cat}")
            select5 = st.selectbox("Pilih Perusahaan Asuransi:", st.session_state.list_p)
            r = st.session_state.df_n['Cluster'][select5]
            st.write(f"Perusahaan {select5} Masuk ke Dalam {r}")
    
    # Prediksi Pendapatan dari Perusahaan Asuransi
    if menu2 == "Pendapatan dari Perusahaan Asuransi" and menu1 == "Default":
        st.subheader(f"Prediksi Pendapatan dari Perusahaan Asuransi {st.session_state.select4}")
        # select6 = st.selectbox("Pilih Perusahaan Asuransi:", st.session_state.table_sum_timeseries[int(st.session_state.select3)].keys())
        tahun_p = st.number_input("Masukkan Tahun Pendapatan Yang Ingin Diprediksi: ", min_value = 2023, step = 1)
        beda_tahun = int(tahun_p) - 2022

        hasil_prediksi = []
        tahun_prediksi = []
        y_pred = st.session_state.y_pred
        for i in range(beda_tahun):
            hasil_prediksi.append(y_pred[-1])
            tahun_prediksi.append(str(2023 + i))
            st.write(f"Prediksi Pendapatan Tahun - {2023 + i} Perusahaan {st.session_state.select4} - Kategori {st.session_state.opt_cat} - {st.session_state.df_n['Cluster'][st.session_state.select4]} Sebesar Rp{int(y_pred[-1])}")

            df = {}
            df[f"Revenue_{st.session_state.opt_cat}"] = y_pred
            df[f"Lag_{st.session_state.num_lag}"] = pd.DataFrame(y_pred)[0].shift(st.session_state.num_lag)

            y_pred = dp.linear_regression(False, df, 
            st.session_state.num_lag, 
            st.session_state.opt_cat,
            st.session_state.year)

        tahun_prediksi = st.session_state.year + tahun_prediksi
        nilai_sebenarnya = st.session_state.dff[st.session_state.select4][f"Revenue_{st.session_state.opt_cat}"].to_list() + [0] * len(hasil_prediksi)
        hasil_prediksi = st.session_state.dff[st.session_state.select4][f"Revenue_{st.session_state.opt_cat}"].to_list() + hasil_prediksi

        df_r = pd.DataFrame({
            "Tahun Prediksi": tahun_prediksi,
            "Hasil Prediksi": hasil_prediksi,
            "Nilai Sebenarnya": nilai_sebenarnya
        })
        # st.dataframe(st.session_state.dff[st.session_state.select4])
        # st.dataframe(df_r)

        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
    
        csv = convert_df(df_r)

        st.dataframe(df_r)
        # st.dataframe(st.session_state.dff[st.session_state.select4][f"Revenue_{st.session_state.opt_cat}"])

        st.download_button(
            "Press to Download",
            csv,
            f"File - Perusahaan {st.session_state.select4} - Kategori {st.session_state.opt_cat} - {st.session_state.df_n['Cluster'][st.session_state.select4]}.csv",
            "text/csv",
            key='download-csv'
    )

            






# Running program
if __name__ == "__main__":
    main()