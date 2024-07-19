# Hybrid GARCH-LSTM

## Pendahuluan
<p align="justify">
Analisis deret waktu digunakan untuk menghasilkan ramalan terhadap kejadian yang akan datang dengan mengasumsikan data dalam kondisi stasioner dan linear. Namun, dalam kehidupan nyata, data sering kali bersifat nonstasioner, nonlinear, dan memiliki volatilitas tinggi. Untuk mengatasi permasalahan tersebut, dikembangkanlah suatu metode dengan mengombinasikan metode statistika klasik dan metode deep learning, yaitu Generalized Autoregressive Conditional Heteroskedasticity (GARCH) dengan Long Short-Term Memory (LSTM), sebagai model hybrid GARCH-LSTM. Metode ini dapat mempertahankan kemampuan model GARCH dalam menginterpretasikan kondisi fluktuatif dan sekaligus meningkatkan akurasi ramalan dengan memodelkan sisaannya menggunakan model LSTM.
</p>

## Prosedur Pemodelan
<p align="justify">
Dalam metode hybrid GARCH-LSTM, model GARCH digunakan untuk menangkap pola linear dalam data. Sementara itu, penggunaan model LSTM ditujukan untuk menangkap pola nonlinear yang tidak bisa dijelaskan oleh model GARCH. Pemodelan hybrid GARCH-LSTM dimulai dengan melakukan pendugaan model terbaik dengan model GARCH. Sisaan dari model GARCH kemudian dijadikan data input dalam prosedur pemodelan LSTM. Oleh karena itu, data input untuk model LSTM dapat diperoleh melalui persamaan berikut:
</p>

$$
e_t = y_t - \hat{y}_t
$$

<p align="justify">
dengan $e_t$ adalah sisaan pada waktu ke-𝑡, $y_t$ adalah nilai aktual pada waktu ke-𝑡, dan $\hat{y}_t$ adalah dugaan nilai pada waktu ke-𝑡 berdasarkan model GARCH.
</p>

<p align="justify">
Setelah prosedur pemodelan LSTM dilakukan, nilai dugaan final model hybrid GARCH-LSTM dapat diperoleh dengan menjumlahkan dugaan berdasarkan model GARCH dengan dugaan sisaan berdasarkan model LSTM. Perhitungan tersebut dapat diperoleh berdasarkan persamaan berikut:
</p>

$$
\hat{y_h}_t = \hat{y}_t + \hat{e}_t
$$

<p align="justify">
dengan $\hat{y_h}_t$ adalah total dugaan nilai model hybrid GARCH-LSTM pada waktu ke-𝑡, $\hat{y}_t$ adalah dugaan nilai pada waktu ke-𝑡 berdasarkan model GARCH, dan $\hat{e}_t$ adalah dugaan sisaan pada waktu ke-𝑡 berdasarkan model LSTM.
</p>
