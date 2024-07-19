# Analisis Time Series Menggunakan Metode Hybrid GARCH-LSTM

<p align="justify">
Analisis deret waktu digunakan untuk menghasilkan ramalan terhadap kejadian yang akan datang dengan mengasumsikan data dalam kondisi stasioner dan linear. Namun, dalam kehidupan nyata, data sering kali bersifat nonstasioner, nonlinear, dan memiliki volatilitas tinggi. Untuk mengatasi permasalahan tersebut, dikembangkanlah suatu metode dengan mengombinasikan metode statistika klasik dan metode deep learning, yaitu *Generalized Autoregressive Conditional Heteroskedasticity* (GARCH) dengan *Long Short-Term Memory* (LSTM), sebagai model *_hybrid_ GARCH-LSTM*. Metode ini dapat mempertahankan kemampuan model GARCH dalam menginterpretasikan kondisi fluktuatif dan sekaligus meningkatkan akurasi ramalan dengan memodelkan sisaannya menggunakan model LSTM.
  
<p align="justify">
Dalam metode *_hybrid_ GARCH-LSTM*, model GARCH digunakan untuk menangkap pola linear dalam data. Sementara itu, penggunaan model LSTM ditujukan untuk menangkap pola nonlinear yang tidak bisa dijelaskan oleh model GARCH. Pemodelan *_hybrid_ GARCH-LSTM* dimulai dengan melakukan pendugaan model terbaik dengan model GARCH. Sisaan dari model GARCH kemudian dijadikan data input dalam prosedur pemodelan LSTM.
</p>
