# Sınıflandırma Modelleri
## KNN Algoritması
En yakın komşuluk algoritması olarak geçen bu algoritma. verinin en yakın komşularına bakarak veriyi sınıflandırma mantığına dayanmaktadır. Bu sınıflandırma işleminde verinin komşularına olan mesafeleri kontrol edilmektedir. Yani yeni verinin en yakın olduğu eleman hangi veri sınıfına dahil olduğuna bakılırak o veri sınıfına eklenmektedir.
Bazen sadece en yakındaki veriyi kontrol etmek yetmeyebilir. Örneğin mavi dairenin en yakınında yeşil daire var ancak yakınlarda çok sayıda da sarı daire vardır. Böyle bir durumda o bölgede sarı daireler daha baskın olmaktadır. Bu sefer “k” yani en yakındaki kaç verinin kontrol edileceğine bakılarak sınıflandırma yapılır. Örneğin k=5 olsun, verinin en yakınındaki 5 veri kontrol edilerek hangi veri çoğunluktaysa o sınıfa dahil edilmektedir (5 rengin 3'ü sarı 2'si yeşil ,sarı gruba dahil edilir.).

## SVM-SVR Destek Vektör Makinesi
Bir veri seti içerinde bulunan iki farklı sınıftan birine ait bir dizi veriler olduğunu düşünelim. SVM algoritması bu iki farklı sınıfa ait olan elemanları bir doğru çizerek birbirinden ayrılmaktadır. Bu ayırma işlemini sınırdaki elemanlara göre yapmaktadır. Bu çizilen doğruya hiper düzlem adı verilmektedir. Bu algoritmadaki en önemli nokta hiper düzlemi en kusursuz şekilde belirleyebilmektir. Detaylı çalışma için örnekleri inceleyebilirsiniz.
