import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_=pd.read_csv("C:\diabetes.csv")
df=df_.copy()

#############################################################
# GÖREV - 1
#############################################################

#ADIM-1 Genel veriseti incelemesi
df.shape
df.head(20)
df.describe().T

#ADIM-2 Numerik kategorik değişkenlerin yakanlaması

#Kategorik değişkenler, numerik değişkenler, kategorik görünüp kardinal olan değişkenler ve
#numerik görünüp kategorik olan değişkenleri  grap_col_name fonksiyonu ile yakalıyoruz
def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols'ta kolon tipi object olan değişkenleri kategorik değişkenler olarak atadık
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    #num_but_cat ile df içindeki kolonlarda tipi object olmayan ve eşsiz sınıf sayısı kullanıcının
     #verdiği cat_th değerinden az olan kolonlar olarak belirledik bu sayede numerik görünümlü
     #kategorikleri elde ettik
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    #cat_but_car ile df içindeki kolonlarda tipi object olup eşsiz sınıf sayısı kullanıcının verdiği
     #car_th değerinden büyük olan değişkenleri belieledik kategorik görünümlü kardinalleri elde ettik
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    #kategorik değişkenler içine numerik görünümlü kategorikleri ekledik
    cat_cols = cat_cols + num_but_cat

    #kategorik kolonlar içinden kategorik görünümlü kardinalleri çıkardık
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #numerik değişkenleri, kolon tipi object olmayan olarak tanımladık
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    #numerik değişkenler içinden numerik görümümlü kategorik değişkenleri çıkardık
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

#fonksiyonu çalıştırarak kategorik,numerik ve kardinal değişkenleri elde ettik
cat_cols, num_cols, cat_but_car= grab_col_names(df)
cat_cols

#ADIM-3 Numerik ve kategorik değişkenlerin analizinin yapılması

#cat_summary fonksiyonu ile kategorik değişkenler için sınıf sayılarını ve sınıf sayılarının oranlarını
#ayrıca istenirse eksik gözlemi ve grafiği gösteriyor
def cat_summary(dataframe, col_name, plot=False, null=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
    if null:
        print(col_name, "NaN number: ", dataframe[col_name].isnull().sum())

#Tek kategorik değişkenimiz olan Outcome için fonksiyonu uyguluyoruz
cat_summary(df,cat_cols[0],null=True,plot=False)

#num_summary fonksiyonu numerik sütunlar için çeyrek değerleri ve tanımlayıcı istatistikleri
# ayrıca istenirse box-plot grafiğini çizdiriyor
def num_summary(dataframe, numerical_col, boxplot=False):
    quantiles = [0.05, 0.10, 0.50, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if boxplot:
        sns.boxplot(x=dataframe[numerical_col])
        plt.xlabel(numerical_col)
        plt.show()

for i in num_cols:
    num_summary(df,i,boxplot=True)

#ADIM-4 Hedef değişken analizinin yapılması

#target_analyser fonksiyonu ile hedef değişken kırılımında kategorik değişkenler için sınıf sayılarını,
# sınıf sayılarının oranları ve ategorik değişkenlere göre hedef değişkenin ortalaması
#numerik değişkenler için hedef değişkene göre numerik değişkenlerin ortalaması hesaplanır.
def target_analyser(dataframe, target, num_cols, cat_cols):
    for col in dataframe.columns:
        if col in cat_cols:
            print(col, ":", len(dataframe[col].value_counts()))
            print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                                "RATIO": dataframe[col].value_counts() / len(dataframe),
                                "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
        if col in num_cols:
            print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(target)[col].mean()}), end="\n\n\n")

target_analyser(df, "Outcome", num_cols, cat_cols)

#ADIM-5 Aykırı gözlem analizinin yapılması

#df için alt ve üst çeyreklere göre aykırı değelerin olup olmadığını buluyoruz
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df,num_cols)

#grab_outliers ile aykırı değeri olan kolonları ve hangi veriler olduğunu buluyoruz
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print("#####################################################")
        print(str(col_name) + " variable have too much outliers")
        print("#####################################################")
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head(15))
        print("#####################################################")
        print("Lower threshold: " + str(low) + "   Lowest outlier: " + str(dataframe[col_name].min()) +
              "   Upper threshold: " + str(up) + "   Highest outlier: " + str(dataframe[col_name].max()))
        print("#####################################################")
    elif (dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] < 10) & \
            (dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 0):
        print("#####################################################")
        print(str(col_name) + " variable have less than 10 outlier values")
        print("#####################################################")
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
        print("#####################################################")
        print("Lower threshold: " + str(low) + "   Lowest outlier: " + str(dataframe[col_name].min()) +
              "   Upper threshold: " + str(up) + "   Highest outlier: " + str(dataframe[col_name].max()))
        print("#####################################################")
    else:
        print("#####################################################")
        print(str(col_name) + " variable does not have outlier values")
        print("#####################################################")

    if index:
        print(str(col_name) + " variable's outlier indexes")
        print("#####################################################")
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df,num_cols)

#ADIM-6 Eksik gözlem analizinin yapılması

#Veride eksik gözlem olup olmadığını kontrol ediyoruz.
df.isnull().sum()

#eksik gözlem yok ancak kimi değişkenler içinde 0 değerleri var bunları bulalım
for col in df.columns:
    print(col + " : " + str((df[f"{col}"] == 0).sum()))

#elde edilen değerler içinde Glucose, BloodPressure, SkinThickness, Insulin, BMI ve
# DiabetesPedigreeFunction sütunları içinde 0 değeri olamayacağından bu gözlemleri eksik gözleme
# dönüştürelim.
df['Glucose'] = df['Glucose'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)

#Dönüştürülen eksik gözlemleri ortalama değerler ile dolduralım
df["Glucose"]=df['Glucose'].fillna(df['Glucose'].mean())
df['BMI']=df['BMI'].fillna(df['BMI'].mean())
df['SkinThickness']=df['SkinThickness'].fillna(df['SkinThickness'].mean())
df['Insulin']=df['Insulin'].fillna(df['Insulin'].mean())
df['BloodPressure']=df['BloodPressure'].fillna(df['BloodPressure'].mean())

#ADIM-6 Korelasyon analizinin yapılması

#sns_heatmap fonksiyonu ile değişkenlerin birbiri arasındaki korelasyon değerlerine baktık
def sns_heatmap(dataset, color):
    heatmap = sns.heatmap(dataset.corr(), vmin=-1, vmax=1, annot=True, cmap=color)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
    plt.show()

sns_heatmap(df,color='Greens')

#############################################################
# GÖREV - 2
#############################################################

#ADIM-2 Yeni değişkenlerin oluşturulması

#korelasyon değerlerinden yola çıkarak yaş ve vücut kitle indeksi değişkenini çarparak yeni bir değişen oluşturalım
#Öncelikle bu iki değeri (1-5) arasında standartlaştıralım:
scaler = MinMaxScaler(feature_range=(1, 5))
ages = scaler.fit_transform(df[['Age']])
bmi = scaler.fit_transform(df[["BMI"]])
age_BMI=ages*bmi
df["age_BMI"]=pd.DataFrame(age_BMI)

#Yaş,doğum sayısı ve Glikoz miktarlarını segmentelere ayıralım:
df.loc[(df["Age"]) <= 35.0, 'New_Age'] = 'young'
df.loc[(df["Age"] > 35.0) & (df["Age"] <= 45.0), 'New_Age'] = 'middle'
df.loc[(df["Age"] > 45.0) & (df["Age"] <= 55.0), 'New_Age'] = 'mature'
df.loc[(df["Age"] > 55.0), 'New_Age'] = 'old'

df.loc[(df["Pregnancies"] <= 3), 'New_Preg'] = 'less&equal_3'
df.loc[(df["Pregnancies"] > 3) & (df["Pregnancies"] <= 6), 'New_Preg'] = 'less&equal_6'
df.loc[(df["Pregnancies"] > 6), 'New_Preg'] = 'more_than_6'

df.loc[(df["Glucose"] <= 140) , 'New_Glucose'] = 'normal'
df.loc[(df["Glucose"] > 140), 'New_Glucose'] = 'high_level'

#ADIM-3 Encoding işlemlerinin gerçekleştirilmesi

#yeni oluşturulduğumuz değişkenler sonrası dfi tekrar kategorik, numerik, kardinal ayırdık
cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Hücrelerinde sadece iki değişken olan sütunları bulalım
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

#label_encoder ile binary kolonları 0-1 olarak değiştiriyoruz.
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    label_encoder(df, col)

#Bağımlı değişken üzerinde etkisi az(0,01) olan  değişkenleri siliyoruz

#rare_analyser ile bağımlı değişkenle kategorik değişkenler arasındaki ilişki analiz ediliyor
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df,"Outcome",cat_cols)

def rare_encoder(dataframe, rare_perc, cat_cols):
    # 1'den fazla rare varsa düzeltme yap. durumu göz önünde bulunduruldu.
    # rare sınıf sorgusu 0.01'e göre yapıldıktan sonra gelen true'ların sum'ı alınıyor.
    # eğer 1'den büyük ise rare cols listesine alınıyor.
    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) < 0.01).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), 'Rare', dataframe[col])

    return dataframe

rare_encoder(df,0.01,cat_cols)

#2den daha fazla string içeren karegorik değişkenleri drop_first=True yani
#dummy değişken tuzağına düşmeden 0-1 olarak dönüştürdük.
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df, ohe_cols)
df.head()

#oluşan yeni değişkenler sonrası veristeini tekrar kategorik numerik ve kardinal ayıralım:
cat_cols, num_cols, cat_but_car = grab_col_names(df)

#ADIM-4 Numerik değişkenlere standartlaştırma yapılması

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

#ADIM-5 Modelin kurulması

#Bağımlı yani hedef değişkeni tanımladık
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

#train ve test setini 70e 30 ayırdık
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

#radomforest ile model kurduk
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

#Hangi değişkenin modelde tahminde daha çok etki ettiğini bulmak için baktık ve
#yeni oluşturduğumuz age_BMI değişkenin 2.sırada yer aldığını böylelikle veriden
#yeni veriler türetmenin önemini gördük
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)





