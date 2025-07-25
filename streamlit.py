import streamlit as st
import pandas as pd
import plotly.express as px
import shap
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

 
st.set_page_config(page_title="Марченко_Егор_Владиславович_2023-ФГиИБ-ПИ-1б_Вариант_15_Ценовой_диапазон_телефонов")
 
st.title("Марченко_Егор_Владиславович_2023_ФгиИБ_ПИ_1б_15_Ценовой_диапазон_телефонов")
data = pd.read_csv('phones1.csv')
 
#описаниедатасета
st.header("Описание набора данных")
st.write("Данныйдатасет представляет собой информацию о характеристиках и функционале телефонов, а так же их ценовому диапазону. Предоставленные данные: "\
         "battery_power — емкостьаккумулятора,"\
         " blue — наличие Bluetooth,"\
         " clock_speed — тактовая частота процессора,"\
         " dual_sim — поддержка двух SIM-карт,"\
         " fc — разрешение фронтальнойкамеры,"\
         " four_g — поддержка 4G,"\
         " int_memory — внутренняя память,"\
         " m_dep — толщина телефона,"\
         " mobile_wt — вес телефона,"\
         " n_cores — количество ядер процессора,"\
         " pc — разрешение основной (задней) камеры,"\
         " px_height — высота экрана в пикселях,"\
         " px_width — ширина экрана в пикселях,"\
         " ram — объем оперативной памяти,"\
         " sc_h — высота экрана в мм,"\
         " sc_w — ширина экрана в мм,"\
         " talk_time — максимальное время разговора на одном заряде,"\
         " three_g — поддержка 3G,"\
         " touch_screen — наличие сенсорного экрана,"\
         " поддержка Wi-Fi,"\
         " price_range — диапазонцены.")

X= data[['CatRam', 'Perfomance', 'SqPx', 'battery_power_scaled']]
Y= data['price_range']
X_train,X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
 
model= DecisionTreeClassifier(
    max_depth=5,          
    min_samples_split=10, 
    min_samples_leaf=5,   
    random_state=42
)
model.fit(X_train,
y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
 
#точность модели
st.header("Точность модели")
st.write(f"Точность модели: {accuracy:.2f}")
 
#график1
st.header("График1: Распределение объема оперативной памяти по ценовым категориям")
fig1 = px.histogram(data, x="CatRam", color="price_range",
                   barmode='group', nbins=15,
                  color_discrete_sequence=['green','yellow','blue','red'],
                   title="Распределение оперативной памяти по ценовым категориям")
fig1.update_layout(xaxis_title="Категория RAM", yaxis_title="Количество")
st.plotly_chart(fig1)

#график 2
st.header("График2: Распределение размера экрана в пикселях по ценовым категориям")
fig2 = px.histogram(data, x="SqrtScr", color="price_range",barmode='group', nbins=5, title="Распределениеразмера экрана по ценовым категориям")
fig2.update_layout(xaxis_title="Размер экрана в пикселях", yaxis_title="Количество")
st.plotly_chart(fig2)

 
#график3 SHAP
st.header("График 3: Важность признаков для модели (SHAP values)")
 
explainer = shap.Explainer(model, X_train)
shap_values= explainer(X_test)
 
mean_shap_values = shap_values.values.mean(axis=2)
 
aggregated_explanation= shap.Explanation( 
    values=mean_shap_values,   
    base_values=shap_values.base_values.mean(axis=1),
    data=X_test.values,
    feature_names=X_test.columns.tolist()    
)
 
#Display SHAP beeswarm plot
plt.figure()
shap.plots.beeswarm(aggregated_explanation)
st.pyplot(plt.gcf())
plt.close()
#Display SHAP beeswarm plot
st.write("""
Beeswarmplot показывает важность каждого признака для модели. 
Признаки упорядочены по важности, где верхние признаки оказывают наибольшее влияние напредсказание. 
Цветпоказывает значение признака (красный - высокое, синий - низкое).
""")
