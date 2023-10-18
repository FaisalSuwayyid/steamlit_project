import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
# import hiplot as hip
# from sklearn import datasets


from PIL import Image



######################################################################



df = pd.read_csv('https://raw.githubusercontent.com/FaisalSuwayyid/CMSE830/main/Mushroom.csv', index_col=0)
df = df[['target',
 'cap-shape',
 'cap-surface',
 'cap-color',
 'odor',
 'gill-attachment', #
 'gill-spacing', #
 'gill-color',
 'stalk-surface-above-ring',
 'stalk-surface-below-ring',
 'stalk-color-above-ring',
 'stalk-color-below-ring',
 'veil-color', #
 'spore-print-color',
 'population',
 'habitat']]

# df = pd.read_csv('/Users/faisalsuwayyid/Desktop/Projects/CMSE_Midterm/The project/7_Mushroom/Mushroom.csv')
# df = df.iloc[:,1:]

st.set_page_config(page_title = 'Classification of Mushrooms to Edible and Poisonous',
                   page_icon = 'bar_chart',
                   layout='wide')

col1, col2, col3 = st.columns([0.1, 0.89, 0.01])
with col2:
    st.title('Classification of Mushrooms to Edible and Poisonous')

image = Image.open('AdobeStock_321292924_Preview copy.jpg')

st.image(image, caption='Mushroom Anatomy')

st.markdown('#')
st.markdown('#')
st.markdown('##### In this application, we have a dataset \
            containing features characterizing mushrooms,\
            and the goal is to classify mushrooms to poisonous and edible. \
            There are 22 features characterizing 8124 different mushrooms. Therefore, we \
            believe we can reduce the number of features without affecting the data. First, \
            when we drop the target column, we find\
            that there is no two samples with the same features. Therefore, we can always tell if \
            a mushroom is poisonous or not with absolute certainty. \
            Carrying out the same process with every feature, we are able to reduce the number of \
            features to 15.')
st.markdown('**The dataset is obtained from UCI Machine Learning Repository, called Mushrooms.**')
st.markdown('We refer to https://rrcultivation.com/blogs/mn/mushroom-anatomy-caps-stems \
            to understand the mushroom anatomy.')
st.markdown('#')
st.markdown('#')





####### Data Preview #########
with st.expander('Data Preview'):
    # st.write(df)
    st.dataframe(df)
    # st.dataframe(df,
    #              column_config={
    #                  'Target': st.column_config.NumberColumn(format='%d')
    #              })
###############################
st.markdown('#')
st.markdown('#')
st.markdown('##### In the below plot, we notice that we have poisonous \
             mushrooms almost as much as edible ones. \
            This suggests that, under the assumption of a uniform distribution, \
            it is very likely to encounter a poisonous mushrooms.')
st.markdown('#')
st.markdown('#')
####### Data Validation #########
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown('# Size of data')

option_data_validation = st.selectbox(
   "Select a feature to view its histogram",
   tuple([None, 'percent']),
   index=0,
   placeholder="Select contact method...",
)
fig = px.histogram(df, x="target", text_auto=True,
                    histnorm = option_data_validation, histfunc = 'count')
st.plotly_chart(fig, use_container_width=True)




###############################

st.markdown('##### We investigate the histogram of each \
            feature to understand if there is any deterministic \
            feature that characterizes mushrooms to edible or poisonous \
            and how mushrooms are distributed over classes of the feature selected. \
            Generally, we look for features where edible and poisonous have less common \
            classes. Additionally, if possible, we also look for the cases where \
            one of the two poisonous or edible is very few in the common classes.')
############ Histogram count ###############
col1, col2, col3 = st.columns([0.2, 0.7, 0.1])
with col2:
    st.markdown('# The percentage of classes in features')
fig = plt.figure(figsize=(6, 4))
s = df.columns.tolist()
s = [ele for ele in s if ele != 'target']
# s = s.remove('target')
option_data_histogram = st.selectbox(
   "Select a feature to view its histogram",
   tuple(s),
   index=0,
   placeholder="Select contact method...",
)


x1 = df[option_data_histogram].unique().tolist()
fig2=plt.figure(figsize=(6,4), dpi=250)
ax=fig2.add_subplot(111)

sns.histplot(data=df, x=option_data_histogram,
              hue='target', multiple='stack', stat='percent', ax=ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_color('#DDDDDD')
plt.xticks(rotation=45)
plt.title(f'Percentage of classes in {option_data_histogram}')
st.pyplot(fig2)



###############################

#### Relative Histogram #####
st.markdown('#')
st.markdown('##### The below plot shows the percentage of both \
            poisonous and edible mushrooms in the same class\
            in the selected feature.')
# relative_histplot(df, option)

fig = px.histogram(df, x=option_data_histogram,
                    text_auto=True, color = 'target', barnorm = 'percent')
st.plotly_chart(fig, use_container_width=True)

##############################
st.markdown('#')
st.markdown('#')
st.markdown('##### Going over the features, we observe that \
            odor, spore-print-color, veil-color and population satisfy our criteria. \
            odor alone seems like a good start as it shows most of the poisonous mushrooms \
            have odor, and very few are with no odor.')
st.markdown('#')
st.markdown('#')

col1, col2, col3 = st.columns(3)
with col2:
    st.markdown('# Pair of features study')
st.markdown('#')
st.markdown('#')
st.markdown('##### Here, we investigate the distribution and probability of selected pair of features. \
            The below table shows the probablities of the pair of classes of the selected two features.\
            The left table shows the probability of edible mushrooms over the pair of classes of\
            the pair of selected features. The right table shows the ones for the poisonous mushrooms.\
            Zero percentage corresponding to a pair of classes in the left table \
            implies that all mushrooms with such pair of classes are \
            poisonous, if any, and vice versa.')
st.markdown('#')
st.markdown('#')

s = df.columns.tolist()
s = [ele for ele in s if ele != 'target']

option_feature1 = st.selectbox(
   "Feature 1",
   tuple(s),
   index=0,
   placeholder="Select contact method...",
)
s = [ele for ele in s if ele != option_feature1]
option_feature2 = st.selectbox(
   "Feature 2",
   tuple(s),
   index=0,
   placeholder="Select contact method...",
)

df_p = df[df['target']=='poisonous']
df_e = df[df['target']=='edible']

feature1 = option_feature1
feature2 = option_feature2

x = df[[feature1, feature2]].value_counts()
feature1_classes = df[feature1].unique()
feature2_classes = df[feature2].unique()



a = np.zeros((len(feature1_classes), len(feature2_classes)))
# df[[feature1, feature2]].value_counts()
for i, labeli in zip(range(len(feature1_classes)), feature1_classes):
    for j, labelj in zip(range(len(feature2_classes)), feature2_classes):
        if labelj in x[labeli].index:
            a[i][j] = x[labeli][labelj]

a = a/df.shape[0]*100
A = pd.DataFrame(a, columns = feature2_classes, index = feature1_classes)

a_e = np.zeros_like(a)

x_e = df_e[[feature1, feature2]].value_counts()
feature1_classes_e = df_e[feature1].unique()

for i, labeli in zip(range(len(feature1_classes)), feature1_classes):
    for j, labelj in zip(range(len(feature2_classes)), feature2_classes):
        if labeli in feature1_classes_e:
            if labelj in x_e[labeli].index:
                a_e[i][j] = x_e[labeli][labelj]
            
a_e = a_e/df.shape[0]*100
A_e = pd.DataFrame(a_e, columns = feature2_classes, index = feature1_classes)

a_p = a - a_e
A_p = A - A_e

with st.expander('Probability Table of the two selected features'):
    # st.write(df)
    st.dataframe(A)
    # st.dataframe(df,
    #              column_config={
    #                  'Target': st.column_config.NumberColumn(format='%d')
    #              })

col1, col2 = st.columns([0.5, 0.5])
with col1:
    with st.expander('Probability table of the edible ones'):
    # st.write(df)
        st.dataframe(A_e)
with col2:
    with st.expander('Probability table of the poisonous ones'):
    # st.write(df)
        st.dataframe(A_p)


st.markdown('#')
st.markdown('#')
st.markdown('##### After experimenting, we conclude that (odor, spore-print-color) \
            is good as there is only one common pair between edible and poisonous \
            with nonzero probability. Generally, using pairs of features reduces the probability \
            significantly.')
st.markdown('#')
st.markdown('#')








# ######### Sunburst1 plot ############
st.markdown('#')
st.markdown('#')
st.markdown('##### We use a sunburst plot to display a 4 dimensional plot to help in understanding \
            how three features can classify mushrooms. We are mainly interested in \
            odor, spore-print-color, and veil color or population.')
st.markdown('#')
st.markdown('#')
col1, col2, col3 = st.columns(3)
with col1:
    s = df.columns.tolist()
    s = [ele for ele in s if ele != 'target']

    option_first_sunburst1_feature = st.selectbox(
    "Select first feature for sunburst plot",
    tuple(s),
    index=0,
    placeholder="Select contact method...",
    )
    s = [ele for ele in s if ele != 'target' and \
          ele != option_first_sunburst1_feature]
    option_second_sunburst1_feature = st.selectbox(
    "Select second feature for sunburst plot",
    tuple(s),
    index=0,
    placeholder="Select contact method...",
    )
    s = [ele for ele in s if ele != 'target' and \
          ele != option_first_sunburst1_feature and ele != option_second_sunburst1_feature]
    option_third_sunburst1_feature = st.selectbox(
    "Select third feature for sunburst plot",
    tuple(s),
    index=0,
    placeholder="Select contact method...",
    )
    fig = px.sunburst(df, path=[option_first_sunburst1_feature,
    option_second_sunburst1_feature,
    option_third_sunburst1_feature,
    'target'])
with col2:
    st.plotly_chart(fig)

# hip.Experiment.from_dataframe(df).display().to_streamlit()

#############################################



# ######### Sunburst2 plot ############

# df2 = df[df['odor']=='none']

# st.markdown('#')
# st.markdown('#')
# st.markdown('##### After trying several combination, we found that it is possible \
#             to classify mushrooms to edible and poisonous using only seven features.')
# st.markdown('#')
# st.markdown('#')

# fig = px.sunburst(df, path = df.columns.tolist())
# with col2:
#     st.plotly_chart(fig)

# hip.Experiment.from_dataframe(df).display().to_streamlit()

#############################################













# ####### select feature to study independently ######

col1, col2, col3 = st.columns([0.1, 0.89, 0.01])
with col2:
    st.markdown('# Relative statistics of a selected class in a feature study')
    st.markdown('#')

st.markdown('#')
st.markdown('#')
st.markdown('##### This study is useful to plot and understand 5 dimensional \
            data when repeating the previous process. We are mainly interested in \
            studying the case when the selected feature is odor, and the \
            selected class is none.')
st.markdown('#')
st.markdown('#')

s = df.columns.tolist()
s = [ele for ele in s if ele != 'target']
option_specific_feature = st.selectbox(
   "Select a feature to study its case",
   tuple(s),
   index=0,
   placeholder="Select contact method...",
)

option_specific_class = st.selectbox(
   "Select a category in that feature",
   tuple(df[option_specific_feature].unique()),
   index=0,
   placeholder="Select contact method...",
)


df_specific = df[df[option_specific_feature]==option_specific_class]
df_specific = df_specific.drop(columns=[option_specific_feature])


# ####### Data Validation #########
# fig = plt.figure(figsize=(6, 4))
# sns.histplot(data =df_specific, x='target', hue='target')

# # st.pyplot(fig1, theme="streamlit", use_container_width=True)
# st.pyplot(fig)
# # st.write(fig1)


option_feature_histogram = st.selectbox(
   "Select histogram type",
   tuple([None, 'percent']),
   index=0,
   placeholder="Select contact method...",
)
fig = px.histogram(df_specific, x="target", text_auto=True, histnorm = option_feature_histogram, histfunc = 'count')
st.plotly_chart(fig, use_container_width=True)



###############################
s = df_specific.columns.tolist()
s = [ele for ele in s if ele != 'target' and option_specific_feature]
option4 = st.selectbox(
   "Select a feature to view its histogram",
   tuple(s),
   index=0,
   placeholder="Select contact method...",
)

x1 = df_specific[option4].unique().tolist()
fig2=plt.figure(figsize=(6,4), dpi=250)
ax=fig2.add_subplot(111)

sns.histplot(data=df_specific, x=option4, hue='target', multiple='stack', stat='percent', ax=ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_color('#DDDDDD')
plt.xticks(rotation=45)
st.pyplot(fig2)


# relative_histplot(df_specific, option4)

fig = px.histogram(df_specific, x=option4, text_auto=True, color = 'target', barnorm = 'percent')
st.plotly_chart(fig, use_container_width=True)
#########################################












s = df_specific.columns.tolist()
s = [ele for ele in s if ele != 'target']

option_feature1_specific = st.selectbox(
   "Feature_sp 1",
   tuple(s),
   index=0,
   placeholder="Select contact method...",
)
s = [ele for ele in s if ele != option_feature1_specific]
option_feature2_specific = st.selectbox(
   "Feature_sp 2",
   tuple(s),
   index=0,
   placeholder="Select contact method...",
)

df_specific_p = df_specific[df_specific['target']=='poisonous']
df_specific_e = df_specific[df_specific['target']=='edible']

feature1 = option_feature1_specific
feature2 = option_feature2_specific

x = df_specific[[feature1, feature2]].value_counts()
feature1_classes = df_specific[feature1].unique()
feature2_classes = df_specific[feature2].unique()



a = np.zeros((len(feature1_classes), len(feature2_classes)))
# df[[feature1, feature2]].value_counts()
for i, labeli in zip(range(len(feature1_classes)), feature1_classes):
    for j, labelj in zip(range(len(feature2_classes)), feature2_classes):
        if labelj in x[labeli].index:
            a[i][j] = x[labeli][labelj]

a = a/df_specific.shape[0]*100
A = pd.DataFrame(a, columns = feature2_classes, index = feature1_classes)

a_e = np.zeros_like(a)

x_e = df_specific_e[[feature1, feature2]].value_counts()
feature1_classes_e = df_specific_e[feature1].unique()

for i, labeli in zip(range(len(feature1_classes)), feature1_classes):
    for j, labelj in zip(range(len(feature2_classes)), feature2_classes):
        if labeli in feature1_classes_e:
            if labelj in x_e[labeli].index:
                a_e[i][j] = x_e[labeli][labelj]
            
a_e = a_e/df.shape[0]*100
A_e = pd.DataFrame(a_e, columns = feature2_classes, index = feature1_classes)

a_p = a - a_e
A_p = A - A_e

with st.expander('Probability Table of the two selected features'):
    # st.write(df)
    st.dataframe(A)
    # st.dataframe(df,
    #              column_config={
    #                  'Target': st.column_config.NumberColumn(format='%d')
    #              })

col1, col2 = st.columns([0.5, 0.5])
with col1:
    with st.expander('Probability table of the edible ones'):
    # st.write(df)
        st.dataframe(A_e)
with col2:
    with st.expander('Probability table of the poisonous ones'):
    # st.write(df)
        st.dataframe(A_p)










# ######### Sunburst2 plot ############

col1, col2, col3 = st.columns(3)
with col1:
    s = df_specific.columns.tolist()
    s = [ele for ele in s if ele != 'target']

    option_first_sunburst2_feature = st.selectbox(
    "Select first feature for sunburst plot",
    tuple(s),
    index=0,
    placeholder="Select contact method...",
    )
    s = [ele for ele in s if ele != 'target' and \
          ele != option_first_sunburst2_feature]
    option_second_sunburst2_feature = st.selectbox(
    "Select second feature for sunburst plot",
    tuple(s),
    index=0,
    placeholder="Select contact method...",
    )
    s = [ele for ele in s if ele != 'target' and \
          ele != option_first_sunburst2_feature and ele != option_second_sunburst2_feature]
    option_third_sunburst2_feature = st.selectbox(
    "Select third feature for sunburst plot",
    tuple(s),
    index=0,
    placeholder="Select contact method...",
    )
    fig = px.sunburst(df_specific, path=[option_first_sunburst2_feature,
    option_second_sunburst2_feature,
    option_third_sunburst2_feature,
    'target'])
with col2:
    st.plotly_chart(fig)

# hip.Experiment.from_dataframe(df).display().to_streamlit()

#############################################




st.markdown('#')
st.markdown('#')
st.markdown('##### In conclusion, we find that to classify mushrooms \
            to edible and poisonous, it is sufficient to use seven features \
            selected in a good way. One such group of seven features is \
            (odor, spore-print-color, veil-color, stalk-color-above-ring, \
            stalk-surface-below-ring, and ring-type.)')
st.markdown('#')
st.markdown('#')
