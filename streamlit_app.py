import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np


######################################################################

# pre-defined functions

##### relative hist plots ####

def relative_histplot(df, x):
    s = df.columns.tolist()
    s.remove('target')

    

    # print(x)
    y = 'target'

    x1 = df[x].unique().tolist()
    y1 = df[y].unique().tolist()

    a = df[[x, y]].value_counts()

    A = np.zeros((len(x1), len(y1)))

    for i, labeli in zip(range(len(x1)), x1):
        for j, labelj in zip(range(len(y1)), y1):
            if labelj in a[labeli].index:
                A[i][j] = a[labeli][labelj]


    # memo of sample number
    snum = np.sum(A, axis=1)


    # normalization
    for i in range(A.shape[0]):
        A[i] = A[i]/snum[i]*100.

    A = A.T
    fig=plt.figure(figsize=(6,4), dpi=250)
    ax=fig.add_subplot(111)


    # stack bars
    for i in range(A.shape[0]):
            plt.bar(x1, A[i], bottom = 0.6*i + np.sum(A[:i], axis = 0), label=y1[i], width=0.8,  linewidth=0.1)


    # add text annotation corresponding to the percentage of each data.

    for i in range(A.shape[0]):
        if i==0:
            for xpos, ypos, yval in zip(x1, A[i]/2, A[i]):
                plt.text(xpos, ypos, "N=%.2f"%yval, ha="center", va="center")
        else :
            for xpos, ypos, yval in zip(x1, np.sum(A[:i], axis=0) + A[i]/2, A[i]):
                plt.text(xpos, ypos, "N=%.2f"%yval, ha="center", va="center")
    # for xpos, ypos, yval in zip(x1, 0.6 + np.sum(A[:], axis=0), np.sum(A[:], axis=0)):
    #     plt.text(xpos, ypos, "N=%.2f"%yval, ha="center", va="bottom") 

    # plt.ylim(0,110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_color('#DDDDDD')

    plt.xticks(rotation=45)
    plt.xlabel(x)
    plt.ylabel('relative percentage')
    plt.title('relative percentage of poisonous and edible in each category')
    plt.legend(bbox_to_anchor=(1.01,0.5), loc='center left')
    # plt.show()
    st.pyplot(fig)




##############################

# st.plotly_chart(figure_or_data, use_container_width=False, sharing="streamlit", theme="streamlit", **kwargs)
# st.pyplot(fig=None, clear_figure=None, use_container_width=True, **kwargs)
#####################################################################





df = pd.read_csv('https://raw.githubusercontent.com/FaisalSuwayyid/CMSE830/main/Mushroom.csv', index_col=0)


# df = pd.read_csv('/Users/faisalsuwayyid/Desktop/Projects/CMSE_Midterm/The project/7_Mushroom/Mushroom.csv')
# df = df.iloc[:,1:]

st.set_page_config(page_title = 'Classification of Mushrooms to Edible and Poisonous',
                   page_icon = 'bar_chart',
                   layout='wide')
st.title('Classification of Mushrooms to Edible and Poisonous')







####### Data Preview #########
with st.expander('Data Preview'):
    # st.write(df)
    st.dataframe(df)
    # st.dataframe(df,
    #              column_config={
    #                  'Target': st.column_config.NumberColumn(format='%d')
    #              })
###############################

####### Data Validation #########
fig = plt.figure(figsize=(6, 4))
sns.histplot(df, x='target', hue='target')

# st.pyplot(fig1, theme="streamlit", use_container_width=True)
st.pyplot(fig)
# st.write(fig1)
###############################


############ Histogram count ###############
s = df.columns.tolist()
s = [ele for ele in s if ele != 'target']
# s = s.remove('target')
option = st.selectbox(
   "Select a feature to view its histogram",
   tuple(s),
   index=0,
   placeholder="Select contact method...",
)


x1 = df[option].unique().tolist()
fig2=plt.figure(figsize=(6,4), dpi=250)
ax=fig2.add_subplot(111)

sns.histplot(data=df, x=option, hue='target', multiple='stack', stat='percent', ax=ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_color('#DDDDDD')
plt.xticks(rotation=45)
st.pyplot(fig2)

###############################

#### Relative Histogram #####

relative_histplot(df, option)


##############################











# ######### Sunburst plot ############
col1, col2, col3 = st.columns(3)
with col1:
    s = df.columns.tolist()
    s = [ele for ele in s if ele != 'target']

    option5 = st.selectbox(
    "Select first feature for sunburst plot",
    tuple(s),
    index=0,
    placeholder="Select contact method...",
    )
    s = [ele for ele in s if ele != 'target' and ele != option5]
    option6 = st.selectbox(
    "Select second feature for sunburst plot",
    tuple(s),
    index=0,
    placeholder="Select contact method...",
    )
    s = [ele for ele in s if ele != 'target' and ele != option5 and ele != option6]
    option7 = st.selectbox(
    "Select third feature for sunburst plot",
    tuple(s),
    index=0,
    placeholder="Select contact method...",
    )
    fig = px.sunburst(df, path=[option5,
    option6,
    option7,
    'target'])
with col2:
    st.plotly_chart(fig)

# hip.Experiment.from_dataframe(df).display().to_streamlit()

#############################################













# ####### select feature to study independently ######
s = df.columns.tolist()
s = [ele for ele in s if ele != 'target']
option2 = st.selectbox(
   "Select a feature to study its case",
   tuple(s),
   index=0,
   placeholder="Select contact method...",
)

option3 = st.selectbox(
   "Select a category in that feature",
   tuple(df[option2].unique()),
   index=0,
   placeholder="Select contact method...",
)


df_specific = df[df[option2]==option3]
df_specific = df_specific.drop(columns=[option2])


####### Data Validation #########
fig = plt.figure(figsize=(6, 4))
sns.histplot(data =df_specific, x='target', hue='target')

# st.pyplot(fig1, theme="streamlit", use_container_width=True)
st.pyplot(fig)
# st.write(fig1)
###############################
s = df_specific.columns.tolist()
s = [ele for ele in s if ele != 'target' and option2]
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



relative_histplot(df_specific, option4)

#########################################
